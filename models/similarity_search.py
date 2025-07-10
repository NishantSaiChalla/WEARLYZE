"""
Similarity Search Module for Fashion Detection.

This module provides comprehensive similarity search capabilities using FAISS
for efficient approximate nearest neighbor search. It includes support for
IVF-PQ indexing, batch processing, and both text and image queries.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import pickle
import json
from dataclasses import dataclass, asdict
import time
from collections import defaultdict
import warnings

try:
    import faiss
except ImportError:
    warnings.warn("FAISS not installed. Please install with: pip install faiss-cpu or faiss-gpu")
    faiss = None

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import h5py

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for similarity search."""
    
    # Index parameters
    embedding_dim: int = 512
    index_type: str = "IVF-PQ"  # "Flat", "IVF", "IVF-PQ", "HNSW"
    
    # IVF parameters
    n_centroids: int = 1024
    n_probe: int = 32
    
    # PQ parameters
    pq_m: int = 64  # Number of subquantizers
    pq_nbits: int = 8  # Number of bits per subquantizer
    
    # HNSW parameters
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50
    
    # Search parameters
    k_neighbors: int = 100
    use_gpu: bool = True
    gpu_id: int = 0
    
    # Quality parameters
    target_recall: float = 0.99
    use_reranking: bool = True
    rerank_k: int = 1000
    
    # Batch processing
    batch_size: int = 1000
    max_memory_gb: float = 8.0


class VectorDatabase:
    """
    High-performance vector database for fashion similarity search.
    
    Supports multiple indexing strategies including IVF-PQ for efficient
    approximate nearest neighbor search with configurable recall targets.
    """
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.index = None
        self.embeddings = None
        self.metadata = {}
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.is_trained = False
        
        # Initialize FAISS resources
        if faiss is not None and config.use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()
            self.gpu_resources.setDefaultNullStreamAllDevices()
        else:
            self.gpu_resources = None
        
        logger.info(f"VectorDatabase initialized with {config.index_type} index")
    
    def create_index(self, embeddings: np.ndarray) -> None:
        """
        Create and train the search index.
        
        Args:
            embeddings: Embedding vectors to index
        """
        if faiss is None:
            raise ImportError("FAISS not available. Please install faiss-cpu or faiss-gpu")
        
        n_vectors, dim = embeddings.shape
        if dim != self.config.embedding_dim:
            raise ValueError(f"Embedding dimension {dim} doesn't match config {self.config.embedding_dim}")
        
        logger.info(f"Creating {self.config.index_type} index for {n_vectors} vectors of dimension {dim}")
        
        # Normalize embeddings for cosine similarity
        embeddings = normalize(embeddings, norm='l2', axis=1)
        
        # Create index based on type
        if self.config.index_type == "Flat":
            index = faiss.IndexFlatIP(dim)
        elif self.config.index_type == "IVF":
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, self.config.n_centroids)
        elif self.config.index_type == "IVF-PQ":
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFPQ(
                quantizer, dim, self.config.n_centroids, 
                self.config.pq_m, self.config.pq_nbits
            )
        elif self.config.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(dim, self.config.hnsw_m)
            index.hnsw.efConstruction = self.config.hnsw_ef_construction
            index.hnsw.efSearch = self.config.hnsw_ef_search
        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")
        
        # Set search parameters
        if hasattr(index, 'nprobe'):
            index.nprobe = self.config.n_probe
        
        # Move to GPU if available
        if self.config.use_gpu and self.gpu_resources is not None:
            index = faiss.index_cpu_to_gpu(self.gpu_resources, self.config.gpu_id, index)
        
        # Train index if necessary
        if not index.is_trained:
            logger.info("Training index...")
            index.train(embeddings.astype(np.float32))
        
        # Add vectors to index
        logger.info("Adding vectors to index...")
        index.add(embeddings.astype(np.float32))
        
        self.index = index
        self.embeddings = embeddings
        self.is_trained = True
        
        logger.info(f"Index created successfully with {index.ntotal} vectors")
    
    def add_embeddings(self, 
                      embeddings: np.ndarray, 
                      ids: List[str],
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add embeddings to the database.
        
        Args:
            embeddings: Embedding vectors
            ids: Unique identifiers for each embedding
            metadata: Optional metadata for each embedding
        """
        if len(embeddings) != len(ids):
            raise ValueError("Number of embeddings must match number of IDs")
        
        # Normalize embeddings
        embeddings = normalize(embeddings, norm='l2', axis=1)
        
        # Update ID mappings
        start_idx = len(self.id_to_idx)
        for i, id_val in enumerate(ids):
            idx = start_idx + i
            self.id_to_idx[id_val] = idx
            self.idx_to_id[idx] = id_val
        
        # Store metadata
        if metadata:
            for id_val, meta in metadata.items():
                self.metadata[id_val] = meta
        
        # Create or update index
        if self.index is None:
            self.create_index(embeddings)
        else:
            if self.config.use_gpu and self.gpu_resources is not None:
                # Move index to CPU for adding vectors
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                cpu_index.add(embeddings.astype(np.float32))
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.config.gpu_id, cpu_index)
            else:
                self.index.add(embeddings.astype(np.float32))
            
            # Update stored embeddings
            if self.embeddings is not None:
                self.embeddings = np.vstack([self.embeddings, embeddings])
            else:
                self.embeddings = embeddings
        
        logger.info(f"Added {len(embeddings)} embeddings to database")
    
    def search(self, 
               query_embeddings: np.ndarray,
               k: Optional[int] = None,
               return_distances: bool = True,
               return_metadata: bool = False) -> Dict[str, Any]:
        """
        Search for similar embeddings.
        
        Args:
            query_embeddings: Query vectors
            k: Number of neighbors to return
            return_distances: Whether to return distances
            return_metadata: Whether to return metadata
            
        Returns:
            Search results with IDs, distances, and optionally metadata
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        if k is None:
            k = self.config.k_neighbors
        
        # Normalize query embeddings
        query_embeddings = normalize(query_embeddings, norm='l2', axis=1)
        
        # Perform search
        start_time = time.time()
        
        if self.config.use_reranking and k > self.config.rerank_k:
            # First stage: retrieve more candidates
            search_k = min(self.config.rerank_k, self.index.ntotal)
            distances, indices = self.index.search(query_embeddings.astype(np.float32), search_k)
            
            # Second stage: rerank with exact cosine similarity
            reranked_results = self._rerank_results(query_embeddings, distances, indices, k)
            distances, indices = reranked_results['distances'], reranked_results['indices']
        else:
            distances, indices = self.index.search(query_embeddings.astype(np.float32), k)
        
        search_time = time.time() - start_time
        
        # Convert indices to IDs
        result_ids = []
        for query_indices in indices:
            query_ids = [self.idx_to_id.get(idx, None) for idx in query_indices if idx != -1]
            result_ids.append(query_ids)
        
        results = {
            'ids': result_ids,
            'search_time': search_time,
            'k': k
        }
        
        if return_distances:
            results['distances'] = distances.tolist()
        
        if return_metadata:
            result_metadata = []
            for query_ids in result_ids:
                query_metadata = [self.metadata.get(id_val, {}) for id_val in query_ids]
                result_metadata.append(query_metadata)
            results['metadata'] = result_metadata
        
        return results
    
    def _rerank_results(self, 
                       query_embeddings: np.ndarray,
                       distances: np.ndarray,
                       indices: np.ndarray,
                       final_k: int) -> Dict[str, np.ndarray]:
        """
        Rerank search results using exact cosine similarity.
        
        Args:
            query_embeddings: Query vectors
            distances: Initial distances
            indices: Initial indices
            final_k: Final number of results to return
            
        Returns:
            Reranked results
        """
        reranked_distances = []
        reranked_indices = []
        
        for i, (query_emb, query_indices) in enumerate(zip(query_embeddings, indices)):
            # Get candidate embeddings
            valid_indices = query_indices[query_indices != -1]
            if len(valid_indices) == 0:
                reranked_distances.append(np.array([]))
                reranked_indices.append(np.array([]))
                continue
            
            candidate_embeddings = self.embeddings[valid_indices]
            
            # Compute exact cosine similarities
            similarities = cosine_similarity(query_emb.reshape(1, -1), candidate_embeddings)[0]
            
            # Sort by similarity (descending)
            sorted_idx = np.argsort(similarities)[::-1]
            
            # Take top k
            top_k = min(final_k, len(sorted_idx))
            final_indices = valid_indices[sorted_idx[:top_k]]
            final_distances = 1 - similarities[sorted_idx[:top_k]]  # Convert to distance
            
            reranked_distances.append(final_distances)
            reranked_indices.append(final_indices)
        
        # Pad arrays to same length
        max_len = max(len(arr) for arr in reranked_distances)
        padded_distances = []
        padded_indices = []
        
        for dist, idx in zip(reranked_distances, reranked_indices):
            padded_dist = np.full(max_len, np.inf)
            padded_idx = np.full(max_len, -1)
            
            if len(dist) > 0:
                padded_dist[:len(dist)] = dist
                padded_idx[:len(idx)] = idx
            
            padded_distances.append(padded_dist)
            padded_indices.append(padded_idx)
        
        return {
            'distances': np.array(padded_distances),
            'indices': np.array(padded_indices)
        }
    
    def batch_search(self, 
                    query_embeddings: np.ndarray,
                    k: Optional[int] = None,
                    batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform batch search for large query sets.
        
        Args:
            query_embeddings: Query vectors
            k: Number of neighbors to return
            batch_size: Batch size for processing
            
        Returns:
            Aggregated search results
        """
        if k is None:
            k = self.config.k_neighbors
        if batch_size is None:
            batch_size = self.config.batch_size
        
        n_queries = len(query_embeddings)
        all_ids = []
        all_distances = []
        total_time = 0
        
        for i in range(0, n_queries, batch_size):
            batch_end = min(i + batch_size, n_queries)
            batch_queries = query_embeddings[i:batch_end]
            
            batch_results = self.search(
                batch_queries, k=k, return_distances=True, return_metadata=False
            )
            
            all_ids.extend(batch_results['ids'])
            all_distances.extend(batch_results['distances'])
            total_time += batch_results['search_time']
        
        return {
            'ids': all_ids,
            'distances': all_distances,
            'search_time': total_time,
            'k': k
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            'total_vectors': self.index.ntotal if self.index else 0,
            'embedding_dim': self.config.embedding_dim,
            'index_type': self.config.index_type,
            'is_trained': self.is_trained,
            'metadata_count': len(self.metadata)
        }
        
        if self.index and hasattr(self.index, 'nprobe'):
            stats['nprobe'] = self.index.nprobe
        
        return stats
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the database to disk.
        
        Args:
            path: Directory path to save the database
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save index
        if self.index is not None:
            if self.config.use_gpu and self.gpu_resources is not None:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(path / "index.faiss"))
            else:
                faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(str(path / "embeddings.npy"), self.embeddings)
        
        # Save metadata and mappings
        with open(path / "metadata.json", "w") as f:
            json.dump({
                'metadata': self.metadata,
                'id_to_idx': self.id_to_idx,
                'idx_to_id': self.idx_to_id,
                'is_trained': self.is_trained
            }, f, indent=2)
        
        # Save configuration
        with open(path / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"Database saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'VectorDatabase':
        """
        Load database from disk.
        
        Args:
            path: Directory path containing the database
            
        Returns:
            Loaded VectorDatabase instance
        """
        path = Path(path)
        
        # Load configuration
        with open(path / "config.json", "r") as f:
            config_dict = json.load(f)
        config = SearchConfig(**config_dict)
        
        # Create database instance
        db = cls(config)
        
        # Load index
        if (path / "index.faiss").exists():
            index = faiss.read_index(str(path / "index.faiss"))
            if config.use_gpu and db.gpu_resources is not None:
                db.index = faiss.index_cpu_to_gpu(db.gpu_resources, config.gpu_id, index)
            else:
                db.index = index
        
        # Load embeddings
        if (path / "embeddings.npy").exists():
            db.embeddings = np.load(str(path / "embeddings.npy"))
        
        # Load metadata and mappings
        with open(path / "metadata.json", "r") as f:
            data = json.load(f)
            db.metadata = data['metadata']
            db.id_to_idx = {k: int(v) for k, v in data['id_to_idx'].items()}
            db.idx_to_id = {int(k): v for k, v in data['idx_to_id'].items()}
            db.is_trained = data['is_trained']
        
        logger.info(f"Database loaded from {path}")
        return db


class SimilaritySearchEngine:
    """
    High-level similarity search engine for fashion applications.
    
    Provides convenient methods for text and image similarity search
    with automatic query preprocessing and result postprocessing.
    """
    
    def __init__(self, 
                 database: VectorDatabase,
                 clip_model: Optional[Any] = None):
        self.database = database
        self.clip_model = clip_model
        
        # Cache for frequent queries
        self._query_cache = {}
        self._cache_size = 1000
    
    def search_by_text(self, 
                      text_queries: Union[str, List[str]],
                      k: int = 50,
                      return_metadata: bool = True) -> Dict[str, Any]:
        """
        Search for similar items using text queries.
        
        Args:
            text_queries: Text query or list of queries
            k: Number of results to return
            return_metadata: Whether to return metadata
            
        Returns:
            Search results
        """
        if isinstance(text_queries, str):
            text_queries = [text_queries]
        
        # Get text embeddings
        if self.clip_model is not None:
            text_embeddings = self.clip_model.get_text_features(text_queries)
            text_embeddings = text_embeddings.cpu().numpy()
        else:
            raise ValueError("CLIP model required for text search")
        
        # Perform search
        return self.database.search(
            text_embeddings, k=k, return_metadata=return_metadata
        )
    
    def search_by_image(self, 
                       image_embeddings: np.ndarray,
                       k: int = 50,
                       return_metadata: bool = True) -> Dict[str, Any]:
        """
        Search for similar items using image embeddings.
        
        Args:
            image_embeddings: Pre-computed image embeddings
            k: Number of results to return
            return_metadata: Whether to return metadata
            
        Returns:
            Search results
        """
        return self.database.search(
            image_embeddings, k=k, return_metadata=return_metadata
        )
    
    def search_by_embedding(self, 
                          embeddings: np.ndarray,
                          k: int = 50,
                          return_metadata: bool = True) -> Dict[str, Any]:
        """
        Search for similar items using pre-computed embeddings.
        
        Args:
            embeddings: Query embeddings
            k: Number of results to return
            return_metadata: Whether to return metadata
            
        Returns:
            Search results
        """
        return self.database.search(
            embeddings, k=k, return_metadata=return_metadata
        )
    
    def get_recommendations(self, 
                          item_id: str,
                          k: int = 10,
                          exclude_self: bool = True) -> Dict[str, Any]:
        """
        Get recommendations for a specific item.
        
        Args:
            item_id: Item ID to get recommendations for
            k: Number of recommendations
            exclude_self: Whether to exclude the item itself
            
        Returns:
            Recommendation results
        """
        if item_id not in self.database.id_to_idx:
            raise ValueError(f"Item ID {item_id} not found in database")
        
        # Get item embedding
        item_idx = self.database.id_to_idx[item_id]
        item_embedding = self.database.embeddings[item_idx:item_idx+1]
        
        # Search for similar items
        search_k = k + 1 if exclude_self else k
        results = self.database.search(
            item_embedding, k=search_k, return_metadata=True
        )
        
        # Remove self if requested
        if exclude_self and len(results['ids'][0]) > 0:
            if results['ids'][0][0] == item_id:
                results['ids'][0] = results['ids'][0][1:]
                results['distances'][0] = results['distances'][0][1:]
                if 'metadata' in results:
                    results['metadata'][0] = results['metadata'][0][1:]
        
        return results
    
    def evaluate_recall(self, 
                       query_embeddings: np.ndarray,
                       ground_truth: List[List[str]],
                       k_values: List[int] = [1, 5, 10, 20, 50]) -> Dict[int, float]:
        """
        Evaluate recall@k for a set of queries.
        
        Args:
            query_embeddings: Query embeddings
            ground_truth: Ground truth relevant items for each query
            k_values: K values to evaluate
            
        Returns:
            Recall@k scores
        """
        if len(query_embeddings) != len(ground_truth):
            raise ValueError("Number of queries must match ground truth")
        
        max_k = max(k_values)
        results = self.database.search(query_embeddings, k=max_k, return_metadata=False)
        
        recall_scores = {}
        for k in k_values:
            total_recall = 0
            valid_queries = 0
            
            for i, (predicted_ids, true_ids) in enumerate(zip(results['ids'], ground_truth)):
                if len(true_ids) == 0:
                    continue
                
                predicted_k = predicted_ids[:k]
                relevant_retrieved = len(set(predicted_k) & set(true_ids))
                total_recall += relevant_retrieved / len(true_ids)
                valid_queries += 1
            
            recall_scores[k] = total_recall / valid_queries if valid_queries > 0 else 0.0
        
        return recall_scores


# Utility functions
def create_vector_database(config: SearchConfig) -> VectorDatabase:
    """Create a vector database with the specified configuration."""
    return VectorDatabase(config)


def optimize_index_parameters(embeddings: np.ndarray,
                            target_recall: float = 0.99,
                            max_memory_gb: float = 8.0) -> SearchConfig:
    """
    Optimize index parameters for given embeddings and constraints.
    
    Args:
        embeddings: Sample embeddings to optimize for
        target_recall: Target recall rate
        max_memory_gb: Maximum memory usage in GB
        
    Returns:
        Optimized SearchConfig
    """
    n_vectors, dim = embeddings.shape
    
    # Estimate optimal number of centroids
    n_centroids = min(1024, int(np.sqrt(n_vectors)))
    n_centroids = max(64, n_centroids)  # Minimum 64 centroids
    
    # Estimate PQ parameters
    pq_m = min(64, dim // 4)  # Typical rule of thumb
    pq_m = max(8, pq_m)  # Minimum 8 subquantizers
    
    # Adjust probe based on target recall
    if target_recall >= 0.99:
        n_probe = min(n_centroids // 4, 64)
    elif target_recall >= 0.95:
        n_probe = min(n_centroids // 8, 32)
    else:
        n_probe = min(n_centroids // 16, 16)
    
    config = SearchConfig(
        embedding_dim=dim,
        index_type="IVF-PQ",
        n_centroids=n_centroids,
        n_probe=n_probe,
        pq_m=pq_m,
        pq_nbits=8,
        target_recall=target_recall,
        max_memory_gb=max_memory_gb
    )
    
    return config


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample embeddings
    n_samples = 10000
    embedding_dim = 512
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    ids = [f"item_{i}" for i in range(n_samples)]
    
    # Create database
    config = SearchConfig(embedding_dim=embedding_dim, use_gpu=False)
    db = VectorDatabase(config)
    
    # Add embeddings
    db.add_embeddings(embeddings, ids)
    
    # Search
    query = np.random.randn(1, embedding_dim).astype(np.float32)
    results = db.search(query, k=10)
    
    print(f"Search results: {results['ids'][0][:5]}")
    print(f"Search time: {results['search_time']:.4f}s")
    
    # Print statistics
    stats = db.get_statistics()
    print(f"Database stats: {stats}")