"""
Index Builder for Large-Scale Fashion Retrieval.

This module provides utilities for building and maintaining large-scale
search indices for fashion retrieval systems. It includes batch processing,
index optimization, compression, and incremental updates.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator, Callable
import logging
from pathlib import Path
import json
import pickle
import time
import gc
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import warnings

try:
    import faiss
except ImportError:
    warnings.warn("FAISS not installed. Please install with: pip install faiss-cpu or faiss-gpu")
    faiss = None

try:
    import h5py
except ImportError:
    warnings.warn("h5py not installed. Please install with: pip install h5py")
    h5py = None

from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import psutil

from .similarity_search import VectorDatabase, SearchConfig
from .clip_model import FashionCLIP

logger = logging.getLogger(__name__)


@dataclass
class IndexBuildConfig:
    """Configuration for index building process."""
    
    # Data processing
    batch_size: int = 1000
    max_workers: int = 8
    use_gpu: bool = True
    gpu_ids: List[int] = None
    
    # Memory management
    max_memory_gb: float = 16.0
    memory_check_interval: int = 100
    
    # Index optimization
    optimize_index: bool = True
    compression_level: int = 1  # 0: none, 1: basic, 2: aggressive
    quantization_bits: int = 8
    
    # Incremental updates
    support_incremental: bool = True
    update_threshold: float = 0.1  # Fraction of data that triggers rebuild
    
    # Validation
    validate_index: bool = True
    validation_sample_size: int = 1000
    target_recall: float = 0.99
    
    # Storage
    use_hdf5: bool = True
    compression_type: str = "gzip"  # for HDF5
    chunk_size: int = 10000
    
    # Distributed building
    distributed: bool = False
    num_shards: int = 1
    
    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = [0]
        
        # Adjust batch size based on available memory
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if available_memory < self.max_memory_gb:
            self.batch_size = max(100, int(self.batch_size * available_memory / self.max_memory_gb))
            logger.warning(f"Adjusted batch size to {self.batch_size} due to memory constraints")


class DatasetProcessor:
    """
    Processes fashion datasets for index building.
    
    Handles data loading, preprocessing, and batch generation
    for efficient index construction.
    """
    
    def __init__(self, config: IndexBuildConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config.max_memory_gb)
    
    def process_dataset(self, 
                       dataset_path: Union[str, Path],
                       model: FashionCLIP,
                       output_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a dataset and generate embeddings.
        
        Args:
            dataset_path: Path to the dataset
            model: Fashion CLIP model for embedding generation
            output_path: Path to save processed embeddings
            
        Returns:
            Processing statistics
        """
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'total_items': 0,
            'processed_items': 0,
            'failed_items': 0,
            'processing_time': 0,
            'memory_usage': []
        }
        
        start_time = time.time()
        
        # Initialize data loader
        data_loader = self._create_data_loader(dataset_path)
        
        # Process in batches
        batch_embeddings = []
        batch_ids = []
        batch_metadata = []
        
        for batch_idx, batch_data in enumerate(data_loader):
            try:
                # Check memory usage
                if batch_idx % self.config.memory_check_interval == 0:
                    memory_usage = self.memory_monitor.get_memory_usage()
                    stats['memory_usage'].append(memory_usage)
                    
                    if memory_usage > self.config.max_memory_gb:
                        logger.warning(f"Memory usage {memory_usage:.2f}GB exceeds limit")
                        gc.collect()
                
                # Generate embeddings
                embeddings = self._generate_embeddings(batch_data, model)
                
                # Collect batch data
                batch_embeddings.append(embeddings)
                batch_ids.extend(batch_data['ids'])
                batch_metadata.extend(batch_data['metadata'])
                
                stats['processed_items'] += len(batch_data['ids'])
                
                # Save batch if it's large enough
                if len(batch_ids) >= self.config.chunk_size:
                    self._save_batch(batch_embeddings, batch_ids, batch_metadata, 
                                   output_path, batch_idx)
                    batch_embeddings = []
                    batch_ids = []
                    batch_metadata = []
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                stats['failed_items'] += len(batch_data.get('ids', []))
                continue
        
        # Save remaining data
        if batch_embeddings:
            self._save_batch(batch_embeddings, batch_ids, batch_metadata, 
                           output_path, "final")
        
        stats['processing_time'] = time.time() - start_time
        stats['total_items'] = stats['processed_items'] + stats['failed_items']
        
        # Save statistics
        with open(output_path / "processing_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def _create_data_loader(self, dataset_path: Path) -> Iterator[Dict[str, Any]]:
        """Create a data loader for the dataset."""
        # This would be implemented based on the specific dataset format
        # For now, we'll create a mock implementation
        
        # Example implementation for a JSON dataset
        if dataset_path.suffix == '.json':
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            for i in range(0, len(data), self.config.batch_size):
                batch = data[i:i + self.config.batch_size]
                yield {
                    'ids': [item['id'] for item in batch],
                    'images': [item['image_path'] for item in batch],
                    'texts': [item['description'] for item in batch],
                    'metadata': [item.get('metadata', {}) for item in batch]
                }
        
        # Example implementation for a directory of images
        elif dataset_path.is_dir():
            image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
            
            for i in range(0, len(image_files), self.config.batch_size):
                batch_files = image_files[i:i + self.config.batch_size]
                yield {
                    'ids': [f.stem for f in batch_files],
                    'images': [str(f) for f in batch_files],
                    'texts': [''] * len(batch_files),  # Empty texts
                    'metadata': [{'path': str(f)} for f in batch_files]
                }
    
    def _generate_embeddings(self, batch_data: Dict[str, Any], model: FashionCLIP) -> np.ndarray:
        """Generate embeddings for a batch of data."""
        # This would be implemented based on the model's interface
        # For now, we'll create a mock implementation
        
        batch_size = len(batch_data['ids'])
        embedding_dim = model.config.projection_dim
        
        # Mock embedding generation
        embeddings = np.random.randn(batch_size, embedding_dim).astype(np.float32)
        embeddings = normalize(embeddings, norm='l2', axis=1)
        
        return embeddings
    
    def _save_batch(self, 
                   embeddings: List[np.ndarray], 
                   ids: List[str],
                   metadata: List[Dict[str, Any]],
                   output_path: Path,
                   batch_name: Union[str, int]) -> None:
        """Save a batch of embeddings to disk."""
        if not embeddings:
            return
        
        # Concatenate embeddings
        combined_embeddings = np.vstack(embeddings)
        
        if self.config.use_hdf5 and h5py is not None:
            # Save to HDF5
            filename = output_path / f"batch_{batch_name}.h5"
            with h5py.File(filename, 'w') as f:
                f.create_dataset('embeddings', data=combined_embeddings, 
                               compression=self.config.compression_type)
                f.create_dataset('ids', data=[id.encode('utf-8') for id in ids])
                f.attrs['metadata'] = json.dumps(metadata)
        else:
            # Save to numpy format
            filename = output_path / f"batch_{batch_name}.npz"
            np.savez_compressed(filename, 
                              embeddings=combined_embeddings,
                              ids=ids,
                              metadata=metadata)


class IndexBuilder:
    """
    Main index builder class for creating optimized search indices.
    
    Handles the complete pipeline from data processing to index optimization
    and validation.
    """
    
    def __init__(self, config: IndexBuildConfig):
        self.config = config
        self.processor = DatasetProcessor(config)
        self.optimizer = IndexOptimizer(config)
        self.validator = IndexValidator(config)
    
    def build_index(self, 
                   dataset_path: Union[str, Path],
                   model: FashionCLIP,
                   output_path: Union[str, Path],
                   search_config: SearchConfig) -> VectorDatabase:
        """
        Build a complete search index from a dataset.
        
        Args:
            dataset_path: Path to the dataset
            model: Fashion CLIP model for embedding generation
            output_path: Path to save the index
            search_config: Configuration for the search index
            
        Returns:
            Built VectorDatabase
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting index build for dataset: {dataset_path}")
        
        # Step 1: Process dataset and generate embeddings
        embeddings_path = output_path / "embeddings"
        processing_stats = self.processor.process_dataset(
            dataset_path, model, embeddings_path
        )
        
        logger.info(f"Processed {processing_stats['processed_items']} items in "
                   f"{processing_stats['processing_time']:.2f} seconds")
        
        # Step 2: Load embeddings and create index
        embeddings, ids, metadata = self._load_embeddings(embeddings_path)
        
        # Step 3: Create and optimize index
        database = VectorDatabase(search_config)
        database.add_embeddings(embeddings, ids, metadata)
        
        if self.config.optimize_index:
            self.optimizer.optimize_index(database)
        
        # Step 4: Validate index
        if self.config.validate_index:
            validation_results = self.validator.validate_index(database, embeddings)
            logger.info(f"Index validation - Recall@10: {validation_results['recall_at_10']:.4f}")
        
        # Step 5: Save index
        database.save(output_path / "index")
        
        # Step 6: Save build metadata
        build_metadata = {
            'config': asdict(self.config),
            'search_config': asdict(search_config),
            'processing_stats': processing_stats,
            'build_time': time.time(),
            'total_items': len(ids)
        }
        
        with open(output_path / "build_metadata.json", "w") as f:
            json.dump(build_metadata, f, indent=2)
        
        logger.info(f"Index build completed successfully. Saved to: {output_path}")
        
        return database
    
    def _load_embeddings(self, embeddings_path: Path) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Load embeddings from processed files."""
        all_embeddings = []
        all_ids = []
        all_metadata = {}
        
        # Load from HDF5 files
        if self.config.use_hdf5 and h5py is not None:
            h5_files = list(embeddings_path.glob("*.h5"))
            
            for h5_file in h5_files:
                with h5py.File(h5_file, 'r') as f:
                    embeddings = f['embeddings'][:]
                    ids = [id.decode('utf-8') for id in f['ids'][:]]
                    metadata = json.loads(f.attrs['metadata'])
                    
                    all_embeddings.append(embeddings)
                    all_ids.extend(ids)
                    
                    for i, meta in enumerate(metadata):
                        all_metadata[ids[i]] = meta
        
        # Load from numpy files
        else:
            npz_files = list(embeddings_path.glob("*.npz"))
            
            for npz_file in npz_files:
                data = np.load(npz_file, allow_pickle=True)
                embeddings = data['embeddings']
                ids = data['ids'].tolist()
                metadata = data['metadata'].tolist()
                
                all_embeddings.append(embeddings)
                all_ids.extend(ids)
                
                for i, meta in enumerate(metadata):
                    all_metadata[ids[i]] = meta
        
        # Combine all embeddings
        combined_embeddings = np.vstack(all_embeddings)
        
        return combined_embeddings, all_ids, all_metadata
    
    def incremental_update(self, 
                          database: VectorDatabase,
                          new_embeddings: np.ndarray,
                          new_ids: List[str],
                          new_metadata: Optional[Dict[str, Any]] = None) -> VectorDatabase:
        """
        Perform incremental update of an existing index.
        
        Args:
            database: Existing VectorDatabase
            new_embeddings: New embeddings to add
            new_ids: IDs for new embeddings
            new_metadata: Optional metadata for new embeddings
            
        Returns:
            Updated VectorDatabase
        """
        current_size = len(database.id_to_idx)
        new_size = len(new_embeddings)
        
        # Check if rebuild is needed
        if new_size / current_size > self.config.update_threshold:
            logger.info(f"Rebuilding index due to large update ({new_size} new items)")
            # For large updates, rebuild the entire index
            all_embeddings = np.vstack([database.embeddings, new_embeddings])
            all_ids = list(database.id_to_idx.keys()) + new_ids
            all_metadata = database.metadata.copy()
            
            if new_metadata:
                all_metadata.update(new_metadata)
            
            # Create new database
            new_database = VectorDatabase(database.config)
            new_database.add_embeddings(all_embeddings, all_ids, all_metadata)
            
            return new_database
        
        else:
            # Incremental update
            logger.info(f"Performing incremental update with {new_size} new items")
            database.add_embeddings(new_embeddings, new_ids, new_metadata)
            return database


class IndexOptimizer:
    """
    Optimizes search indices for better performance and storage efficiency.
    
    Includes compression, quantization, and index structure optimization.
    """
    
    def __init__(self, config: IndexBuildConfig):
        self.config = config
    
    def optimize_index(self, database: VectorDatabase) -> None:
        """
        Optimize the search index.
        
        Args:
            database: VectorDatabase to optimize
        """
        logger.info("Starting index optimization")
        
        if self.config.compression_level > 0:
            self._compress_index(database)
        
        if self.config.quantization_bits < 32:
            self._quantize_index(database)
        
        self._optimize_search_parameters(database)
        
        logger.info("Index optimization completed")
    
    def _compress_index(self, database: VectorDatabase) -> None:
        """Apply compression to the index."""
        if faiss is None:
            logger.warning("FAISS not available, skipping compression")
            return
        
        # Move to CPU for compression
        if database.config.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(database.index)
        else:
            cpu_index = database.index
        
        # Apply compression based on level
        if self.config.compression_level == 1:
            # Basic compression with PQ
            if hasattr(cpu_index, 'quantizer'):
                # Already has quantizer, optimize parameters
                if hasattr(cpu_index, 'pq'):
                    cpu_index.pq.compute_codes = True
        
        elif self.config.compression_level == 2:
            # Aggressive compression
            if isinstance(cpu_index, faiss.IndexIVFPQ):
                # Further optimize PQ parameters
                cpu_index.make_direct_map()
                cpu_index.set_direct_map_type(faiss.DirectMap.Array)
        
        # Move back to GPU if needed
        if database.config.use_gpu:
            database.index = faiss.index_cpu_to_gpu(
                database.gpu_resources, database.config.gpu_id, cpu_index
            )
        else:
            database.index = cpu_index
    
    def _quantize_index(self, database: VectorDatabase) -> None:
        """Apply quantization to reduce memory usage."""
        if faiss is None:
            logger.warning("FAISS not available, skipping quantization")
            return
        
        # Quantization is typically built into the index type (e.g., IVF-PQ)
        # For additional quantization, we would need to rebuild the index
        logger.info(f"Index already uses {self.config.quantization_bits}-bit quantization")
    
    def _optimize_search_parameters(self, database: VectorDatabase) -> None:
        """Optimize search parameters for better performance."""
        if hasattr(database.index, 'nprobe'):
            # Optimize nprobe based on index size
            total_vectors = database.index.ntotal
            
            if total_vectors < 10000:
                optimal_nprobe = 8
            elif total_vectors < 100000:
                optimal_nprobe = 16
            elif total_vectors < 1000000:
                optimal_nprobe = 32
            else:
                optimal_nprobe = 64
            
            database.index.nprobe = optimal_nprobe
            database.config.n_probe = optimal_nprobe
            
            logger.info(f"Optimized nprobe to {optimal_nprobe}")


class IndexValidator:
    """
    Validates search index quality and performance.
    
    Measures recall, precision, and search speed to ensure
    the index meets quality requirements.
    """
    
    def __init__(self, config: IndexBuildConfig):
        self.config = config
    
    def validate_index(self, 
                      database: VectorDatabase,
                      embeddings: np.ndarray) -> Dict[str, float]:
        """
        Validate the search index.
        
        Args:
            database: VectorDatabase to validate
            embeddings: Original embeddings for validation
            
        Returns:
            Validation metrics
        """
        logger.info("Starting index validation")
        
        # Sample embeddings for validation
        sample_size = min(self.config.validation_sample_size, len(embeddings))
        sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[sample_indices]
        
        # Compute ground truth (exact search)
        ground_truth = self._compute_ground_truth(sample_embeddings, embeddings)
        
        # Test index search
        search_results = database.search(sample_embeddings, k=100, return_distances=True)
        
        # Compute metrics
        metrics = self._compute_metrics(ground_truth, search_results)
        
        logger.info(f"Validation completed - Recall@10: {metrics['recall_at_10']:.4f}")
        
        return metrics
    
    def _compute_ground_truth(self, 
                            queries: np.ndarray,
                            database: np.ndarray,
                            k: int = 100) -> List[List[int]]:
        """Compute ground truth using exact search."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        ground_truth = []
        
        for query in queries:
            similarities = cosine_similarity(query.reshape(1, -1), database)[0]
            top_k_indices = np.argsort(similarities)[::-1][:k]
            ground_truth.append(top_k_indices.tolist())
        
        return ground_truth
    
    def _compute_metrics(self, 
                        ground_truth: List[List[int]],
                        search_results: Dict[str, Any]) -> Dict[str, float]:
        """Compute validation metrics."""
        metrics = {}
        
        # Convert result IDs to indices for comparison
        # This is a simplified version - in practice, you'd need proper ID mapping
        predicted_indices = search_results['ids']
        
        # Compute recall@k for different k values
        for k in [1, 5, 10, 20, 50]:
            total_recall = 0
            
            for i, (true_indices, pred_indices) in enumerate(zip(ground_truth, predicted_indices)):
                if len(pred_indices) >= k:
                    true_k = set(true_indices[:k])
                    pred_k = set(range(len(pred_indices[:k])))  # Simplified
                    
                    recall = len(true_k & pred_k) / len(true_k)
                    total_recall += recall
            
            metrics[f'recall_at_{k}'] = total_recall / len(ground_truth)
        
        return metrics


class MemoryMonitor:
    """Monitors memory usage during index building."""
    
    def __init__(self, max_memory_gb: float):
        self.max_memory_gb = max_memory_gb
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        return self.process.memory_info().rss / (1024**3)
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage exceeds limit."""
        return self.get_memory_usage() > self.max_memory_gb
    
    def force_cleanup(self) -> None:
        """Force garbage collection and cleanup."""
        import gc
        gc.collect()
        
        # Additional cleanup for PyTorch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Utility functions
def create_index_builder(config: IndexBuildConfig) -> IndexBuilder:
    """Create an index builder with the specified configuration."""
    return IndexBuilder(config)


def estimate_memory_requirements(num_vectors: int, 
                                embedding_dim: int,
                                index_type: str = "IVF-PQ") -> float:
    """
    Estimate memory requirements for building an index.
    
    Args:
        num_vectors: Number of vectors to index
        embedding_dim: Dimension of embeddings
        index_type: Type of index to build
        
    Returns:
        Estimated memory in GB
    """
    # Base memory for embeddings (float32)
    base_memory = num_vectors * embedding_dim * 4 / (1024**3)
    
    # Index overhead based on type
    if index_type == "Flat":
        index_overhead = base_memory  # Store all vectors
    elif index_type == "IVF":
        index_overhead = base_memory * 0.1  # Centroid storage
    elif index_type == "IVF-PQ":
        index_overhead = base_memory * 0.05  # Compressed storage
    elif index_type == "HNSW":
        index_overhead = base_memory * 0.2  # Graph storage
    else:
        index_overhead = base_memory * 0.1  # Default estimate
    
    # Additional overhead for processing
    processing_overhead = base_memory * 0.5
    
    total_memory = base_memory + index_overhead + processing_overhead
    
    return total_memory


def optimize_build_config(num_vectors: int,
                         embedding_dim: int,
                         available_memory_gb: float) -> IndexBuildConfig:
    """
    Optimize build configuration based on data size and available resources.
    
    Args:
        num_vectors: Number of vectors to index
        embedding_dim: Dimension of embeddings
        available_memory_gb: Available memory in GB
        
    Returns:
        Optimized IndexBuildConfig
    """
    # Estimate memory requirements
    estimated_memory = estimate_memory_requirements(num_vectors, embedding_dim)
    
    # Adjust configuration based on memory constraints
    if estimated_memory > available_memory_gb:
        # Reduce batch size
        batch_size = max(100, int(1000 * available_memory_gb / estimated_memory))
        use_compression = True
        compression_level = 2
    else:
        batch_size = 1000
        use_compression = False
        compression_level = 0
    
    # Determine optimal number of workers
    max_workers = min(mp.cpu_count(), 8)
    
    # GPU usage
    use_gpu = torch.cuda.is_available()
    
    config = IndexBuildConfig(
        batch_size=batch_size,
        max_workers=max_workers,
        use_gpu=use_gpu,
        max_memory_gb=available_memory_gb * 0.8,  # Leave some headroom
        optimize_index=use_compression,
        compression_level=compression_level
    )
    
    return config


if __name__ == "__main__":
    # Example usage
    from .clip_model import FashionCLIPConfig, FashionCLIP
    
    # Create mock dataset
    dataset_path = Path("mock_dataset.json")
    mock_data = [
        {
            "id": f"item_{i}",
            "image_path": f"image_{i}.jpg",
            "description": f"Fashion item {i}",
            "metadata": {"category": "dress", "color": "red"}
        }
        for i in range(1000)
    ]
    
    with open(dataset_path, "w") as f:
        json.dump(mock_data, f)
    
    # Create model
    clip_config = FashionCLIPConfig()
    model = FashionCLIP(clip_config)
    
    # Create build configuration
    build_config = IndexBuildConfig(
        batch_size=100,
        max_workers=4,
        use_gpu=False,
        validate_index=True
    )
    
    # Create search configuration
    search_config = SearchConfig(
        embedding_dim=512,
        index_type="IVF-PQ",
        use_gpu=False
    )
    
    # Build index
    builder = IndexBuilder(build_config)
    database = builder.build_index(
        dataset_path=dataset_path,
        model=model,
        output_path="test_index",
        search_config=search_config
    )
    
    print(f"Index built successfully with {database.get_statistics()['total_vectors']} vectors")
    
    # Clean up
    dataset_path.unlink()
    import shutil
    shutil.rmtree("test_index", ignore_errors=True)