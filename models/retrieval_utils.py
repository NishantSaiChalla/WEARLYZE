"""
Retrieval Utilities for Fashion Similarity Search.

This module provides utilities for hard negative mining, embedding visualization,
similarity metrics, and evaluation metrics for fashion retrieval systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging
from pathlib import Path
from collections import defaultdict
import json
from dataclasses import dataclass
import warnings

try:
    import umap
except ImportError:
    warnings.warn("UMAP not installed. Install with: pip install umap-learn")
    umap = None

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    mean_average_precision: float
    ndcg_at_k: Dict[int, float]
    mean_reciprocal_rank: float
    hit_rate_at_k: Dict[int, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'recall_at_k': self.recall_at_k,
            'precision_at_k': self.precision_at_k,
            'mean_average_precision': self.mean_average_precision,
            'ndcg_at_k': self.ndcg_at_k,
            'mean_reciprocal_rank': self.mean_reciprocal_rank,
            'hit_rate_at_k': self.hit_rate_at_k
        }


class HardNegativeMiner:
    """
    Advanced hard negative mining for contrastive learning.
    
    Implements multiple strategies for mining hard negatives including
    semi-hard negatives, hardest negatives, and random hard negatives.
    """
    
    def __init__(self, 
                 margin: float = 0.2,
                 strategy: str = "semi_hard",
                 temperature: float = 0.1,
                 negative_ratio: float = 0.1):
        """
        Initialize hard negative miner.
        
        Args:
            margin: Margin for triplet loss
            strategy: Mining strategy ("hard", "semi_hard", "random_hard")
            temperature: Temperature for softmax-based sampling
            negative_ratio: Ratio of negatives to mine
        """
        self.margin = margin
        self.strategy = strategy
        self.temperature = temperature
        self.negative_ratio = negative_ratio
        
        if strategy not in ["hard", "semi_hard", "random_hard"]:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def mine_triplets(self, 
                     embeddings: torch.Tensor,
                     labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mine hard triplets from embeddings and labels.
        
        Args:
            embeddings: Embedding vectors [N, D]
            labels: Corresponding labels [N]
            
        Returns:
            Triplet indices: (anchor_idx, positive_idx, negative_idx)
        """
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Create masks for positive and negative pairs
        labels_equal = labels.unsqueeze(1) == labels.unsqueeze(0)
        labels_not_equal = ~labels_equal
        
        # Remove diagonal (self-comparisons)
        eye = torch.eye(len(embeddings), device=embeddings.device).bool()
        labels_equal = labels_equal & ~eye
        labels_not_equal = labels_not_equal & ~eye
        
        triplets = []
        
        for i in range(len(embeddings)):
            # Get positive and negative distances for anchor i
            positive_distances = distances[i][labels_equal[i]]
            negative_distances = distances[i][labels_not_equal[i]]
            
            if len(positive_distances) == 0 or len(negative_distances) == 0:
                continue
            
            # Find positive and negative indices
            positive_indices = torch.where(labels_equal[i])[0]
            negative_indices = torch.where(labels_not_equal[i])[0]
            
            if self.strategy == "hard":
                # Hardest positive and hardest negative
                hardest_positive_idx = positive_indices[torch.argmax(positive_distances)]
                hardest_negative_idx = negative_indices[torch.argmin(negative_distances)]
                
                triplets.append([i, hardest_positive_idx.item(), hardest_negative_idx.item()])
            
            elif self.strategy == "semi_hard":
                # Semi-hard negatives: d(a,n) > d(a,p) but d(a,n) < d(a,p) + margin
                for pos_idx, pos_dist in zip(positive_indices, positive_distances):
                    semi_hard_negatives = negative_indices[
                        (negative_distances > pos_dist) & 
                        (negative_distances < pos_dist + self.margin)
                    ]
                    
                    if len(semi_hard_negatives) > 0:
                        # Randomly sample from semi-hard negatives
                        neg_idx = semi_hard_negatives[torch.randint(len(semi_hard_negatives), (1,))]
                        triplets.append([i, pos_idx.item(), neg_idx.item()])
            
            elif self.strategy == "random_hard":
                # Random sampling with temperature-based weighting
                # Sample positives with inverse distance weighting
                pos_weights = F.softmax(-positive_distances / self.temperature, dim=0)
                pos_idx = positive_indices[torch.multinomial(pos_weights, 1)]
                
                # Sample negatives with distance-based weighting
                neg_weights = F.softmax(-negative_distances / self.temperature, dim=0)
                neg_idx = negative_indices[torch.multinomial(neg_weights, 1)]
                
                triplets.append([i, pos_idx.item(), neg_idx.item()])
        
        if len(triplets) == 0:
            return torch.empty((0, 3), dtype=torch.long, device=embeddings.device)
        
        triplets = torch.tensor(triplets, device=embeddings.device)
        return triplets[:, 0], triplets[:, 1], triplets[:, 2]
    
    def compute_triplet_loss(self, 
                           embeddings: torch.Tensor,
                           labels: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss with hard negative mining.
        
        Args:
            embeddings: Embedding vectors
            labels: Corresponding labels
            
        Returns:
            Triplet loss
        """
        anchor_idx, positive_idx, negative_idx = self.mine_triplets(embeddings, labels)
        
        if len(anchor_idx) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Get embeddings for triplets
        anchor_embeddings = embeddings[anchor_idx]
        positive_embeddings = embeddings[positive_idx]
        negative_embeddings = embeddings[negative_idx]
        
        # Compute distances
        pos_distances = F.pairwise_distance(anchor_embeddings, positive_embeddings)
        neg_distances = F.pairwise_distance(anchor_embeddings, negative_embeddings)
        
        # Compute triplet loss
        loss = F.relu(pos_distances - neg_distances + self.margin)
        return loss.mean()


class SimilarityMetrics:
    """Collection of similarity metrics for fashion retrieval."""
    
    @staticmethod
    def cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between two sets of vectors."""
        return cosine_similarity(x, y)
    
    @staticmethod
    def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance between two sets of vectors."""
        return euclidean_distances(x, y)
    
    @staticmethod
    def dot_product_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute dot product similarity between two sets of vectors."""
        return np.dot(x, y.T)
    
    @staticmethod
    def manhattan_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute Manhattan distance between two sets of vectors."""
        return np.sum(np.abs(x[:, None] - y), axis=2)
    
    @staticmethod
    def chebyshev_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute Chebyshev distance between two sets of vectors."""
        return np.max(np.abs(x[:, None] - y), axis=2)
    
    @staticmethod
    def pearson_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute Pearson correlation between two sets of vectors."""
        correlations = []
        for i in range(len(x)):
            corr_row = []
            for j in range(len(y)):
                corr = np.corrcoef(x[i], y[j])[0, 1]
                corr_row.append(corr if not np.isnan(corr) else 0.0)
            correlations.append(corr_row)
        return np.array(correlations)


class RetrievalEvaluator:
    """
    Comprehensive evaluator for fashion retrieval systems.
    
    Provides standard retrieval metrics including Recall@K, Precision@K,
    Mean Average Precision (mAP), and NDCG@K.
    """
    
    def __init__(self, k_values: List[int] = [1, 5, 10, 20, 50, 100]):
        """
        Initialize evaluator.
        
        Args:
            k_values: List of K values for evaluation
        """
        self.k_values = k_values
    
    def evaluate(self, 
                predictions: List[List[str]],
                ground_truth: List[List[str]],
                relevance_scores: Optional[List[List[float]]] = None) -> RetrievalMetrics:
        """
        Evaluate retrieval performance.
        
        Args:
            predictions: Predicted items for each query
            ground_truth: Ground truth relevant items for each query
            relevance_scores: Optional relevance scores for each prediction
            
        Returns:
            RetrievalMetrics object with evaluation results
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Number of predictions must match ground truth")
        
        # Compute metrics
        recall_at_k = self._compute_recall_at_k(predictions, ground_truth)
        precision_at_k = self._compute_precision_at_k(predictions, ground_truth)
        mean_average_precision = self._compute_mean_average_precision(predictions, ground_truth)
        ndcg_at_k = self._compute_ndcg_at_k(predictions, ground_truth, relevance_scores)
        mean_reciprocal_rank = self._compute_mean_reciprocal_rank(predictions, ground_truth)
        hit_rate_at_k = self._compute_hit_rate_at_k(predictions, ground_truth)
        
        return RetrievalMetrics(
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            mean_average_precision=mean_average_precision,
            ndcg_at_k=ndcg_at_k,
            mean_reciprocal_rank=mean_reciprocal_rank,
            hit_rate_at_k=hit_rate_at_k
        )
    
    def _compute_recall_at_k(self, 
                           predictions: List[List[str]],
                           ground_truth: List[List[str]]) -> Dict[int, float]:
        """Compute Recall@K for different K values."""
        recall_at_k = {}
        
        for k in self.k_values:
            total_recall = 0
            valid_queries = 0
            
            for pred, truth in zip(predictions, ground_truth):
                if len(truth) == 0:
                    continue
                
                pred_k = set(pred[:k])
                truth_set = set(truth)
                
                recall = len(pred_k & truth_set) / len(truth_set)
                total_recall += recall
                valid_queries += 1
            
            recall_at_k[k] = total_recall / valid_queries if valid_queries > 0 else 0.0
        
        return recall_at_k
    
    def _compute_precision_at_k(self, 
                              predictions: List[List[str]],
                              ground_truth: List[List[str]]) -> Dict[int, float]:
        """Compute Precision@K for different K values."""
        precision_at_k = {}
        
        for k in self.k_values:
            total_precision = 0
            valid_queries = 0
            
            for pred, truth in zip(predictions, ground_truth):
                if len(truth) == 0:
                    continue
                
                pred_k = set(pred[:k])
                truth_set = set(truth)
                
                precision = len(pred_k & truth_set) / len(pred_k) if len(pred_k) > 0 else 0.0
                total_precision += precision
                valid_queries += 1
            
            precision_at_k[k] = total_precision / valid_queries if valid_queries > 0 else 0.0
        
        return precision_at_k
    
    def _compute_mean_average_precision(self, 
                                      predictions: List[List[str]],
                                      ground_truth: List[List[str]]) -> float:
        """Compute Mean Average Precision (mAP)."""
        total_ap = 0
        valid_queries = 0
        
        for pred, truth in zip(predictions, ground_truth):
            if len(truth) == 0:
                continue
            
            truth_set = set(truth)
            hits = 0
            ap = 0
            
            for i, item in enumerate(pred):
                if item in truth_set:
                    hits += 1
                    ap += hits / (i + 1)
            
            if hits > 0:
                ap /= len(truth_set)
            
            total_ap += ap
            valid_queries += 1
        
        return total_ap / valid_queries if valid_queries > 0 else 0.0
    
    def _compute_ndcg_at_k(self, 
                         predictions: List[List[str]],
                         ground_truth: List[List[str]],
                         relevance_scores: Optional[List[List[float]]] = None) -> Dict[int, float]:
        """Compute Normalized Discounted Cumulative Gain (NDCG@K)."""
        ndcg_at_k = {}
        
        for k in self.k_values:
            total_ndcg = 0
            valid_queries = 0
            
            for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
                if len(truth) == 0:
                    continue
                
                truth_set = set(truth)
                pred_k = pred[:k]
                
                # Use relevance scores if provided, otherwise binary relevance
                if relevance_scores and i < len(relevance_scores):
                    rel_scores = relevance_scores[i][:k]
                else:
                    rel_scores = [1.0 if item in truth_set else 0.0 for item in pred_k]
                
                # Compute DCG
                dcg = 0
                for j, rel in enumerate(rel_scores):
                    dcg += rel / np.log2(j + 2)
                
                # Compute IDCG (ideal DCG)
                ideal_scores = sorted([1.0] * len(truth), reverse=True)[:k]
                idcg = 0
                for j, rel in enumerate(ideal_scores):
                    idcg += rel / np.log2(j + 2)
                
                # Compute NDCG
                ndcg = dcg / idcg if idcg > 0 else 0.0
                total_ndcg += ndcg
                valid_queries += 1
            
            ndcg_at_k[k] = total_ndcg / valid_queries if valid_queries > 0 else 0.0
        
        return ndcg_at_k
    
    def _compute_mean_reciprocal_rank(self, 
                                    predictions: List[List[str]],
                                    ground_truth: List[List[str]]) -> float:
        """Compute Mean Reciprocal Rank (MRR)."""
        total_rr = 0
        valid_queries = 0
        
        for pred, truth in zip(predictions, ground_truth):
            if len(truth) == 0:
                continue
            
            truth_set = set(truth)
            rr = 0
            
            for i, item in enumerate(pred):
                if item in truth_set:
                    rr = 1 / (i + 1)
                    break
            
            total_rr += rr
            valid_queries += 1
        
        return total_rr / valid_queries if valid_queries > 0 else 0.0
    
    def _compute_hit_rate_at_k(self, 
                             predictions: List[List[str]],
                             ground_truth: List[List[str]]) -> Dict[int, float]:
        """Compute Hit Rate@K for different K values."""
        hit_rate_at_k = {}
        
        for k in self.k_values:
            total_hits = 0
            valid_queries = 0
            
            for pred, truth in zip(predictions, ground_truth):
                if len(truth) == 0:
                    continue
                
                pred_k = set(pred[:k])
                truth_set = set(truth)
                
                hit = 1 if len(pred_k & truth_set) > 0 else 0
                total_hits += hit
                valid_queries += 1
            
            hit_rate_at_k[k] = total_hits / valid_queries if valid_queries > 0 else 0.0
        
        return hit_rate_at_k
    
    def compare_methods(self, 
                       results: Dict[str, RetrievalMetrics],
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare multiple retrieval methods.
        
        Args:
            results: Dictionary mapping method names to their metrics
            save_path: Optional path to save the comparison plot
            
        Returns:
            Matplotlib figure with comparison plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot Recall@K
        ax = axes[0, 0]
        for method_name, metrics in results.items():
            k_vals = list(metrics.recall_at_k.keys())
            recall_vals = list(metrics.recall_at_k.values())
            ax.plot(k_vals, recall_vals, marker='o', label=method_name)
        ax.set_xlabel('K')
        ax.set_ylabel('Recall@K')
        ax.set_title('Recall@K Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot Precision@K
        ax = axes[0, 1]
        for method_name, metrics in results.items():
            k_vals = list(metrics.precision_at_k.keys())
            precision_vals = list(metrics.precision_at_k.values())
            ax.plot(k_vals, precision_vals, marker='s', label=method_name)
        ax.set_xlabel('K')
        ax.set_ylabel('Precision@K')
        ax.set_title('Precision@K Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot NDCG@K
        ax = axes[1, 0]
        for method_name, metrics in results.items():
            k_vals = list(metrics.ndcg_at_k.keys())
            ndcg_vals = list(metrics.ndcg_at_k.values())
            ax.plot(k_vals, ndcg_vals, marker='^', label=method_name)
        ax.set_xlabel('K')
        ax.set_ylabel('NDCG@K')
        ax.set_title('NDCG@K Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot overall metrics
        ax = axes[1, 1]
        methods = list(results.keys())
        map_scores = [results[m].mean_average_precision for m in methods]
        mrr_scores = [results[m].mean_reciprocal_rank for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax.bar(x - width/2, map_scores, width, label='mAP', alpha=0.8)
        ax.bar(x + width/2, mrr_scores, width, label='MRR', alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title('Overall Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class EmbeddingVisualizer:
    """
    Utility class for visualizing embeddings and similarity relationships.
    
    Provides methods for dimensionality reduction and visualization of
    high-dimensional embeddings using t-SNE, UMAP, and PCA.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize visualizer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
    
    def plot_embeddings_2d(self, 
                          embeddings: np.ndarray,
                          labels: Optional[np.ndarray] = None,
                          method: str = "tsne",
                          save_path: Optional[str] = None,
                          title: str = "Embedding Visualization") -> plt.Figure:
        """
        Plot 2D visualization of embeddings.
        
        Args:
            embeddings: High-dimensional embeddings
            labels: Optional labels for coloring
            method: Dimensionality reduction method ("tsne", "umap", "pca")
            save_path: Optional path to save the plot
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Reduce dimensionality
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=self.random_state, perplexity=30)
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method == "umap":
            if umap is None:
                raise ImportError("UMAP not installed. Install with: pip install umap-learn")
            reducer = umap.UMAP(n_components=2, random_state=self.random_state)
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=self.random_state)
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                          c=[colors[i]], label=str(label), alpha=0.6, s=50)
            
            ax.legend()
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=50)
        
        ax.set_title(f"{title} ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_similarity_matrix(self, 
                             embeddings: np.ndarray,
                             labels: Optional[List[str]] = None,
                             metric: str = "cosine",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot similarity matrix heatmap.
        
        Args:
            embeddings: Embeddings to compute similarity for
            labels: Optional labels for the items
            metric: Similarity metric ("cosine", "euclidean", "dot")
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Compute similarity matrix
        if metric == "cosine":
            similarity_matrix = cosine_similarity(embeddings)
        elif metric == "euclidean":
            similarity_matrix = -euclidean_distances(embeddings)  # Negative for similarity
        elif metric == "dot":
            similarity_matrix = np.dot(embeddings, embeddings.T)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(similarity_matrix, 
                   xticklabels=labels, 
                   yticklabels=labels,
                   cmap='viridis',
                   center=0,
                   ax=ax)
        
        ax.set_title(f"Similarity Matrix ({metric.capitalize()})")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_nearest_neighbors(self, 
                             query_embedding: np.ndarray,
                             database_embeddings: np.ndarray,
                             k: int = 10,
                             labels: Optional[List[str]] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize nearest neighbors for a query.
        
        Args:
            query_embedding: Query embedding
            database_embeddings: Database embeddings
            k: Number of nearest neighbors
            labels: Optional labels for database items
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Compute similarities
        similarities = cosine_similarity(query_embedding.reshape(1, -1), database_embeddings)[0]
        
        # Get top k neighbors
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_similarities = similarities[top_k_indices]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_labels = [labels[i] if labels else f"Item {i}" for i in top_k_indices]
        bars = ax.bar(range(k), top_k_similarities, alpha=0.7)
        
        # Color bars based on similarity
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(top_k_similarities[i]))
        
        ax.set_xlabel("Nearest Neighbors")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"Top {k} Nearest Neighbors")
        ax.set_xticks(range(k))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add similarity values on top of bars
        for i, (bar, sim) in enumerate(zip(bars, top_k_similarities)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{sim:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# Utility functions
def evaluate_retrieval_system(predictions: List[List[str]],
                            ground_truth: List[List[str]],
                            k_values: List[int] = [1, 5, 10, 20, 50]) -> RetrievalMetrics:
    """
    Evaluate a retrieval system with standard metrics.
    
    Args:
        predictions: Predicted items for each query
        ground_truth: Ground truth relevant items for each query
        k_values: K values for evaluation
        
    Returns:
        RetrievalMetrics object
    """
    evaluator = RetrievalEvaluator(k_values)
    return evaluator.evaluate(predictions, ground_truth)


def mine_hard_negatives(embeddings: torch.Tensor,
                       labels: torch.Tensor,
                       strategy: str = "semi_hard") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mine hard negatives for contrastive learning.
    
    Args:
        embeddings: Embedding vectors
        labels: Corresponding labels
        strategy: Mining strategy
        
    Returns:
        Triplet indices (anchor, positive, negative)
    """
    miner = HardNegativeMiner(strategy=strategy)
    return miner.mine_triplets(embeddings, labels)


def compute_embedding_statistics(embeddings: np.ndarray) -> Dict[str, Any]:
    """
    Compute statistics for embedding vectors.
    
    Args:
        embeddings: Embedding vectors
        
    Returns:
        Dictionary with embedding statistics
    """
    stats = {
        'num_embeddings': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'mean_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
        'std_norm': np.std(np.linalg.norm(embeddings, axis=1)),
        'mean_cosine_similarity': np.mean(cosine_similarity(embeddings)),
        'std_cosine_similarity': np.std(cosine_similarity(embeddings))
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    import torch
    
    # Create sample data
    embeddings = torch.randn(100, 128)
    labels = torch.randint(0, 10, (100,))
    
    # Test hard negative mining
    miner = HardNegativeMiner(strategy="semi_hard")
    anchor_idx, pos_idx, neg_idx = miner.mine_triplets(embeddings, labels)
    
    print(f"Mined {len(anchor_idx)} triplets")
    
    # Test triplet loss
    triplet_loss = miner.compute_triplet_loss(embeddings, labels)
    print(f"Triplet loss: {triplet_loss.item():.4f}")
    
    # Test evaluation
    predictions = [["item1", "item2", "item3"], ["item4", "item5", "item6"]]
    ground_truth = [["item1", "item3"], ["item4"]]
    
    evaluator = RetrievalEvaluator()
    metrics = evaluator.evaluate(predictions, ground_truth)
    
    print(f"Recall@5: {metrics.recall_at_k[5]:.4f}")
    print(f"mAP: {metrics.mean_average_precision:.4f}")
    print(f"MRR: {metrics.mean_reciprocal_rank:.4f}")