"""
Ensemble Methods for Fashion Classification

This module provides ensemble learning capabilities for combining multiple
fashion classification models to improve prediction accuracy and robustness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from collections import defaultdict
import logging
from abc import ABC, abstractmethod
from .classifiers import BaseFashionClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseEnsemble(nn.Module, ABC):
    """
    Abstract base class for ensemble methods.
    """
    
    def __init__(self, models: List[BaseFashionClassifier]):
        """
        Initialize base ensemble.
        
        Args:
            models: List of fashion classification models
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.num_classes = models[0].num_classes
        
        # Validate models
        self._validate_models()
    
    def _validate_models(self):
        """Validate that all models have the same number of classes."""
        for i, model in enumerate(self.models):
            if model.num_classes != self.num_classes:
                raise ValueError(f"Model {i} has {model.num_classes} classes, "
                               f"expected {self.num_classes}")
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        pass
    
    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from individual models."""
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                if isinstance(pred, tuple):  # Handle auxiliary outputs
                    pred = pred[0]
                predictions.append(pred)
        return predictions
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get ensemble information."""
        model_info = []
        for i, model in enumerate(self.models):
            if hasattr(model, 'get_model_info'):
                model_info.append(model.get_model_info())
            else:
                model_info.append({'model_name': f'Model_{i}'})
        
        return {
            'ensemble_type': self.__class__.__name__,
            'num_models': self.num_models,
            'num_classes': self.num_classes,
            'models': model_info
        }


class SoftVotingEnsemble(BaseEnsemble):
    """
    Soft voting ensemble that averages class probabilities.
    """
    
    def __init__(
        self,
        models: List[BaseFashionClassifier],
        weights: Optional[List[float]] = None,
        temperature: float = 1.0
    ):
        """
        Initialize soft voting ensemble.
        
        Args:
            models: List of fashion classification models
            weights: Optional weights for each model (default: equal weights)
            temperature: Temperature scaling for softmax (default: 1.0)
        """
        super().__init__(models)
        
        if weights is None:
            weights = [1.0] * self.num_models
        
        if len(weights) != self.num_models:
            raise ValueError(f"Number of weights ({len(weights)}) must match "
                           f"number of models ({self.num_models})")
        
        # Normalize weights
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
        self.temperature = temperature
        
        logger.info(f"Initialized SoftVotingEnsemble with {self.num_models} models")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with soft voting."""
        ensemble_logits = None
        
        for i, model in enumerate(self.models):
            logits = model(x)
            if isinstance(logits, tuple):  # Handle auxiliary outputs
                logits = logits[0]
            
            # Apply temperature scaling
            logits = logits / self.temperature
            
            # Apply model weight
            weighted_logits = logits * self.weights[i]
            
            if ensemble_logits is None:
                ensemble_logits = weighted_logits
            else:
                ensemble_logits += weighted_logits
        
        return ensemble_logits


class HardVotingEnsemble(BaseEnsemble):
    """
    Hard voting ensemble that uses majority vote of class predictions.
    """
    
    def __init__(
        self,
        models: List[BaseFashionClassifier],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize hard voting ensemble.
        
        Args:
            models: List of fashion classification models
            weights: Optional weights for each model (default: equal weights)
        """
        super().__init__(models)
        
        if weights is None:
            weights = [1.0] * self.num_models
        
        if len(weights) != self.num_models:
            raise ValueError(f"Number of weights ({len(weights)}) must match "
                           f"number of models ({self.num_models})")
        
        self.weights = weights
        
        logger.info(f"Initialized HardVotingEnsemble with {self.num_models} models")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with hard voting."""
        batch_size = x.size(0)
        vote_counts = torch.zeros(batch_size, self.num_classes, device=x.device)
        
        for i, model in enumerate(self.models):
            logits = model(x)
            if isinstance(logits, tuple):  # Handle auxiliary outputs
                logits = logits[0]
            
            # Get class predictions
            predictions = torch.argmax(logits, dim=1)
            
            # Add weighted votes
            for j in range(batch_size):
                vote_counts[j, predictions[j]] += self.weights[i]
        
        return vote_counts


class WeightedEnsemble(BaseEnsemble):
    """
    Weighted ensemble that learns optimal weights for model combination.
    """
    
    def __init__(
        self,
        models: List[BaseFashionClassifier],
        learning_rate: float = 0.001,
        regularization: float = 0.01
    ):
        """
        Initialize weighted ensemble.
        
        Args:
            models: List of fashion classification models
            learning_rate: Learning rate for weight optimization
            regularization: L2 regularization strength
        """
        super().__init__(models)
        
        # Learnable weights for each model
        self.model_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        logger.info(f"Initialized WeightedEnsemble with {self.num_models} models")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with learnable weights."""
        # Apply softmax to ensure weights sum to 1
        normalized_weights = F.softmax(self.model_weights, dim=0)
        
        ensemble_logits = None
        
        for i, model in enumerate(self.models):
            logits = model(x)
            if isinstance(logits, tuple):  # Handle auxiliary outputs
                logits = logits[0]
            
            # Apply learned weight
            weighted_logits = logits * normalized_weights[i]
            
            if ensemble_logits is None:
                ensemble_logits = weighted_logits
            else:
                ensemble_logits += weighted_logits
        
        return ensemble_logits
    
    def get_model_weights(self) -> torch.Tensor:
        """Get normalized model weights."""
        return F.softmax(self.model_weights, dim=0)


class StackedEnsemble(BaseEnsemble):
    """
    Stacked ensemble that uses a meta-learner to combine model predictions.
    """
    
    def __init__(
        self,
        models: List[BaseFashionClassifier],
        meta_learner: Optional[nn.Module] = None,
        use_original_features: bool = False
    ):
        """
        Initialize stacked ensemble.
        
        Args:
            models: List of fashion classification models
            meta_learner: Meta-learner model (default: simple linear layer)
            use_original_features: Whether to include original features in meta-learner
        """
        super().__init__(models)
        
        self.use_original_features = use_original_features
        
        # Create meta-learner
        if meta_learner is None:
            input_dim = self.num_models * self.num_classes
            if use_original_features:
                # Assume feature dimension from first model
                feature_dim = getattr(models[0], 'feature_dim', 2048)
                input_dim += feature_dim
            
            self.meta_learner = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(input_dim // 2, self.num_classes)
            )
        else:
            self.meta_learner = meta_learner
        
        logger.info(f"Initialized StackedEnsemble with {self.num_models} models")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with stacked ensemble."""
        # Get predictions from all models
        model_predictions = []
        model_features = []
        
        for model in self.models:
            if self.use_original_features:
                features = model.extract_features(x)
                pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
                model_features.append(pooled_features)
            
            logits = model(x)
            if isinstance(logits, tuple):  # Handle auxiliary outputs
                logits = logits[0]
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            model_predictions.append(probs)
        
        # Concatenate all predictions
        stacked_predictions = torch.cat(model_predictions, dim=1)
        
        # Optionally include original features
        if self.use_original_features:
            stacked_features = torch.cat(model_features, dim=1)
            meta_input = torch.cat([stacked_predictions, stacked_features], dim=1)
        else:
            meta_input = stacked_predictions
        
        # Pass through meta-learner
        final_logits = self.meta_learner(meta_input)
        
        return final_logits


class DynamicEnsemble(BaseEnsemble):
    """
    Dynamic ensemble that selects models based on input characteristics.
    """
    
    def __init__(
        self,
        models: List[BaseFashionClassifier],
        selector_network: Optional[nn.Module] = None,
        top_k: int = 3
    ):
        """
        Initialize dynamic ensemble.
        
        Args:
            models: List of fashion classification models
            selector_network: Network to select models (default: simple CNN)
            top_k: Number of top models to select per input
        """
        super().__init__(models)
        
        self.top_k = min(top_k, self.num_models)
        
        # Create selector network
        if selector_network is None:
            self.selector_network = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_models)
            )
        else:
            self.selector_network = selector_network
        
        logger.info(f"Initialized DynamicEnsemble with {self.num_models} models, top_k={self.top_k}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic model selection."""
        # Get model selection scores
        selection_scores = self.selector_network(x)
        selection_probs = F.softmax(selection_scores, dim=1)
        
        # Select top-k models for each input
        top_k_scores, top_k_indices = torch.topk(selection_probs, self.top_k, dim=1)
        
        batch_size = x.size(0)
        ensemble_logits = torch.zeros(batch_size, self.num_classes, device=x.device)
        
        # Get predictions from all models
        all_predictions = []
        for model in self.models:
            logits = model(x)
            if isinstance(logits, tuple):  # Handle auxiliary outputs
                logits = logits[0]
            all_predictions.append(logits)
        
        # Combine predictions using dynamic weights
        for i in range(batch_size):
            weighted_logits = torch.zeros(self.num_classes, device=x.device)
            
            for j in range(self.top_k):
                model_idx = top_k_indices[i, j]
                weight = top_k_scores[i, j]
                weighted_logits += weight * all_predictions[model_idx][i]
            
            ensemble_logits[i] = weighted_logits
        
        return ensemble_logits


class EnsembleClassifier(nn.Module):
    """
    Main ensemble classifier that provides easy interface for different ensemble methods.
    """
    
    def __init__(
        self,
        models: List[BaseFashionClassifier],
        ensemble_method: str = 'soft_voting',
        **kwargs
    ):
        """
        Initialize ensemble classifier.
        
        Args:
            models: List of fashion classification models
            ensemble_method: Type of ensemble method
            **kwargs: Additional arguments for specific ensemble methods
        """
        super().__init__()
        
        self.models = models
        self.ensemble_method = ensemble_method
        
        # Create ensemble based on method
        if ensemble_method == 'soft_voting':
            self.ensemble = SoftVotingEnsemble(models, **kwargs)
        elif ensemble_method == 'hard_voting':
            self.ensemble = HardVotingEnsemble(models, **kwargs)
        elif ensemble_method == 'weighted':
            self.ensemble = WeightedEnsemble(models, **kwargs)
        elif ensemble_method == 'stacked':
            self.ensemble = StackedEnsemble(models, **kwargs)
        elif ensemble_method == 'dynamic':
            self.ensemble = DynamicEnsemble(models, **kwargs)
        else:
            raise ValueError(f"Unsupported ensemble method: {ensemble_method}")
        
        logger.info(f"Initialized EnsembleClassifier with {ensemble_method} method")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        return self.ensemble(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from individual models."""
        return self.ensemble.get_individual_predictions(x)
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get ensemble information."""
        return self.ensemble.get_ensemble_info()
    
    def set_training_mode(self, mode: bool = True):
        """Set training mode for all models."""
        for model in self.models:
            model.train(mode)
        self.ensemble.train(mode)
    
    def set_eval_mode(self):
        """Set evaluation mode for all models."""
        self.set_training_mode(False)


def create_ensemble(
    models: List[BaseFashionClassifier],
    method: str = 'soft_voting',
    **kwargs
) -> EnsembleClassifier:
    """
    Factory function to create ensemble classifiers.
    
    Args:
        models: List of fashion classification models
        method: Ensemble method ('soft_voting', 'hard_voting', 'weighted', 'stacked', 'dynamic')
        **kwargs: Additional arguments for specific ensemble methods
    
    Returns:
        Ensemble classifier instance
    """
    return EnsembleClassifier(models, method, **kwargs)


def evaluate_ensemble_diversity(
    models: List[BaseFashionClassifier],
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate diversity metrics for ensemble models.
    
    Args:
        models: List of fashion classification models
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
    
    Returns:
        Dictionary of diversity metrics
    """
    all_predictions = []
    all_correct = []
    
    for model in models:
        model.eval()
        predictions = []
        correct = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                if isinstance(output, tuple):  # Handle auxiliary outputs
                    output = output[0]
                
                pred = torch.argmax(output, dim=1)
                predictions.extend(pred.cpu().numpy())
                correct.extend((pred == target).cpu().numpy())
        
        all_predictions.append(predictions)
        all_correct.append(correct)
    
    # Calculate diversity metrics
    predictions_array = np.array(all_predictions)
    correct_array = np.array(all_correct)
    
    # Pairwise diversity
    pairwise_diversity = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            # Calculate disagreement
            disagreement = np.mean(predictions_array[i] != predictions_array[j])
            pairwise_diversity.append(disagreement)
    
    # Q-statistic (Yule's Q)
    q_statistics = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            # Calculate Q-statistic
            n11 = np.sum((correct_array[i] == 1) & (correct_array[j] == 1))
            n10 = np.sum((correct_array[i] == 1) & (correct_array[j] == 0))
            n01 = np.sum((correct_array[i] == 0) & (correct_array[j] == 1))
            n00 = np.sum((correct_array[i] == 0) & (correct_array[j] == 0))
            
            if (n11 * n00 + n10 * n01) > 0:
                q = (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01)
                q_statistics.append(q)
    
    return {
        'average_pairwise_diversity': np.mean(pairwise_diversity),
        'average_q_statistic': np.mean(q_statistics) if q_statistics else 0.0,
        'individual_accuracies': [np.mean(correct) for correct in all_correct]
    }


# Export all classes and functions
__all__ = [
    'BaseEnsemble',
    'SoftVotingEnsemble',
    'HardVotingEnsemble',
    'WeightedEnsemble',
    'StackedEnsemble',
    'DynamicEnsemble',
    'EnsembleClassifier',
    'create_ensemble',
    'evaluate_ensemble_diversity'
]