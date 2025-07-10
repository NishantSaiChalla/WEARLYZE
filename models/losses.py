"""
Custom Loss Functions for Fashion Classification

This module provides specialized loss functions optimized for fashion classification
tasks, including handling class imbalance, label smoothing, and fashion-specific
loss adaptations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in fashion classification.
    
    This loss function reduces the relative loss for well-classified examples
    and focuses training on hard negatives.
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (default: None)
            gamma: Focusing parameter (default: 2.0)
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Focal Loss.
        
        Args:
            inputs: Predictions (logits) of shape (N, C)
            targets: Ground truth labels of shape (N,)
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for regularization in fashion classification.
    
    This loss function applies label smoothing to reduce overfitting and
    improve generalization.
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        """
        Initialize Label Smoothing Loss.
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing parameter (default: 0.1)
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Label Smoothing Loss.
        
        Args:
            inputs: Predictions (logits) of shape (N, C)
            targets: Ground truth labels of shape (N,)
        
        Returns:
            Label smoothing loss value
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed labels
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = -true_dist * log_probs
        
        if self.reduction == 'mean':
            return loss.sum(dim=1).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.sum(dim=1)


class FashionTripletLoss(nn.Module):
    """
    Triplet Loss for fashion similarity learning.
    
    This loss function learns embeddings such that similar fashion items
    are closer and dissimilar items are farther apart.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        mining_strategy: str = 'hard',
        reduction: str = 'mean'
    ):
        """
        Initialize Triplet Loss.
        
        Args:
            margin: Margin for triplet loss (default: 1.0)
            mining_strategy: Strategy for mining triplets ('hard', 'semi_hard', 'easy')
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        self.reduction = reduction
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for Triplet Loss.
        
        Args:
            embeddings: Feature embeddings of shape (N, D)
            labels: Ground truth labels of shape (N,)
        
        Returns:
            Triplet loss value
        """
        # Calculate pairwise distances
        dist_matrix = self._pairwise_distance(embeddings)
        
        # Mine triplets based on strategy
        if self.mining_strategy == 'hard':
            triplets = self._hard_mining(dist_matrix, labels)
        elif self.mining_strategy == 'semi_hard':
            triplets = self._semi_hard_mining(dist_matrix, labels)
        else:  # easy
            triplets = self._easy_mining(dist_matrix, labels)
        
        if len(triplets) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Calculate triplet loss
        anchor_indices, positive_indices, negative_indices = zip(*triplets)
        
        anchor_positive_dist = dist_matrix[anchor_indices, positive_indices]
        anchor_negative_dist = dist_matrix[anchor_indices, negative_indices]
        
        losses = F.relu(anchor_positive_dist - anchor_negative_dist + self.margin)
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses
    
    def _pairwise_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Calculate pairwise Euclidean distances."""
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = F.relu(distances)
        
        # Add small epsilon to avoid numerical issues
        distances = torch.sqrt(distances + 1e-12)
        
        return distances
    
    def _hard_mining(self, dist_matrix: torch.Tensor, labels: torch.Tensor) -> List[Tuple[int, int, int]]:
        """Mine hard triplets."""
        triplets = []
        
        for i in range(len(labels)):
            anchor_label = labels[i]
            
            # Find hardest positive (farthest positive)
            positive_mask = (labels == anchor_label) & (torch.arange(len(labels)) != i)
            if positive_mask.sum() > 0:
                positive_distances = dist_matrix[i][positive_mask]
                hardest_positive_idx = torch.argmax(positive_distances)
                hardest_positive = torch.arange(len(labels))[positive_mask][hardest_positive_idx]
                
                # Find hardest negative (closest negative)
                negative_mask = labels != anchor_label
                if negative_mask.sum() > 0:
                    negative_distances = dist_matrix[i][negative_mask]
                    hardest_negative_idx = torch.argmin(negative_distances)
                    hardest_negative = torch.arange(len(labels))[negative_mask][hardest_negative_idx]
                    
                    triplets.append((i, hardest_positive.item(), hardest_negative.item()))
        
        return triplets
    
    def _semi_hard_mining(self, dist_matrix: torch.Tensor, labels: torch.Tensor) -> List[Tuple[int, int, int]]:
        """Mine semi-hard triplets."""
        triplets = []
        
        for i in range(len(labels)):
            anchor_label = labels[i]
            
            # Find all positives and negatives
            positive_mask = (labels == anchor_label) & (torch.arange(len(labels)) != i)
            negative_mask = labels != anchor_label
            
            if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                positive_distances = dist_matrix[i][positive_mask]
                negative_distances = dist_matrix[i][negative_mask]
                
                positive_indices = torch.arange(len(labels))[positive_mask]
                negative_indices = torch.arange(len(labels))[negative_mask]
                
                # For each positive, find semi-hard negatives
                for j, pos_idx in enumerate(positive_indices):
                    pos_dist = positive_distances[j]
                    
                    # Semi-hard negatives: closer than positive but still maintaining margin
                    semi_hard_mask = (negative_distances > pos_dist) & (negative_distances < pos_dist + self.margin)
                    
                    if semi_hard_mask.sum() > 0:
                        semi_hard_negatives = negative_indices[semi_hard_mask]
                        for neg_idx in semi_hard_negatives:
                            triplets.append((i, pos_idx.item(), neg_idx.item()))
        
        return triplets
    
    def _easy_mining(self, dist_matrix: torch.Tensor, labels: torch.Tensor) -> List[Tuple[int, int, int]]:
        """Mine easy triplets."""
        triplets = []
        
        for i in range(len(labels)):
            anchor_label = labels[i]
            
            # Find all positives and negatives
            positive_mask = (labels == anchor_label) & (torch.arange(len(labels)) != i)
            negative_mask = labels != anchor_label
            
            if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                positive_indices = torch.arange(len(labels))[positive_mask]
                negative_indices = torch.arange(len(labels))[negative_mask]
                
                # Sample random positive and negative
                pos_idx = positive_indices[torch.randint(len(positive_indices), (1,))]
                neg_idx = negative_indices[torch.randint(len(negative_indices), (1,))]
                
                triplets.append((i, pos_idx.item(), neg_idx.item()))
        
        return triplets


class FashionContrastiveLoss(nn.Module):
    """
    Contrastive Loss for fashion similarity learning.
    
    This loss function learns embeddings by contrasting similar and dissimilar
    fashion item pairs.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Contrastive Loss.
        
        Args:
            margin: Margin for contrastive loss (default: 1.0)
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for Contrastive Loss.
        
        Args:
            embeddings1: First set of embeddings of shape (N, D)
            embeddings2: Second set of embeddings of shape (N, D)
            labels: Binary labels (1 for similar, 0 for dissimilar) of shape (N,)
        
        Returns:
            Contrastive loss value
        """
        # Calculate Euclidean distance
        distance = F.pairwise_distance(embeddings1, embeddings2)
        
        # Calculate contrastive loss
        positive_loss = labels.float() * torch.pow(distance, 2)
        negative_loss = (1 - labels.float()) * torch.pow(F.relu(self.margin - distance), 2)
        
        losses = positive_loss + negative_loss
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses


class FashionCenterLoss(nn.Module):
    """
    Center Loss for fashion classification with intra-class feature compactness.
    
    This loss function learns class centers and minimizes the distance between
    features and their corresponding class centers.
    """
    
    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        alpha: float = 0.5,
        reduction: str = 'mean'
    ):
        """
        Initialize Center Loss.
        
        Args:
            num_classes: Number of classes
            feature_dim: Feature dimension
            alpha: Learning rate for center updates (default: 0.5)
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.reduction = reduction
        
        # Initialize centers
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.kaiming_uniform_(self.centers)
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Center Loss.
        
        Args:
            features: Feature embeddings of shape (N, D)
            labels: Ground truth labels of shape (N,)
        
        Returns:
            Center loss value
        """
        batch_size = features.size(0)
        
        # Calculate distances to centers
        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        
        distmat.addmm_(features, self.centers.t(), beta=1, alpha=-2)
        
        # Select distances to correct centers
        classes = torch.arange(self.num_classes).long()
        if features.is_cuda:
            classes = classes.cuda()
        
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * batch_size
        else:
            return dist.sum(dim=1)


class FashionArcFaceLoss(nn.Module):
    """
    ArcFace Loss for fashion classification with angular margin.
    
    This loss function adds angular margin to the classification loss,
    improving inter-class separability.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        margin: float = 0.5,
        scale: float = 64.0,
        reduction: str = 'mean'
    ):
        """
        Initialize ArcFace Loss.
        
        Args:
            in_features: Input feature dimension
            out_features: Number of classes
            margin: Angular margin (default: 0.5)
            scale: Feature scale (default: 64.0)
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.scale = scale
        self.reduction = reduction
        
        # Initialize weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ArcFace Loss.
        
        Args:
            features: Feature embeddings of shape (N, D)
            labels: Ground truth labels of shape (N,)
        
        Returns:
            ArcFace loss value
        """
        # Normalize features and weights
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Calculate cosine similarity
        cosine = F.linear(features, weight)
        
        # Calculate angle
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * torch.cos(torch.tensor(self.margin)) - sine * torch.sin(torch.tensor(self.margin))
        
        # Create one-hot labels
        one_hot = torch.zeros(cosine.size(), device=features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(output, labels, reduction=self.reduction)
        
        return loss


class FashionMixupLoss(nn.Module):
    """
    Mixup Loss for fashion classification with data augmentation.
    
    This loss function applies mixup augmentation and calculates the
    corresponding mixed loss.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Mixup Loss.
        
        Args:
            alpha: Beta distribution parameter (default: 1.0)
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        """
        Forward pass for Mixup Loss.
        
        Args:
            predictions: Model predictions
            targets_a: First set of targets
            targets_b: Second set of targets
            lam: Mixup parameter
        
        Returns:
            Mixed loss value
        """
        loss_a = F.cross_entropy(predictions, targets_a, reduction=self.reduction)
        loss_b = F.cross_entropy(predictions, targets_b, reduction=self.reduction)
        
        return lam * loss_a + (1 - lam) * loss_b


class FashionCompositeLoss(nn.Module):
    """
    Composite Loss that combines multiple loss functions for fashion classification.
    
    This loss function allows combining different loss types with different weights.
    """
    
    def __init__(
        self,
        loss_functions: Dict[str, nn.Module],
        loss_weights: Dict[str, float],
        reduction: str = 'mean'
    ):
        """
        Initialize Composite Loss.
        
        Args:
            loss_functions: Dictionary of loss function names to loss modules
            loss_weights: Dictionary of loss function names to weights
            reduction: Specifies the reduction to apply to the output
        """
        super().__init__()
        self.loss_functions = nn.ModuleDict(loss_functions)
        self.loss_weights = loss_weights
        self.reduction = reduction
        
        # Validate weights
        for name in loss_functions.keys():
            if name not in loss_weights:
                raise ValueError(f"Weight not provided for loss function: {name}")
    
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Composite Loss.
        
        Args:
            **kwargs: Arguments for individual loss functions
        
        Returns:
            Dictionary of individual and total loss values
        """
        losses = {}
        total_loss = 0
        
        for name, loss_fn in self.loss_functions.items():
            if name in kwargs:
                loss_value = loss_fn(**kwargs[name])
                weighted_loss = self.loss_weights[name] * loss_value
                losses[name] = loss_value
                total_loss += weighted_loss
        
        losses['total'] = total_loss
        return losses


def create_fashion_loss(
    loss_type: str,
    num_classes: int = 50,
    **kwargs
) -> nn.Module:
    """
    Factory function to create fashion loss functions.
    
    Args:
        loss_type: Type of loss function
        num_classes: Number of fashion classes
        **kwargs: Additional arguments for specific loss functions
    
    Returns:
        Loss function instance
    
    Raises:
        ValueError: If loss_type is not supported
    """
    loss_registry = {
        'focal': FocalLoss,
        'label_smoothing': lambda **args: LabelSmoothingLoss(num_classes, **args),
        'triplet': FashionTripletLoss,
        'contrastive': FashionContrastiveLoss,
        'center': lambda **args: FashionCenterLoss(num_classes, **args),
        'arcface': lambda **args: FashionArcFaceLoss(out_features=num_classes, **args),
        'mixup': FashionMixupLoss,
        'cross_entropy': nn.CrossEntropyLoss
    }
    
    if loss_type not in loss_registry:
        raise ValueError(f"Unsupported loss type: {loss_type}. "
                        f"Supported types: {list(loss_registry.keys())}")
    
    loss_fn = loss_registry[loss_type]
    
    if callable(loss_fn):
        return loss_fn(**kwargs)
    else:
        return loss_fn


# Export all classes and functions
__all__ = [
    'FocalLoss',
    'LabelSmoothingLoss',
    'FashionTripletLoss',
    'FashionContrastiveLoss',
    'FashionCenterLoss',
    'FashionArcFaceLoss',
    'FashionMixupLoss',
    'FashionCompositeLoss',
    'create_fashion_loss'
]