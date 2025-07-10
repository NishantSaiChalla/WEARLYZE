"""
Fashion Classification Models

This module provides various deep learning models for fashion classification tasks,
including ResNet, MobileNet, ConvNeXt, and Vision Transformer architectures.
All models are fine-tuned for fashion category classification with 50 classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from typing import Dict, List, Optional, Tuple, Union, Any
import timm
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseFashionClassifier(nn.Module, ABC):
    """
    Abstract base class for fashion classification models.
    
    This class provides common functionality for all fashion classifiers including
    feature extraction, classification head, and utility methods.
    """
    
    def __init__(
        self,
        num_classes: int = 50,
        pretrained: bool = True,
        dropout_rate: float = 0.1,
        use_auxiliary_head: bool = False
    ):
        """
        Initialize the base fashion classifier.
        
        Args:
            num_classes: Number of fashion categories (default: 50)
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout probability for regularization
            use_auxiliary_head: Whether to use auxiliary classification head
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.use_auxiliary_head = use_auxiliary_head
        
        # Initialize backbone and heads
        self.backbone = self._create_backbone()
        self.feature_dim = self._get_feature_dim()
        self.classifier = self._create_classifier_head()
        
        if self.use_auxiliary_head:
            self.auxiliary_classifier = self._create_auxiliary_head()
        
        # Initialize weights
        self._initialize_weights()
    
    @abstractmethod
    def _create_backbone(self) -> nn.Module:
        """Create the backbone network."""
        pass
    
    @abstractmethod
    def _get_feature_dim(self) -> int:
        """Get the feature dimension of the backbone."""
        pass
    
    def _create_classifier_head(self) -> nn.Module:
        """Create the main classification head."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim // 2, self.num_classes)
        )
    
    def _create_auxiliary_head(self) -> nn.Module:
        """Create auxiliary classification head for training stability."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim, self.num_classes)
        )
    
    def _initialize_weights(self):
        """Initialize weights for newly added layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone."""
        return self.backbone(x)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the model."""
        features = self.extract_features(x)
        main_output = self.classifier(features)
        
        if self.use_auxiliary_head and self.training:
            aux_output = self.auxiliary_classifier(features)
            return main_output, aux_output
        
        return main_output
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including parameters and memory usage."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_dim': self.feature_dim,
            'dropout_rate': self.dropout_rate,
            'use_auxiliary_head': self.use_auxiliary_head
        }


class FashionResNet(BaseFashionClassifier):
    """
    ResNet-based fashion classifier.
    
    This model uses ResNet-50 as the backbone and adds a custom classification head
    optimized for fashion category recognition.
    """
    
    def __init__(
        self,
        num_classes: int = 50,
        pretrained: bool = True,
        dropout_rate: float = 0.1,
        use_auxiliary_head: bool = False,
        variant: str = 'resnet50'
    ):
        """
        Initialize FashionResNet.
        
        Args:
            num_classes: Number of fashion categories
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout probability for regularization
            use_auxiliary_head: Whether to use auxiliary classification head
            variant: ResNet variant ('resnet50', 'resnet101', 'resnet152')
        """
        self.variant = variant
        super().__init__(num_classes, pretrained, dropout_rate, use_auxiliary_head)
    
    def _create_backbone(self) -> nn.Module:
        """Create ResNet backbone."""
        backbone = timm.create_model(
            self.variant,
            pretrained=self.pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        return backbone
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension based on ResNet variant."""
        if self.variant in ['resnet50', 'resnet101', 'resnet152']:
            return 2048
        else:
            return 512  # For ResNet-18, ResNet-34


class FashionMobileNet(BaseFashionClassifier):
    """
    MobileNet-based fashion classifier for efficient deployment.
    
    This model uses MobileNet-V3 as the backbone, optimized for mobile and edge devices
    while maintaining competitive accuracy for fashion classification.
    """
    
    def __init__(
        self,
        num_classes: int = 50,
        pretrained: bool = True,
        dropout_rate: float = 0.1,
        use_auxiliary_head: bool = False,
        variant: str = 'mobilenetv3_large_100'
    ):
        """
        Initialize FashionMobileNet.
        
        Args:
            num_classes: Number of fashion categories
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout probability for regularization
            use_auxiliary_head: Whether to use auxiliary classification head
            variant: MobileNet variant ('mobilenetv3_large_100', 'mobilenetv3_small_100')
        """
        self.variant = variant
        super().__init__(num_classes, pretrained, dropout_rate, use_auxiliary_head)
    
    def _create_backbone(self) -> nn.Module:
        """Create MobileNet backbone."""
        backbone = timm.create_model(
            self.variant,
            pretrained=self.pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        return backbone
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension based on MobileNet variant."""
        if 'large' in self.variant:
            return 960
        else:  # small variant
            return 576


class FashionConvNeXt(BaseFashionClassifier):
    """
    ConvNeXt-based fashion classifier for state-of-the-art performance.
    
    This model uses ConvNeXt-Tiny as the backbone, providing excellent accuracy
    with reasonable computational requirements.
    """
    
    def __init__(
        self,
        num_classes: int = 50,
        pretrained: bool = True,
        dropout_rate: float = 0.1,
        use_auxiliary_head: bool = False,
        variant: str = 'convnext_tiny'
    ):
        """
        Initialize FashionConvNeXt.
        
        Args:
            num_classes: Number of fashion categories
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout probability for regularization
            use_auxiliary_head: Whether to use auxiliary classification head
            variant: ConvNeXt variant ('convnext_tiny', 'convnext_small', 'convnext_base')
        """
        self.variant = variant
        super().__init__(num_classes, pretrained, dropout_rate, use_auxiliary_head)
    
    def _create_backbone(self) -> nn.Module:
        """Create ConvNeXt backbone."""
        backbone = timm.create_model(
            self.variant,
            pretrained=self.pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        return backbone
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension based on ConvNeXt variant."""
        variant_dims = {
            'convnext_tiny': 768,
            'convnext_small': 768,
            'convnext_base': 1024,
            'convnext_large': 1536
        }
        return variant_dims.get(self.variant, 768)


class FashionViT(BaseFashionClassifier):
    """
    Vision Transformer-based fashion classifier.
    
    This model uses Vision Transformer as the backbone, providing excellent performance
    for fashion classification tasks with attention mechanisms.
    """
    
    def __init__(
        self,
        num_classes: int = 50,
        pretrained: bool = True,
        dropout_rate: float = 0.1,
        use_auxiliary_head: bool = False,
        variant: str = 'vit_base_patch16_224',
        patch_size: int = 16,
        embed_dim: int = 768
    ):
        """
        Initialize FashionViT.
        
        Args:
            num_classes: Number of fashion categories
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout probability for regularization
            use_auxiliary_head: Whether to use auxiliary classification head
            variant: ViT variant ('vit_base_patch16_224', 'vit_small_patch16_224')
            patch_size: Size of image patches
            embed_dim: Embedding dimension
        """
        self.variant = variant
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        super().__init__(num_classes, pretrained, dropout_rate, use_auxiliary_head)
    
    def _create_backbone(self) -> nn.Module:
        """Create Vision Transformer backbone."""
        backbone = timm.create_model(
            self.variant,
            pretrained=self.pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        return backbone
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension based on ViT variant."""
        variant_dims = {
            'vit_tiny_patch16_224': 192,
            'vit_small_patch16_224': 384,
            'vit_base_patch16_224': 768,
            'vit_large_patch16_224': 1024
        }
        return variant_dims.get(self.variant, self.embed_dim)
    
    def _create_classifier_head(self) -> nn.Module:
        """Create classifier head for Vision Transformer."""
        return nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim // 2, self.num_classes)
        )
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from ViT backbone."""
        features = self.backbone(x)
        # For ViT, we typically use the class token
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
            # Extract class token (first token)
            if features.dim() == 3:  # [batch, seq_len, embed_dim]
                features = features[:, 0]  # Class token
        return features


class FashionEfficientNet(BaseFashionClassifier):
    """
    EfficientNet-based fashion classifier for balanced performance and efficiency.
    
    This model uses EfficientNet as the backbone, providing good accuracy with
    optimal computational efficiency.
    """
    
    def __init__(
        self,
        num_classes: int = 50,
        pretrained: bool = True,
        dropout_rate: float = 0.1,
        use_auxiliary_head: bool = False,
        variant: str = 'efficientnet_b0'
    ):
        """
        Initialize FashionEfficientNet.
        
        Args:
            num_classes: Number of fashion categories
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout probability for regularization
            use_auxiliary_head: Whether to use auxiliary classification head
            variant: EfficientNet variant ('efficientnet_b0' to 'efficientnet_b7')
        """
        self.variant = variant
        super().__init__(num_classes, pretrained, dropout_rate, use_auxiliary_head)
    
    def _create_backbone(self) -> nn.Module:
        """Create EfficientNet backbone."""
        backbone = timm.create_model(
            self.variant,
            pretrained=self.pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        return backbone
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension based on EfficientNet variant."""
        variant_dims = {
            'efficientnet_b0': 1280,
            'efficientnet_b1': 1280,
            'efficientnet_b2': 1408,
            'efficientnet_b3': 1536,
            'efficientnet_b4': 1792,
            'efficientnet_b5': 2048,
            'efficientnet_b6': 2304,
            'efficientnet_b7': 2560
        }
        return variant_dims.get(self.variant, 1280)


class FashionMultiScale(BaseFashionClassifier):
    """
    Multi-scale fashion classifier that processes images at different scales.
    
    This model combines features from multiple scales to improve classification
    accuracy for fashion items with varying sizes and details.
    """
    
    def __init__(
        self,
        num_classes: int = 50,
        pretrained: bool = True,
        dropout_rate: float = 0.1,
        use_auxiliary_head: bool = False,
        backbone_type: str = 'resnet50',
        scales: List[int] = [224, 288, 384]
    ):
        """
        Initialize FashionMultiScale.
        
        Args:
            num_classes: Number of fashion categories
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout probability for regularization
            use_auxiliary_head: Whether to use auxiliary classification head
            backbone_type: Type of backbone network
            scales: List of input image scales
        """
        self.backbone_type = backbone_type
        self.scales = scales
        super().__init__(num_classes, pretrained, dropout_rate, use_auxiliary_head)
    
    def _create_backbone(self) -> nn.Module:
        """Create multi-scale backbone."""
        backbone = timm.create_model(
            self.backbone_type,
            pretrained=self.pretrained,
            num_classes=0,
            global_pool=''
        )
        return backbone
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension multiplied by number of scales."""
        base_dims = {
            'resnet50': 2048,
            'mobilenetv3_large_100': 960,
            'efficientnet_b0': 1280
        }
        base_dim = base_dims.get(self.backbone_type, 2048)
        return base_dim * len(self.scales)
    
    def _create_classifier_head(self) -> nn.Module:
        """Create classifier head for multi-scale features."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim // 4, self.num_classes)
        )
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale features."""
        batch_size = x.size(0)
        multi_scale_features = []
        
        for scale in self.scales:
            # Resize input to current scale
            if x.size(-1) != scale:
                scaled_x = F.interpolate(x, size=(scale, scale), mode='bilinear', align_corners=False)
            else:
                scaled_x = x
            
            # Extract features at current scale
            features = self.backbone(scaled_x)
            pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
            multi_scale_features.append(pooled_features)
        
        # Concatenate multi-scale features
        combined_features = torch.cat(multi_scale_features, dim=1)
        return combined_features.view(batch_size, -1, 1, 1)


def create_fashion_classifier(
    model_type: str,
    num_classes: int = 50,
    pretrained: bool = True,
    **kwargs
) -> BaseFashionClassifier:
    """
    Factory function to create fashion classifiers.
    
    Args:
        model_type: Type of model to create
        num_classes: Number of fashion categories
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments for specific models
    
    Returns:
        Fashion classifier instance
    
    Raises:
        ValueError: If model_type is not supported
    """
    model_registry = {
        'resnet': FashionResNet,
        'mobilenet': FashionMobileNet,
        'convnext': FashionConvNeXt,
        'vit': FashionViT,
        'efficientnet': FashionEfficientNet,
        'multiscale': FashionMultiScale
    }
    
    if model_type not in model_registry:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: {list(model_registry.keys())}")
    
    model_class = model_registry[model_type]
    return model_class(num_classes=num_classes, pretrained=pretrained, **kwargs)


def load_fashion_classifier(
    checkpoint_path: str,
    model_type: str,
    num_classes: int = 50,
    device: str = 'cuda'
) -> BaseFashionClassifier:
    """
    Load a fashion classifier from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_type: Type of model to load
        num_classes: Number of fashion categories
        device: Device to load model on
    
    Returns:
        Loaded fashion classifier
    """
    model = create_fashion_classifier(model_type, num_classes, pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded {model_type} model from {checkpoint_path}")
    return model


# Export all classes and functions
__all__ = [
    'BaseFashionClassifier',
    'FashionResNet',
    'FashionMobileNet',
    'FashionConvNeXt',
    'FashionViT',
    'FashionEfficientNet',
    'FashionMultiScale',
    'create_fashion_classifier',
    'load_fashion_classifier'
]