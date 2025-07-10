"""
Fashion-Specific Embedding Layers for Fashion Detection.

This module provides specialized embedding layers for fashion attributes,
multi-modal fusion techniques, and style/texture feature extractors
optimized for fashion similarity search and retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import warnings

try:
    import torchvision.models as models
    from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
except ImportError:
    warnings.warn("torchvision not available. Some features may not work.")
    models = None

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class FashionEmbeddingConfig:
    """Configuration for fashion embedding layers."""
    
    # Base embedding parameters
    base_embedding_dim: int = 512
    attribute_embedding_dim: int = 128
    style_embedding_dim: int = 256
    texture_embedding_dim: int = 256
    
    # Fashion attributes
    categories: List[str] = None
    colors: List[str] = None
    patterns: List[str] = None
    materials: List[str] = None
    styles: List[str] = None
    seasons: List[str] = None
    
    # Multi-modal fusion
    fusion_method: str = "attention"  # "concat", "attention", "bilinear", "cross_attention"
    dropout_rate: float = 0.1
    
    # Texture analysis
    use_texture_features: bool = True
    texture_patch_size: int = 32
    texture_stride: int = 16
    
    # Style analysis
    use_style_features: bool = True
    style_layers: List[str] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = [
                "dress", "shirt", "pants", "skirt", "jacket", "coat", "sweater",
                "blouse", "t-shirt", "jeans", "shorts", "suit", "hoodie", "cardigan",
                "tank-top", "polo", "blazer", "vest", "romper", "jumpsuit"
            ]
        
        if self.colors is None:
            self.colors = [
                "red", "blue", "green", "yellow", "orange", "purple", "pink",
                "brown", "black", "white", "gray", "navy", "beige", "khaki",
                "maroon", "teal", "coral", "burgundy", "olive", "turquoise"
            ]
        
        if self.patterns is None:
            self.patterns = [
                "solid", "striped", "polka-dot", "floral", "geometric", "abstract",
                "animal-print", "plaid", "checkered", "paisley", "tribal", "tie-dye",
                "camouflage", "houndstooth", "argyle", "chevron", "damask"
            ]
        
        if self.materials is None:
            self.materials = [
                "cotton", "polyester", "wool", "silk", "linen", "denim", "leather",
                "cashmere", "satin", "chiffon", "velvet", "lace", "mesh", "fleece",
                "spandex", "bamboo", "modal", "viscose", "nylon", "rayon"
            ]
        
        if self.styles is None:
            self.styles = [
                "casual", "formal", "business", "sporty", "bohemian", "vintage",
                "modern", "classic", "trendy", "minimalist", "romantic", "edgy",
                "preppy", "grunge", "punk", "gothic", "retro", "hippie"
            ]
        
        if self.seasons is None:
            self.seasons = ["spring", "summer", "fall", "winter", "all-season"]
        
        if self.style_layers is None:
            self.style_layers = ["conv2_x", "conv3_x", "conv4_x", "conv5_x"]


class FashionAttributeEmbedding(nn.Module):
    """
    Embedding layer for fashion attributes with learned representations.
    
    Supports multiple attribute types and provides contextual embeddings
    that can be combined with visual features.
    """
    
    def __init__(self, config: FashionEmbeddingConfig):
        super().__init__()
        self.config = config
        
        # Category embedding
        self.category_embedding = nn.Embedding(
            len(config.categories), config.attribute_embedding_dim
        )
        
        # Color embedding
        self.color_embedding = nn.Embedding(
            len(config.colors), config.attribute_embedding_dim
        )
        
        # Pattern embedding
        self.pattern_embedding = nn.Embedding(
            len(config.patterns), config.attribute_embedding_dim
        )
        
        # Material embedding
        self.material_embedding = nn.Embedding(
            len(config.materials), config.attribute_embedding_dim
        )
        
        # Style embedding
        self.style_embedding = nn.Embedding(
            len(config.styles), config.attribute_embedding_dim
        )
        
        # Season embedding
        self.season_embedding = nn.Embedding(
            len(config.seasons), config.attribute_embedding_dim
        )
        
        # Fusion layer
        total_attr_dim = 6 * config.attribute_embedding_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_attr_dim, config.base_embedding_dim),
            nn.LayerNorm(config.base_embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.base_embedding_dim, config.base_embedding_dim)
        )
        
        # Attribute mappings
        self.category_to_idx = {cat: i for i, cat in enumerate(config.categories)}
        self.color_to_idx = {col: i for i, col in enumerate(config.colors)}
        self.pattern_to_idx = {pat: i for i, pat in enumerate(config.patterns)}
        self.material_to_idx = {mat: i for i, mat in enumerate(config.materials)}
        self.style_to_idx = {sty: i for i, sty in enumerate(config.styles)}
        self.season_to_idx = {sea: i for i, sea in enumerate(config.seasons)}
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize embedding weights."""
        for module in [self.category_embedding, self.color_embedding, 
                      self.pattern_embedding, self.material_embedding,
                      self.style_embedding, self.season_embedding]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, attribute_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for attribute embedding.
        
        Args:
            attribute_dict: Dictionary containing attribute indices
            
        Returns:
            Fused attribute embeddings
        """
        embeddings = []
        
        # Get individual attribute embeddings
        if 'category' in attribute_dict:
            cat_emb = self.category_embedding(attribute_dict['category'])
            embeddings.append(cat_emb)
        
        if 'color' in attribute_dict:
            col_emb = self.color_embedding(attribute_dict['color'])
            embeddings.append(col_emb)
        
        if 'pattern' in attribute_dict:
            pat_emb = self.pattern_embedding(attribute_dict['pattern'])
            embeddings.append(pat_emb)
        
        if 'material' in attribute_dict:
            mat_emb = self.material_embedding(attribute_dict['material'])
            embeddings.append(mat_emb)
        
        if 'style' in attribute_dict:
            sty_emb = self.style_embedding(attribute_dict['style'])
            embeddings.append(sty_emb)
        
        if 'season' in attribute_dict:
            sea_emb = self.season_embedding(attribute_dict['season'])
            embeddings.append(sea_emb)
        
        # Pad with zeros if some attributes are missing
        while len(embeddings) < 6:
            embeddings.append(torch.zeros_like(embeddings[0]))
        
        # Concatenate and fuse
        combined_emb = torch.cat(embeddings, dim=-1)
        fused_emb = self.fusion_layer(combined_emb)
        
        return fused_emb
    
    def get_attribute_similarities(self, attribute_type: str) -> torch.Tensor:
        """
        Get similarity matrix for a specific attribute type.
        
        Args:
            attribute_type: Type of attribute ('category', 'color', etc.)
            
        Returns:
            Similarity matrix
        """
        if attribute_type == 'category':
            embeddings = self.category_embedding.weight
        elif attribute_type == 'color':
            embeddings = self.color_embedding.weight
        elif attribute_type == 'pattern':
            embeddings = self.pattern_embedding.weight
        elif attribute_type == 'material':
            embeddings = self.material_embedding.weight
        elif attribute_type == 'style':
            embeddings = self.style_embedding.weight
        elif attribute_type == 'season':
            embeddings = self.season_embedding.weight
        else:
            raise ValueError(f"Unknown attribute type: {attribute_type}")
        
        # Compute cosine similarity
        normalized_emb = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_emb, normalized_emb.T)
        
        return similarity_matrix


class TextureFeatureExtractor(nn.Module):
    """
    Extracts texture features from fashion images using multiple approaches.
    
    Combines traditional texture analysis with learned representations
    for comprehensive texture understanding.
    """
    
    def __init__(self, config: FashionEmbeddingConfig):
        super().__init__()
        self.config = config
        
        # Texture CNN
        self.texture_cnn = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Texture descriptor
        self.texture_descriptor = nn.Sequential(
            nn.Linear(256, config.texture_embedding_dim),
            nn.LayerNorm(config.texture_embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.texture_embedding_dim, config.texture_embedding_dim)
        )
        
        # Local Binary Pattern (LBP) approximation
        self.lbp_conv = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False)
        self._initialize_lbp_weights()
        
        # Gabor filter approximation
        self.gabor_filters = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
            for _ in range(8)  # 8 different orientations
        ])
        self._initialize_gabor_weights()
    
    def _initialize_lbp_weights(self):
        """Initialize LBP-like weights."""
        # Create LBP-like filters
        lbp_kernel = torch.tensor([
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            [[-1, -1, 0], [-1, 0, 1], [0, 1, 1]],
            [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
            [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]],
            [[1, 1, 1], [-1, 0, -1], [-1, -1, -1]],
            [[1, 1, 0], [1, 0, -1], [0, -1, -1]],
            [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
            [[0, -1, -1], [1, 0, -1], [1, 1, 0]]
        ], dtype=torch.float32)
        
        self.lbp_conv.weight.data = lbp_kernel.unsqueeze(1)
        self.lbp_conv.weight.requires_grad = False
    
    def _initialize_gabor_weights(self):
        """Initialize Gabor filter weights."""
        for i, gabor_filter in enumerate(self.gabor_filters):
            # Create Gabor filter
            theta = i * np.pi / 8  # 8 orientations
            sigma = 2.0
            lambd = 4.0
            gamma = 0.5
            
            # Create meshgrid
            size = 7
            x = torch.arange(-size//2 + 1, size//2 + 1, dtype=torch.float32)
            y = torch.arange(-size//2 + 1, size//2 + 1, dtype=torch.float32)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            # Apply rotation
            X_rot = X * np.cos(theta) + Y * np.sin(theta)
            Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
            
            # Compute Gabor filter
            gabor_kernel = torch.exp(-(X_rot**2 + gamma**2 * Y_rot**2) / (2 * sigma**2)) * \
                          torch.cos(2 * np.pi * X_rot / lambd)
            
            gabor_filter.weight.data = gabor_kernel.unsqueeze(0).unsqueeze(0)
            gabor_filter.weight.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract texture features from images.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Texture embeddings [B, texture_embedding_dim]
        """
        batch_size = images.shape[0]
        
        # Extract CNN-based texture features
        texture_features = self.texture_cnn(images)
        texture_features = texture_features.view(batch_size, -1)
        
        # Convert to grayscale for traditional texture analysis
        gray_images = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        gray_images = gray_images.unsqueeze(1)
        
        # Extract LBP features
        lbp_features = self.lbp_conv(gray_images)
        lbp_features = F.adaptive_avg_pool2d(lbp_features, (1, 1)).view(batch_size, -1)
        
        # Extract Gabor features
        gabor_features = []
        for gabor_filter in self.gabor_filters:
            gabor_response = gabor_filter(gray_images)
            gabor_response = F.adaptive_avg_pool2d(gabor_response, (1, 1)).view(batch_size, -1)
            gabor_features.append(gabor_response)
        
        gabor_features = torch.cat(gabor_features, dim=1)
        
        # Combine all texture features
        combined_features = torch.cat([texture_features, lbp_features, gabor_features], dim=1)
        
        # Apply texture descriptor
        texture_embeddings = self.texture_descriptor(combined_features)
        
        return texture_embeddings


class StyleFeatureExtractor(nn.Module):
    """
    Extracts style features using pre-trained networks and style transfer techniques.
    
    Captures high-level style characteristics including color schemes,
    composition, and aesthetic properties.
    """
    
    def __init__(self, config: FashionEmbeddingConfig):
        super().__init__()
        self.config = config
        
        # Pre-trained backbone for style extraction
        if models is not None:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            # Remove final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            # Simple CNN fallback
            self.backbone = self._create_simple_cnn()
        
        # Style layers for different levels of abstraction
        self.style_layers = nn.ModuleDict({
            'low_level': nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            ),
            'mid_level': nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            ),
            'high_level': nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        })
        
        # Style descriptor
        self.style_descriptor = nn.Sequential(
            nn.Linear(2048 + 64 + 128 + 256, config.style_embedding_dim),
            nn.LayerNorm(config.style_embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.style_embedding_dim, config.style_embedding_dim)
        )
        
        # Color histogram analyzer
        self.color_analyzer = ColorHistogramAnalyzer()
        
        # Composition analyzer
        self.composition_analyzer = CompositionAnalyzer()
    
    def _create_simple_cnn(self):
        """Create a simple CNN for style extraction."""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract style features from images.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Style embeddings [B, style_embedding_dim]
        """
        batch_size = images.shape[0]
        
        # Extract backbone features
        backbone_features = self.backbone(images)
        backbone_features = backbone_features.view(batch_size, -1)
        
        # Extract multi-level style features
        style_features = []
        x = images
        
        for layer_name, layer in self.style_layers.items():
            x = layer(x)
            style_features.append(x.view(batch_size, -1))
        
        # Combine all style features
        combined_features = torch.cat([backbone_features] + style_features, dim=1)
        
        # Apply style descriptor
        style_embeddings = self.style_descriptor(combined_features)
        
        return style_embeddings


class ColorHistogramAnalyzer(nn.Module):
    """Analyzes color distribution in fashion images."""
    
    def __init__(self, num_bins: int = 32):
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer('bin_edges', torch.linspace(0, 1, num_bins + 1))
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute color histograms for RGB channels.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Color histogram features [B, num_bins * 3]
        """
        batch_size = images.shape[0]
        histograms = []
        
        for i in range(3):  # RGB channels
            channel = images[:, i].flatten(start_dim=1)
            
            # Compute histogram for each image in batch
            channel_histograms = []
            for j in range(batch_size):
                hist = torch.histogram(channel[j], bins=self.bin_edges)[0]
                hist = hist.float() / hist.sum()  # Normalize
                channel_histograms.append(hist)
            
            histograms.append(torch.stack(channel_histograms))
        
        return torch.cat(histograms, dim=1)


class CompositionAnalyzer(nn.Module):
    """Analyzes composition and spatial layout of fashion images."""
    
    def __init__(self):
        super().__init__()
        
        # Edge detector
        self.edge_detector = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        edge_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
        self.edge_detector.weight.data = edge_kernel.unsqueeze(0).unsqueeze(0)
        self.edge_detector.weight.requires_grad = False
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze composition features.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Dictionary of composition features
        """
        batch_size = images.shape[0]
        
        # Convert to grayscale
        gray_images = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        gray_images = gray_images.unsqueeze(1)
        
        # Detect edges
        edges = self.edge_detector(gray_images)
        edge_density = edges.mean(dim=(2, 3))
        
        # Compute center-surround contrast
        center_region = gray_images[:, :, 
                                  gray_images.shape[2]//4:3*gray_images.shape[2]//4,
                                  gray_images.shape[3]//4:3*gray_images.shape[3]//4]
        center_mean = center_region.mean(dim=(2, 3))
        surround_mean = gray_images.mean(dim=(2, 3))
        contrast = torch.abs(center_mean - surround_mean)
        
        return {
            'edge_density': edge_density,
            'center_surround_contrast': contrast
        }


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion module for combining different types of embeddings.
    
    Supports various fusion strategies including attention-based fusion,
    bilinear pooling, and cross-modal attention.
    """
    
    def __init__(self, config: FashionEmbeddingConfig):
        super().__init__()
        self.config = config
        self.fusion_method = config.fusion_method
        
        # Input dimensions
        self.visual_dim = config.base_embedding_dim
        self.attribute_dim = config.base_embedding_dim
        self.texture_dim = config.texture_embedding_dim
        self.style_dim = config.style_embedding_dim
        
        # Output dimension
        self.output_dim = config.base_embedding_dim
        
        if self.fusion_method == "concat":
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.visual_dim + self.attribute_dim + self.texture_dim + self.style_dim, 
                         self.output_dim),
                nn.LayerNorm(self.output_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            )
        
        elif self.fusion_method == "attention":
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=self.output_dim,
                num_heads=8,
                dropout=config.dropout_rate,
                batch_first=True
            )
            
            # Projection layers
            self.visual_proj = nn.Linear(self.visual_dim, self.output_dim)
            self.attribute_proj = nn.Linear(self.attribute_dim, self.output_dim)
            self.texture_proj = nn.Linear(self.texture_dim, self.output_dim)
            self.style_proj = nn.Linear(self.style_dim, self.output_dim)
        
        elif self.fusion_method == "bilinear":
            self.bilinear_pools = nn.ModuleList([
                nn.Bilinear(self.visual_dim, self.attribute_dim, self.output_dim),
                nn.Bilinear(self.visual_dim, self.texture_dim, self.output_dim),
                nn.Bilinear(self.visual_dim, self.style_dim, self.output_dim)
            ])
            
            self.fusion_layer = nn.Sequential(
                nn.Linear(3 * self.output_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            )
        
        elif self.fusion_method == "cross_attention":
            self.cross_attention_layers = nn.ModuleList([
                nn.MultiheadAttention(self.output_dim, 4, batch_first=True),
                nn.MultiheadAttention(self.output_dim, 4, batch_first=True),
                nn.MultiheadAttention(self.output_dim, 4, batch_first=True)
            ])
            
            # Projection layers
            self.visual_proj = nn.Linear(self.visual_dim, self.output_dim)
            self.attribute_proj = nn.Linear(self.attribute_dim, self.output_dim)
            self.texture_proj = nn.Linear(self.texture_dim, self.output_dim)
            self.style_proj = nn.Linear(self.style_dim, self.output_dim)
    
    def forward(self, 
                visual_features: torch.Tensor,
                attribute_features: torch.Tensor,
                texture_features: torch.Tensor,
                style_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse multi-modal features.
        
        Args:
            visual_features: Visual embeddings
            attribute_features: Attribute embeddings
            texture_features: Texture embeddings
            style_features: Style embeddings
            
        Returns:
            Fused embeddings
        """
        if self.fusion_method == "concat":
            # Simple concatenation
            combined = torch.cat([visual_features, attribute_features, 
                                texture_features, style_features], dim=1)
            return self.fusion_layer(combined)
        
        elif self.fusion_method == "attention":
            # Project to same dimension
            visual_proj = self.visual_proj(visual_features)
            attribute_proj = self.attribute_proj(attribute_features)
            texture_proj = self.texture_proj(texture_features)
            style_proj = self.style_proj(style_features)
            
            # Stack features for attention
            features = torch.stack([visual_proj, attribute_proj, texture_proj, style_proj], dim=1)
            
            # Apply self-attention
            attended_features, _ = self.attention_layer(features, features, features)
            
            # Average pool
            return attended_features.mean(dim=1)
        
        elif self.fusion_method == "bilinear":
            # Bilinear pooling between visual and other modalities
            bilinear_outputs = []
            
            for i, (other_features, bilinear_layer) in enumerate(zip(
                [attribute_features, texture_features, style_features],
                self.bilinear_pools
            )):
                bilinear_output = bilinear_layer(visual_features, other_features)
                bilinear_outputs.append(bilinear_output)
            
            # Combine bilinear outputs
            combined = torch.cat(bilinear_outputs, dim=1)
            return self.fusion_layer(combined)
        
        elif self.fusion_method == "cross_attention":
            # Project to same dimension
            visual_proj = self.visual_proj(visual_features).unsqueeze(1)
            attribute_proj = self.attribute_proj(attribute_features).unsqueeze(1)
            texture_proj = self.texture_proj(texture_features).unsqueeze(1)
            style_proj = self.style_proj(style_features).unsqueeze(1)
            
            # Cross-attention between visual and each modality
            cross_attended = []
            
            for modality_features, cross_attention in zip(
                [attribute_proj, texture_proj, style_proj],
                self.cross_attention_layers
            ):
                attended, _ = cross_attention(visual_proj, modality_features, modality_features)
                cross_attended.append(attended.squeeze(1))
            
            # Average the cross-attended features
            return torch.stack(cross_attended).mean(dim=0)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


class FashionEmbeddingModel(nn.Module):
    """
    Complete fashion embedding model that combines all embedding components.
    
    Integrates visual features, attribute embeddings, texture features,
    and style features into a unified representation.
    """
    
    def __init__(self, config: FashionEmbeddingConfig):
        super().__init__()
        self.config = config
        
        # Embedding components
        self.attribute_embedding = FashionAttributeEmbedding(config)
        
        if config.use_texture_features:
            self.texture_extractor = TextureFeatureExtractor(config)
        
        if config.use_style_features:
            self.style_extractor = StyleFeatureExtractor(config)
        
        # Multi-modal fusion
        self.fusion = MultiModalFusion(config)
        
        # Final projection layer
        self.final_projection = nn.Sequential(
            nn.Linear(config.base_embedding_dim, config.base_embedding_dim),
            nn.LayerNorm(config.base_embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.base_embedding_dim, config.base_embedding_dim)
        )
    
    def forward(self, 
                visual_features: torch.Tensor,
                images: Optional[torch.Tensor] = None,
                attributes: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass for fashion embedding model.
        
        Args:
            visual_features: Pre-computed visual features
            images: Raw images (optional, for texture/style extraction)
            attributes: Fashion attributes (optional)
            
        Returns:
            Comprehensive fashion embeddings
        """
        # Get attribute embeddings
        if attributes is not None:
            attribute_features = self.attribute_embedding(attributes)
        else:
            attribute_features = torch.zeros_like(visual_features)
        
        # Get texture features
        if self.config.use_texture_features and images is not None:
            texture_features = self.texture_extractor(images)
        else:
            texture_features = torch.zeros(visual_features.shape[0], 
                                         self.config.texture_embedding_dim,
                                         device=visual_features.device)
        
        # Get style features
        if self.config.use_style_features and images is not None:
            style_features = self.style_extractor(images)
        else:
            style_features = torch.zeros(visual_features.shape[0], 
                                       self.config.style_embedding_dim,
                                       device=visual_features.device)
        
        # Fuse all features
        fused_features = self.fusion(visual_features, attribute_features, 
                                   texture_features, style_features)
        
        # Apply final projection
        embeddings = self.final_projection(fused_features)
        
        return F.normalize(embeddings, p=2, dim=1)


# Utility functions
def create_fashion_embedding_model(config: FashionEmbeddingConfig) -> FashionEmbeddingModel:
    """Create a fashion embedding model with the specified configuration."""
    return FashionEmbeddingModel(config)


def extract_fashion_attributes(text_description: str, config: FashionEmbeddingConfig) -> Dict[str, int]:
    """
    Extract fashion attributes from text description.
    
    Args:
        text_description: Text description of fashion item
        config: Embedding configuration
        
    Returns:
        Dictionary of attribute indices
    """
    text_lower = text_description.lower()
    attributes = {}
    
    # Extract category
    for i, category in enumerate(config.categories):
        if category in text_lower:
            attributes['category'] = i
            break
    
    # Extract color
    for i, color in enumerate(config.colors):
        if color in text_lower:
            attributes['color'] = i
            break
    
    # Extract pattern
    for i, pattern in enumerate(config.patterns):
        if pattern in text_lower:
            attributes['pattern'] = i
            break
    
    # Extract material
    for i, material in enumerate(config.materials):
        if material in text_lower:
            attributes['material'] = i
            break
    
    # Extract style
    for i, style in enumerate(config.styles):
        if style in text_lower:
            attributes['style'] = i
            break
    
    # Extract season
    for i, season in enumerate(config.seasons):
        if season in text_lower:
            attributes['season'] = i
            break
    
    return attributes


if __name__ == "__main__":
    # Example usage
    config = FashionEmbeddingConfig(
        base_embedding_dim=512,
        use_texture_features=True,
        use_style_features=True,
        fusion_method="attention"
    )
    
    # Create model
    model = FashionEmbeddingModel(config)
    
    # Example forward pass
    batch_size = 4
    visual_features = torch.randn(batch_size, config.base_embedding_dim)
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Create sample attributes
    attributes = {
        'category': torch.randint(0, len(config.categories), (batch_size,)),
        'color': torch.randint(0, len(config.colors), (batch_size,)),
        'pattern': torch.randint(0, len(config.patterns), (batch_size,)),
        'material': torch.randint(0, len(config.materials), (batch_size,)),
        'style': torch.randint(0, len(config.styles), (batch_size,)),
        'season': torch.randint(0, len(config.seasons), (batch_size,))
    }
    
    # Forward pass
    embeddings = model(visual_features, images, attributes)
    
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Embedding norm: {torch.norm(embeddings, dim=1).mean():.4f}")
    
    # Test attribute extraction
    description = "red floral dress made of silk for summer"
    extracted_attrs = extract_fashion_attributes(description, config)
    print(f"Extracted attributes: {extracted_attrs}")