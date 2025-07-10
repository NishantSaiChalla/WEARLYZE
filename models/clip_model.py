"""
CLIP-based Fashion Detection Model.

This module provides a comprehensive implementation of CLIP (Contrastive Language-Image Pre-training)
adapted for fashion detection tasks. It includes support for multiple CLIP architectures,
fashion-specific fine-tuning, and contrastive learning with hard negative mining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import warnings

try:
    import clip
    from transformers import CLIPModel, CLIPProcessor, CLIPConfig
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    warnings.warn("CLIP or transformers not installed. Please install with: pip install openai-clip transformers")
    clip = None
    CLIPModel = None
    CLIPProcessor = None
    CLIPConfig = None

logger = logging.getLogger(__name__)


@dataclass
class FashionCLIPConfig:
    """Configuration for Fashion CLIP model."""
    
    # Model architecture
    model_name: str = "ViT-B/32"  # ViT-B/32, ViT-L/14, or custom HuggingFace model
    embedding_dim: int = 512
    use_custom_projection: bool = True
    projection_dim: int = 512
    
    # Training parameters
    learning_rate: float = 1e-4
    temperature: float = 0.07
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    
    # Fashion-specific parameters
    fashion_classes: List[str] = None
    attribute_classes: List[str] = None
    use_attribute_loss: bool = True
    attribute_loss_weight: float = 0.3
    
    # Contrastive learning
    use_hard_negatives: bool = True
    negative_sampling_ratio: float = 0.1
    margin: float = 0.2
    
    # Fine-tuning
    freeze_backbone: bool = False
    fine_tune_layers: List[str] = None
    
    def __post_init__(self):
        if self.fashion_classes is None:
            self.fashion_classes = [
                "dress", "shirt", "pants", "skirt", "jacket", "coat", "sweater",
                "blouse", "t-shirt", "jeans", "shorts", "suit", "hoodie", "cardigan"
            ]
        
        if self.attribute_classes is None:
            self.attribute_classes = [
                "casual", "formal", "sporty", "vintage", "modern", "elegant",
                "comfortable", "fitted", "loose", "long-sleeve", "short-sleeve",
                "striped", "solid", "patterned", "denim", "cotton", "silk"
            ]


class FashionTextEncoder(nn.Module):
    """Fashion-specific text encoder with enhanced fashion vocabulary."""
    
    def __init__(self, config: FashionCLIPConfig):
        super().__init__()
        self.config = config
        
        # Load base CLIP text encoder
        if clip is not None:
            self.clip_model, _ = clip.load(config.model_name, device="cpu")
            self.text_encoder = self.clip_model.transformer
            self.token_embedding = self.clip_model.token_embedding
            self.positional_embedding = self.clip_model.positional_embedding
            self.ln_final = self.clip_model.ln_final
            self.text_projection = self.clip_model.text_projection
        else:
            # Fallback to transformers implementation
            self.clip_model = CLIPModel.from_pretrained(f"openai/clip-{config.model_name.lower()}")
            self.text_encoder = self.clip_model.text_model
            self.text_projection = self.clip_model.text_projection
        
        # Fashion-specific vocabulary enhancement
        self.fashion_vocab_size = len(config.fashion_classes) + len(config.attribute_classes)
        self.fashion_embeddings = nn.Embedding(self.fashion_vocab_size, config.embedding_dim)
        
        # Custom projection layer
        if config.use_custom_projection:
            self.custom_projection = nn.Sequential(
                nn.Linear(config.embedding_dim, config.projection_dim),
                nn.LayerNorm(config.projection_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.projection_dim, config.projection_dim)
            )
        else:
            self.custom_projection = None
    
    def forward(self, text_tokens: torch.Tensor, fashion_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for text encoding.
        
        Args:
            text_tokens: Tokenized text input
            fashion_labels: Optional fashion category labels
            
        Returns:
            Text embeddings
        """
        if clip is not None:
            # Original CLIP implementation
            x = self.token_embedding(text_tokens)
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.text_encoder(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.text_projection
        else:
            # Transformers implementation
            text_outputs = self.text_encoder(text_tokens)
            x = text_outputs.last_hidden_state
            x = self.text_projection(x)
        
        # Enhance with fashion-specific embeddings
        if fashion_labels is not None:
            fashion_emb = self.fashion_embeddings(fashion_labels)
            x = x + fashion_emb
        
        # Apply custom projection
        if self.custom_projection is not None:
            x = self.custom_projection(x)
        
        return F.normalize(x, dim=-1)


class FashionImageEncoder(nn.Module):
    """Fashion-specific image encoder with enhanced visual features."""
    
    def __init__(self, config: FashionCLIPConfig):
        super().__init__()
        self.config = config
        
        # Load base CLIP image encoder
        if clip is not None:
            self.clip_model, _ = clip.load(config.model_name, device="cpu")
            self.image_encoder = self.clip_model.visual
            self.image_projection = self.clip_model.visual.proj
        else:
            # Fallback to transformers implementation
            self.clip_model = CLIPModel.from_pretrained(f"openai/clip-{config.model_name.lower()}")
            self.image_encoder = self.clip_model.vision_model
            self.image_projection = self.clip_model.visual_projection
        
        # Fashion-specific visual features
        self.fashion_visual_layers = nn.Sequential(
            nn.Conv2d(config.embedding_dim, config.embedding_dim, 3, padding=1),
            nn.BatchNorm2d(config.embedding_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(config.embedding_dim, config.embedding_dim)
        )
        
        # Custom projection layer
        if config.use_custom_projection:
            self.custom_projection = nn.Sequential(
                nn.Linear(config.embedding_dim, config.projection_dim),
                nn.LayerNorm(config.projection_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.projection_dim, config.projection_dim)
            )
        else:
            self.custom_projection = None
    
    def forward(self, images: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for image encoding.
        
        Args:
            images: Input images
            return_features: Whether to return intermediate features
            
        Returns:
            Image embeddings and optionally intermediate features
        """
        if clip is not None:
            # Original CLIP implementation
            x = self.image_encoder(images)
            if hasattr(self.image_encoder, 'proj') and self.image_encoder.proj is not None:
                x = x @ self.image_encoder.proj
        else:
            # Transformers implementation
            image_outputs = self.image_encoder(images)
            x = image_outputs.last_hidden_state
            x = self.image_projection(x)
        
        # Store intermediate features
        features = x.clone() if return_features else None
        
        # Apply custom projection
        if self.custom_projection is not None:
            x = self.custom_projection(x)
        
        x = F.normalize(x, dim=-1)
        
        if return_features:
            return x, features
        return x


class FashionCLIP(nn.Module):
    """
    Fashion-adapted CLIP model with contrastive learning and hard negative mining.
    
    This model extends CLIP for fashion-specific tasks with:
    - Fashion vocabulary enhancement
    - Attribute-aware embeddings
    - Hard negative mining
    - Fashion-specific fine-tuning
    """
    
    def __init__(self, config: FashionCLIPConfig):
        super().__init__()
        self.config = config
        
        # Initialize encoders
        self.text_encoder = FashionTextEncoder(config)
        self.image_encoder = FashionImageEncoder(config)
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(config.temperature))
        
        # Fashion attribute classifier
        if config.use_attribute_loss:
            self.attribute_classifier = nn.Sequential(
                nn.Linear(config.projection_dim, config.projection_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.projection_dim // 2, len(config.attribute_classes))
            )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize custom layer weights."""
        for module in [self.text_encoder.custom_projection, self.image_encoder.custom_projection]:
            if module is not None:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
    
    def encode_text(self, text_tokens: torch.Tensor, fashion_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text inputs into embeddings."""
        return self.text_encoder(text_tokens, fashion_labels)
    
    def encode_image(self, images: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode image inputs into embeddings."""
        return self.image_encoder(images, return_features)
    
    def forward(self, 
                images: torch.Tensor, 
                text_tokens: torch.Tensor,
                fashion_labels: Optional[torch.Tensor] = None,
                attribute_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            images: Input images
            text_tokens: Tokenized text descriptions
            fashion_labels: Fashion category labels
            attribute_labels: Fashion attribute labels
            
        Returns:
            Dictionary containing embeddings and losses
        """
        # Encode inputs
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(text_tokens, fashion_labels)
        
        # Compute contrastive loss
        contrastive_loss = self._compute_contrastive_loss(image_embeddings, text_embeddings)
        
        outputs = {
            'image_embeddings': image_embeddings,
            'text_embeddings': text_embeddings,
            'contrastive_loss': contrastive_loss,
            'temperature': self.temperature.item()
        }
        
        # Compute attribute loss if enabled
        if self.config.use_attribute_loss and attribute_labels is not None:
            attribute_logits = self.attribute_classifier(image_embeddings)
            attribute_loss = F.cross_entropy(attribute_logits, attribute_labels)
            outputs['attribute_loss'] = attribute_loss
            outputs['attribute_logits'] = attribute_logits
        
        return outputs
    
    def _compute_contrastive_loss(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between image and text embeddings."""
        # Compute similarities
        logits = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature
        
        # Create labels (diagonal should be positive pairs)
        batch_size = image_embeddings.shape[0]
        labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # Compute cross-entropy loss for both directions
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def compute_similarity(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity scores between image and text embeddings."""
        return torch.matmul(image_embeddings, text_embeddings.T)
    
    def get_text_features(self, text_descriptions: List[str], device: str = "cuda") -> torch.Tensor:
        """
        Extract text features from descriptions.
        
        Args:
            text_descriptions: List of text descriptions
            device: Device to run inference on
            
        Returns:
            Text embeddings
        """
        if clip is not None:
            text_tokens = clip.tokenize(text_descriptions).to(device)
        else:
            # Use transformers tokenizer
            processor = CLIPProcessor.from_pretrained(f"openai/clip-{self.config.model_name.lower()}")
            text_tokens = processor(text=text_descriptions, return_tensors="pt", padding=True, truncation=True)
            text_tokens = text_tokens.input_ids.to(device)
        
        with torch.no_grad():
            text_features = self.encode_text(text_tokens)
        
        return text_features
    
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image features.
        
        Args:
            images: Input images
            
        Returns:
            Image embeddings
        """
        with torch.no_grad():
            image_features = self.encode_image(images)
        
        return image_features
    
    def save_model(self, path: Union[str, Path]):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'temperature': self.temperature.item()
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: Union[str, Path], device: str = "cuda") -> 'FashionCLIP':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.temperature.data = torch.tensor(checkpoint['temperature'])
        
        model.to(device)
        logger.info(f"Model loaded from {path}")
        
        return model


class FashionCLIPTrainer:
    """Trainer for Fashion CLIP model with hard negative mining."""
    
    def __init__(self, 
                 model: FashionCLIP,
                 config: FashionCLIPConfig,
                 device: str = "cuda"):
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10000
        )
        
        # Hard negative mining
        self.hard_negative_miner = HardNegativeMiner(
            negative_ratio=config.negative_sampling_ratio,
            margin=config.margin
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Compute total loss
        total_loss = outputs['contrastive_loss']
        
        if 'attribute_loss' in outputs:
            total_loss += self.config.attribute_loss_weight * outputs['attribute_loss']
        
        # Apply hard negative mining
        if self.config.use_hard_negatives:
            hard_negative_loss = self.hard_negative_miner(
                outputs['image_embeddings'],
                outputs['text_embeddings']
            )
            total_loss += 0.1 * hard_negative_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Return loss values
        loss_dict = {
            'total_loss': total_loss.item(),
            'contrastive_loss': outputs['contrastive_loss'].item(),
            'temperature': outputs['temperature']
        }
        
        if 'attribute_loss' in outputs:
            loss_dict['attribute_loss'] = outputs['attribute_loss'].item()
        
        return loss_dict
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        total_contrastive_loss = 0
        total_attribute_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = self.model(**batch)
                
                total_loss += outputs['contrastive_loss'].item()
                total_contrastive_loss += outputs['contrastive_loss'].item()
                
                if 'attribute_loss' in outputs:
                    total_attribute_loss += outputs['attribute_loss'].item()
                
                num_batches += 1
        
        results = {
            'val_loss': total_loss / num_batches,
            'val_contrastive_loss': total_contrastive_loss / num_batches,
        }
        
        if total_attribute_loss > 0:
            results['val_attribute_loss'] = total_attribute_loss / num_batches
        
        return results


class HardNegativeMiner:
    """Hard negative mining for contrastive learning."""
    
    def __init__(self, negative_ratio: float = 0.1, margin: float = 0.2):
        self.negative_ratio = negative_ratio
        self.margin = margin
    
    def __call__(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Mine hard negatives and compute triplet loss.
        
        Args:
            image_embeddings: Image embeddings
            text_embeddings: Text embeddings
            
        Returns:
            Hard negative loss
        """
        batch_size = image_embeddings.shape[0]
        
        # Compute similarity matrix
        similarities = torch.matmul(image_embeddings, text_embeddings.T)
        
        # Create positive and negative masks
        positive_mask = torch.eye(batch_size, device=similarities.device).bool()
        negative_mask = ~positive_mask
        
        # Get positive similarities
        positive_similarities = similarities[positive_mask]
        
        # Get hard negatives (highest similarity negative pairs)
        negative_similarities = similarities[negative_mask]
        num_negatives = int(self.negative_ratio * negative_similarities.numel())
        
        if num_negatives > 0:
            hard_negatives, _ = torch.topk(negative_similarities, num_negatives)
            
            # Compute triplet loss
            positive_expanded = positive_similarities.unsqueeze(1).expand(-1, num_negatives)
            hard_negatives_expanded = hard_negatives.unsqueeze(0).expand(batch_size, -1)
            
            loss = F.relu(hard_negatives_expanded - positive_expanded + self.margin)
            return loss.mean()
        
        return torch.tensor(0.0, device=similarities.device)


# Utility functions
def create_fashion_clip_model(model_name: str = "ViT-B/32", 
                            embedding_dim: int = 512,
                            device: str = "cuda") -> FashionCLIP:
    """Create a Fashion CLIP model with default configuration."""
    config = FashionCLIPConfig(
        model_name=model_name,
        embedding_dim=embedding_dim,
        projection_dim=embedding_dim
    )
    
    model = FashionCLIP(config)
    model.to(device)
    
    return model


def load_fashion_clip_model(checkpoint_path: Union[str, Path], 
                          device: str = "cuda") -> FashionCLIP:
    """Load a pre-trained Fashion CLIP model."""
    return FashionCLIP.load_model(checkpoint_path, device)


if __name__ == "__main__":
    # Example usage
    config = FashionCLIPConfig(
        model_name="ViT-B/32",
        embedding_dim=512,
        use_attribute_loss=True
    )
    
    model = FashionCLIP(config)
    
    # Example forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    text_tokens = torch.randint(0, 1000, (batch_size, 77))
    attribute_labels = torch.randint(0, len(config.attribute_classes), (batch_size,))
    
    outputs = model(images, text_tokens, attribute_labels=attribute_labels)
    
    print(f"Image embeddings shape: {outputs['image_embeddings'].shape}")
    print(f"Text embeddings shape: {outputs['text_embeddings'].shape}")
    print(f"Contrastive loss: {outputs['contrastive_loss'].item():.4f}")
    print(f"Attribute loss: {outputs['attribute_loss'].item():.4f}")