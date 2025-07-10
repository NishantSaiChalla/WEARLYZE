"""
Model Factory for Fashion Detection System

This module provides factory functions and utilities for creating, configuring,
and managing different types of fashion classification models with various
backbones and configurations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import yaml
import logging
from pathlib import Path

from .classifiers import (
    BaseFashionClassifier,
    FashionResNet,
    FashionMobileNet,
    FashionConvNeXt,
    FashionViT,
    FashionEfficientNet,
    FashionMultiScale,
    create_fashion_classifier,
    load_fashion_classifier
)
from .ensemble import (
    EnsembleClassifier,
    create_ensemble
)
from .losses import create_fashion_loss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration class for model creation."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize model configuration.
        
        Args:
            config_dict: Dictionary containing model configuration
        """
        self.config = config_dict
        self.model_type = config_dict.get('model_type', 'resnet')
        self.num_classes = config_dict.get('num_classes', 50)
        self.pretrained = config_dict.get('pretrained', True)
        self.dropout_rate = config_dict.get('dropout_rate', 0.1)
        self.use_auxiliary_head = config_dict.get('use_auxiliary_head', False)
        
        # Model-specific parameters
        self.backbone_params = config_dict.get('backbone_params', {})
        self.training_params = config_dict.get('training_params', {})
        self.optimization_params = config_dict.get('optimization_params', {})
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get parameters for model creation."""
        return {
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'dropout_rate': self.dropout_rate,
            'use_auxiliary_head': self.use_auxiliary_head,
            **self.backbone_params
        }
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get training parameters."""
        return self.training_params
    
    def get_optimization_params(self) -> Dict[str, Any]:
        """Get optimization parameters."""
        return self.optimization_params


class ModelFactory:
    """Factory class for creating fashion classification models."""
    
    # Model registry mapping model names to classes
    MODEL_REGISTRY = {
        'resnet': FashionResNet,
        'resnet50': FashionResNet,
        'resnet101': FashionResNet,
        'resnet152': FashionResNet,
        'mobilenet': FashionMobileNet,
        'mobilenet_v3_large': FashionMobileNet,
        'mobilenet_v3_small': FashionMobileNet,
        'convnext': FashionConvNeXt,
        'convnext_tiny': FashionConvNeXt,
        'convnext_small': FashionConvNeXt,
        'convnext_base': FashionConvNeXt,
        'vit': FashionViT,
        'vit_base': FashionViT,
        'vit_small': FashionViT,
        'vit_tiny': FashionViT,
        'efficientnet': FashionEfficientNet,
        'efficientnet_b0': FashionEfficientNet,
        'efficientnet_b1': FashionEfficientNet,
        'efficientnet_b2': FashionEfficientNet,
        'efficientnet_b3': FashionEfficientNet,
        'efficientnet_b4': FashionEfficientNet,
        'multiscale': FashionMultiScale
    }
    
    # Default configurations for each model type
    DEFAULT_CONFIGS = {
        'resnet': {
            'variant': 'resnet50',
            'dropout_rate': 0.1,
            'use_auxiliary_head': False
        },
        'mobilenet': {
            'variant': 'mobilenetv3_large_100',
            'dropout_rate': 0.2,
            'use_auxiliary_head': False
        },
        'convnext': {
            'variant': 'convnext_tiny',
            'dropout_rate': 0.1,
            'use_auxiliary_head': False
        },
        'vit': {
            'variant': 'vit_base_patch16_224',
            'dropout_rate': 0.1,
            'use_auxiliary_head': False,
            'patch_size': 16,
            'embed_dim': 768
        },
        'efficientnet': {
            'variant': 'efficientnet_b0',
            'dropout_rate': 0.2,
            'use_auxiliary_head': False
        },
        'multiscale': {
            'backbone_type': 'resnet50',
            'scales': [224, 288, 384],
            'dropout_rate': 0.1,
            'use_auxiliary_head': False
        }
    }
    
    @classmethod
    def create_model(
        cls,
        model_type: str,
        config: Optional[Union[Dict[str, Any], ModelConfig]] = None,
        **kwargs
    ) -> BaseFashionClassifier:
        """
        Create a fashion classification model.
        
        Args:
            model_type: Type of model to create
            config: Model configuration (dict or ModelConfig object)
            **kwargs: Additional arguments for model creation
        
        Returns:
            Fashion classification model instance
        
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {list(cls.MODEL_REGISTRY.keys())}")
        
        model_class = cls.MODEL_REGISTRY[model_type]
        
        # Get default configuration
        base_model_type = cls._get_base_model_type(model_type)
        default_config = cls.DEFAULT_CONFIGS.get(base_model_type, {})
        
        # Process configuration
        if config is None:
            config = {}
        elif isinstance(config, ModelConfig):
            config = config.get_model_params()
        
        # Merge configurations
        final_config = {**default_config, **config, **kwargs}
        
        # Handle specific model variants
        if model_type.startswith('resnet'):
            final_config['variant'] = model_type
        elif model_type.startswith('mobilenet'):
            if 'small' in model_type:
                final_config['variant'] = 'mobilenetv3_small_100'
            else:
                final_config['variant'] = 'mobilenetv3_large_100'
        elif model_type.startswith('convnext'):
            final_config['variant'] = model_type
        elif model_type.startswith('vit'):
            final_config['variant'] = f"{model_type}_patch16_224"
        elif model_type.startswith('efficientnet'):
            final_config['variant'] = model_type
        
        logger.info(f"Creating {model_type} model with config: {final_config}")
        
        return model_class(**final_config)
    
    @classmethod
    def create_ensemble(
        cls,
        model_configs: List[Dict[str, Any]],
        ensemble_method: str = 'soft_voting',
        **kwargs
    ) -> EnsembleClassifier:
        """
        Create an ensemble of fashion classification models.
        
        Args:
            model_configs: List of model configurations
            ensemble_method: Ensemble method to use
            **kwargs: Additional arguments for ensemble creation
        
        Returns:
            Ensemble classifier instance
        """
        models = []
        
        for config in model_configs:
            model_type = config.get('model_type', 'resnet')
            model = cls.create_model(model_type, config)
            models.append(model)
        
        logger.info(f"Creating {ensemble_method} ensemble with {len(models)} models")
        
        return create_ensemble(models, ensemble_method, **kwargs)
    
    @classmethod
    def load_model(
        cls,
        checkpoint_path: str,
        model_type: str,
        config: Optional[Union[Dict[str, Any], ModelConfig]] = None,
        device: str = 'cuda',
        strict: bool = True
    ) -> BaseFashionClassifier:
        """
        Load a fashion classification model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_type: Type of model to load
            config: Model configuration
            device: Device to load model on
            strict: Whether to strictly enforce that the keys in state_dict match
        
        Returns:
            Loaded fashion classification model
        """
        # Create model
        model = cls.create_model(model_type, config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        model.load_state_dict(state_dict, strict=strict)
        model.to(device)
        model.eval()
        
        logger.info(f"Loaded {model_type} model from {checkpoint_path}")
        
        return model
    
    @classmethod
    def create_from_config_file(
        cls,
        config_path: str,
        model_name: Optional[str] = None
    ) -> BaseFashionClassifier:
        """
        Create a model from a configuration file.
        
        Args:
            config_path: Path to configuration file (YAML)
            model_name: Name of specific model in config (if multiple models)
        
        Returns:
            Fashion classification model instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if model_name:
            if 'models' not in config or model_name not in config['models']:
                raise ValueError(f"Model '{model_name}' not found in configuration")
            model_config = config['models'][model_name]
        else:
            if 'model' in config:
                model_config = config['model']
            else:
                raise ValueError("No model configuration found in file")
        
        model_type = model_config.get('type', 'resnet')
        
        return cls.create_model(model_type, model_config)
    
    @classmethod
    def create_multi_task_model(
        cls,
        base_model_type: str,
        tasks: List[str],
        task_configs: Dict[str, Dict[str, Any]],
        shared_backbone: bool = True
    ) -> nn.Module:
        """
        Create a multi-task fashion classification model.
        
        Args:
            base_model_type: Base model type for backbone
            tasks: List of task names
            task_configs: Configuration for each task
            shared_backbone: Whether to share backbone across tasks
        
        Returns:
            Multi-task model instance
        """
        class MultiTaskFashionModel(nn.Module):
            def __init__(self, base_model, tasks, task_configs):
                super().__init__()
                self.base_model = base_model
                self.tasks = tasks
                self.task_heads = nn.ModuleDict()
                
                # Create task-specific heads
                for task in tasks:
                    task_config = task_configs.get(task, {})
                    num_classes = task_config.get('num_classes', 50)
                    
                    self.task_heads[task] = nn.Sequential(
                        nn.Linear(base_model.feature_dim, base_model.feature_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(base_model.feature_dim // 2, num_classes)
                    )
            
            def forward(self, x, task=None):
                features = self.base_model.extract_features(x)
                pooled_features = nn.AdaptiveAvgPool2d((1, 1))(features).flatten(1)
                
                if task:
                    return self.task_heads[task](pooled_features)
                else:
                    outputs = {}
                    for task_name in self.tasks:
                        outputs[task_name] = self.task_heads[task_name](pooled_features)
                    return outputs
        
        # Create base model
        base_model = cls.create_model(base_model_type, num_classes=0)  # No classifier
        
        # Create multi-task model
        multi_task_model = MultiTaskFashionModel(base_model, tasks, task_configs)
        
        logger.info(f"Created multi-task model with tasks: {tasks}")
        
        return multi_task_model
    
    @classmethod
    def _get_base_model_type(cls, model_type: str) -> str:
        """Get base model type from specific variant."""
        if model_type.startswith('resnet'):
            return 'resnet'
        elif model_type.startswith('mobilenet'):
            return 'mobilenet'
        elif model_type.startswith('convnext'):
            return 'convnext'
        elif model_type.startswith('vit'):
            return 'vit'
        elif model_type.startswith('efficientnet'):
            return 'efficientnet'
        else:
            return model_type
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """Get information about a specific model type."""
        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")
        
        base_type = cls._get_base_model_type(model_type)
        default_config = cls.DEFAULT_CONFIGS.get(base_type, {})
        
        return {
            'model_type': model_type,
            'base_type': base_type,
            'model_class': cls.MODEL_REGISTRY[model_type].__name__,
            'default_config': default_config,
            'supported_variants': cls._get_supported_variants(base_type)
        }
    
    @classmethod
    def _get_supported_variants(cls, base_type: str) -> List[str]:
        """Get supported variants for a base model type."""
        variant_map = {
            'resnet': ['resnet50', 'resnet101', 'resnet152'],
            'mobilenet': ['mobilenetv3_large_100', 'mobilenetv3_small_100'],
            'convnext': ['convnext_tiny', 'convnext_small', 'convnext_base'],
            'vit': ['vit_base_patch16_224', 'vit_small_patch16_224', 'vit_tiny_patch16_224'],
            'efficientnet': [f'efficientnet_b{i}' for i in range(8)]
        }
        
        return variant_map.get(base_type, [])
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """List all available model types."""
        return list(cls.MODEL_REGISTRY.keys())


class OptimizationFactory:
    """Factory class for creating optimizers and schedulers."""
    
    @staticmethod
    def create_optimizer(
        model: nn.Module,
        optimizer_type: str = 'adamw',
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        Create an optimizer for the model.
        
        Args:
            model: Model to optimize
            optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
            learning_rate: Learning rate
            weight_decay: Weight decay
            **kwargs: Additional optimizer parameters
        
        Returns:
            Optimizer instance
        """
        optimizer_registry = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop
        }
        
        if optimizer_type not in optimizer_registry:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        optimizer_class = optimizer_registry[optimizer_type]
        
        # Filter parameters based on optimizer type
        if optimizer_type in ['adam', 'adamw']:
            params = {
                'lr': learning_rate,
                'weight_decay': weight_decay,
                'betas': kwargs.get('betas', (0.9, 0.999)),
                'eps': kwargs.get('eps', 1e-8)
            }
        elif optimizer_type == 'sgd':
            params = {
                'lr': learning_rate,
                'weight_decay': weight_decay,
                'momentum': kwargs.get('momentum', 0.9),
                'nesterov': kwargs.get('nesterov', True)
            }
        elif optimizer_type == 'rmsprop':
            params = {
                'lr': learning_rate,
                'weight_decay': weight_decay,
                'momentum': kwargs.get('momentum', 0.0),
                'alpha': kwargs.get('alpha', 0.99)
            }
        
        return optimizer_class(model.parameters(), **params)
    
    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = 'cosine',
        **kwargs
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Create a learning rate scheduler.
        
        Args:
            optimizer: Optimizer instance
            scheduler_type: Type of scheduler ('cosine', 'step', 'exponential', 'plateau')
            **kwargs: Additional scheduler parameters
        
        Returns:
            Scheduler instance
        """
        scheduler_registry = {
            'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
            'step': torch.optim.lr_scheduler.StepLR,
            'exponential': torch.optim.lr_scheduler.ExponentialLR,
            'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'multistep': torch.optim.lr_scheduler.MultiStepLR
        }
        
        if scheduler_type not in scheduler_registry:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        scheduler_class = scheduler_registry[scheduler_type]
        
        # Filter parameters based on scheduler type
        if scheduler_type == 'cosine':
            params = {
                'T_max': kwargs.get('T_max', 100),
                'eta_min': kwargs.get('eta_min', 0.0)
            }
        elif scheduler_type == 'step':
            params = {
                'step_size': kwargs.get('step_size', 30),
                'gamma': kwargs.get('gamma', 0.1)
            }
        elif scheduler_type == 'exponential':
            params = {
                'gamma': kwargs.get('gamma', 0.95)
            }
        elif scheduler_type == 'plateau':
            params = {
                'mode': kwargs.get('mode', 'min'),
                'factor': kwargs.get('factor', 0.5),
                'patience': kwargs.get('patience', 10),
                'min_lr': kwargs.get('min_lr', 0.0)
            }
        elif scheduler_type == 'multistep':
            params = {
                'milestones': kwargs.get('milestones', [30, 60, 90]),
                'gamma': kwargs.get('gamma', 0.1)
            }
        
        return scheduler_class(optimizer, **params)


def create_complete_training_setup(
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Create a complete training setup with model, optimizer, scheduler, and loss function.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        device: Device to use
    
    Returns:
        Dictionary containing all training components
    """
    # Create model
    model_type = model_config.get('type', 'resnet')
    model = ModelFactory.create_model(model_type, model_config)
    model.to(device)
    
    # Create optimizer
    optimizer_config = training_config.get('optimizer', {})
    optimizer = OptimizationFactory.create_optimizer(
        model,
        optimizer_type=optimizer_config.get('type', 'adamw'),
        learning_rate=optimizer_config.get('learning_rate', 0.001),
        weight_decay=optimizer_config.get('weight_decay', 0.0001),
        **optimizer_config.get('params', {})
    )
    
    # Create scheduler
    scheduler_config = training_config.get('scheduler', {})
    scheduler = OptimizationFactory.create_scheduler(
        optimizer,
        scheduler_type=scheduler_config.get('type', 'cosine'),
        **scheduler_config.get('params', {})
    )
    
    # Create loss function
    loss_config = training_config.get('loss', {})
    loss_fn = create_fashion_loss(
        loss_type=loss_config.get('type', 'cross_entropy'),
        num_classes=model_config.get('num_classes', 50),
        **loss_config.get('params', {})
    )
    
    logger.info(f"Created complete training setup for {model_type} model")
    
    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_fn': loss_fn,
        'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {}
    }


# Export all classes and functions
__all__ = [
    'ModelConfig',
    'ModelFactory',
    'OptimizationFactory',
    'create_complete_training_setup'
]