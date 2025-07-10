"""
Example Usage of Fashion Classification Models

This script demonstrates how to use the various fashion classification models,
ensemble methods, loss functions, and quantization utilities provided in this module.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from typing import Dict, List, Any
import logging

# Import fashion classification components
from .classifiers import (
    FashionResNet,
    FashionMobileNet,
    FashionConvNeXt,
    FashionViT,
    create_fashion_classifier,
    load_fashion_classifier
)

from .ensemble import (
    EnsembleClassifier,
    create_ensemble,
    evaluate_ensemble_diversity
)

from .losses import (
    FocalLoss,
    LabelSmoothingLoss,
    FashionTripletLoss,
    create_fashion_loss
)

from .model_factory import (
    ModelFactory,
    OptimizationFactory,
    create_complete_training_setup
)

from .quantization import (
    QuantizationConfig,
    PostTrainingQuantizer,
    ModelCompressor,
    save_compressed_model
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(batch_size: int = 32, num_classes: int = 50) -> tuple:
    """Create sample data for demonstration purposes."""
    # Create sample image data (3, 224, 224)
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Create sample labels
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Create dataset and dataloader
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    return images, labels, dataloader


def example_single_model_usage():
    """Demonstrate usage of individual fashion classification models."""
    logger.info("=== Single Model Usage Example ===")
    
    # Create sample data
    images, labels, dataloader = create_sample_data()
    
    # Example 1: Create ResNet model
    logger.info("Creating ResNet-50 model...")
    resnet_model = FashionResNet(
        num_classes=50,
        pretrained=True,
        dropout_rate=0.1,
        variant='resnet50'
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = resnet_model(images)
        logger.info(f"ResNet output shape: {outputs.shape}")
    
    # Example 2: Create MobileNet model
    logger.info("Creating MobileNet-V3 model...")
    mobilenet_model = FashionMobileNet(
        num_classes=50,
        pretrained=True,
        variant='mobilenetv3_large_100'
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = mobilenet_model(images)
        logger.info(f"MobileNet output shape: {outputs.shape}")
    
    # Example 3: Create ConvNeXt model
    logger.info("Creating ConvNeXt-Tiny model...")
    convnext_model = FashionConvNeXt(
        num_classes=50,
        pretrained=True,
        variant='convnext_tiny'
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = convnext_model(images)
        logger.info(f"ConvNeXt output shape: {outputs.shape}")
    
    # Example 4: Create Vision Transformer model
    logger.info("Creating Vision Transformer model...")
    vit_model = FashionViT(
        num_classes=50,
        pretrained=True,
        variant='vit_base_patch16_224'
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = vit_model(images)
        logger.info(f"ViT output shape: {outputs.shape}")
    
    # Get model information
    logger.info("Model Information:")
    for name, model in [('ResNet', resnet_model), ('MobileNet', mobilenet_model), 
                        ('ConvNeXt', convnext_model), ('ViT', vit_model)]:
        info = model.get_model_info()
        logger.info(f"{name}: {info['total_parameters']:,} parameters")


def example_ensemble_usage():
    """Demonstrate usage of ensemble methods."""
    logger.info("=== Ensemble Usage Example ===")
    
    # Create sample data
    images, labels, dataloader = create_sample_data()
    
    # Create individual models
    models = [
        FashionResNet(num_classes=50, pretrained=False),
        FashionMobileNet(num_classes=50, pretrained=False),
        FashionConvNeXt(num_classes=50, pretrained=False)
    ]
    
    # Example 1: Soft Voting Ensemble
    logger.info("Creating Soft Voting Ensemble...")
    soft_ensemble = create_ensemble(
        models,
        method='soft_voting',
        weights=[0.4, 0.3, 0.3],
        temperature=1.0
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = soft_ensemble(images)
        logger.info(f"Soft Voting Ensemble output shape: {outputs.shape}")
    
    # Example 2: Hard Voting Ensemble
    logger.info("Creating Hard Voting Ensemble...")
    hard_ensemble = create_ensemble(
        models,
        method='hard_voting',
        weights=[1.0, 1.0, 1.0]
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = hard_ensemble(images)
        logger.info(f"Hard Voting Ensemble output shape: {outputs.shape}")
    
    # Example 3: Weighted Ensemble
    logger.info("Creating Weighted Ensemble...")
    weighted_ensemble = create_ensemble(
        models,
        method='weighted',
        learning_rate=0.001
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = weighted_ensemble(images)
        logger.info(f"Weighted Ensemble output shape: {outputs.shape}")
    
    # Example 4: Stacked Ensemble
    logger.info("Creating Stacked Ensemble...")
    stacked_ensemble = create_ensemble(
        models,
        method='stacked',
        use_original_features=True
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = stacked_ensemble(images)
        logger.info(f"Stacked Ensemble output shape: {outputs.shape}")
    
    # Get ensemble information
    ensemble_info = soft_ensemble.get_ensemble_info()
    logger.info(f"Ensemble has {ensemble_info['num_models']} models")


def example_loss_functions():
    """Demonstrate usage of custom loss functions."""
    logger.info("=== Loss Functions Example ===")
    
    # Create sample predictions and targets
    batch_size = 32
    num_classes = 50
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Example 1: Focal Loss
    logger.info("Using Focal Loss...")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss_value = focal_loss(predictions, targets)
    logger.info(f"Focal Loss value: {loss_value.item():.4f}")
    
    # Example 2: Label Smoothing Loss
    logger.info("Using Label Smoothing Loss...")
    label_smoothing_loss = LabelSmoothingLoss(
        num_classes=num_classes,
        smoothing=0.1
    )
    loss_value = label_smoothing_loss(predictions, targets)
    logger.info(f"Label Smoothing Loss value: {loss_value.item():.4f}")
    
    # Example 3: Triplet Loss
    logger.info("Using Triplet Loss...")
    embeddings = torch.randn(batch_size, 512)  # Feature embeddings
    triplet_loss = FashionTripletLoss(margin=1.0, mining_strategy='hard')
    loss_value = triplet_loss(embeddings, targets)
    logger.info(f"Triplet Loss value: {loss_value.item():.4f}")
    
    # Example 4: Create loss using factory
    logger.info("Creating loss using factory...")
    factory_loss = create_fashion_loss(
        loss_type='focal',
        num_classes=num_classes,
        alpha=0.25,
        gamma=2.0
    )
    loss_value = factory_loss(predictions, targets)
    logger.info(f"Factory-created Focal Loss value: {loss_value.item():.4f}")


def example_model_factory():
    """Demonstrate usage of model factory."""
    logger.info("=== Model Factory Example ===")
    
    # Create sample data
    images, labels, dataloader = create_sample_data()
    
    # Example 1: Create model using factory
    logger.info("Creating model using factory...")
    model = ModelFactory.create_model(
        model_type='resnet50',
        config={
            'num_classes': 50,
            'pretrained': True,
            'dropout_rate': 0.1,
            'use_auxiliary_head': False
        }
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images)
        logger.info(f"Factory-created model output shape: {outputs.shape}")
    
    # Example 2: Create ensemble using factory
    logger.info("Creating ensemble using factory...")
    model_configs = [
        {'model_type': 'resnet', 'num_classes': 50, 'pretrained': False},
        {'model_type': 'mobilenet', 'num_classes': 50, 'pretrained': False},
        {'model_type': 'convnext', 'num_classes': 50, 'pretrained': False}
    ]
    
    ensemble = ModelFactory.create_ensemble(
        model_configs,
        ensemble_method='soft_voting',
        weights=[0.4, 0.3, 0.3]
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = ensemble(images)
        logger.info(f"Factory-created ensemble output shape: {outputs.shape}")
    
    # Example 3: Create complete training setup
    logger.info("Creating complete training setup...")
    model_config = {
        'type': 'resnet',
        'num_classes': 50,
        'pretrained': True,
        'dropout_rate': 0.1
    }
    
    training_config = {
        'optimizer': {
            'type': 'adamw',
            'learning_rate': 0.001,
            'weight_decay': 0.0001
        },
        'scheduler': {
            'type': 'cosine',
            'params': {'T_max': 100}
        },
        'loss': {
            'type': 'focal',
            'params': {'alpha': 0.25, 'gamma': 2.0}
        }
    }
    
    training_setup = create_complete_training_setup(
        model_config,
        training_config,
        device='cpu'
    )
    
    logger.info(f"Training setup created with {len(training_setup)} components")
    
    # List available models
    available_models = ModelFactory.list_available_models()
    logger.info(f"Available models: {available_models[:5]}...")  # Show first 5


def example_quantization():
    """Demonstrate usage of quantization utilities."""
    logger.info("=== Quantization Example ===")
    
    # Create sample data
    images, labels, dataloader = create_sample_data()
    
    # Create a simple model for quantization
    model = FashionResNet(num_classes=50, pretrained=False)
    model.eval()
    
    # Example 1: Dynamic Quantization
    logger.info("Performing dynamic quantization...")
    quantizer = PostTrainingQuantizer(
        QuantizationConfig(quantization_type='dynamic')
    )
    
    quantized_model = quantizer.quantize_dynamic(model)
    
    # Test quantized model
    with torch.no_grad():
        outputs = quantized_model(images)
        logger.info(f"Quantized model output shape: {outputs.shape}")
    
    # Example 2: Model Compression
    logger.info("Performing model compression...")
    compressor = ModelCompressor()
    
    compression_config = {
        'pruning': {
            'enabled': True,
            'ratio': 0.3
        },
        'quantization': {
            'enabled': True,
            'type': 'dynamic'
        }
    }
    
    compressed_models = compressor.compress_model(
        model,
        compression_config
    )
    
    # Test compressed models
    for name, model_info in compressed_models.items():
        compressed_model = model_info['model']
        compression_ratio = model_info['compression_ratio']
        
        with torch.no_grad():
            outputs = compressed_model(images)
            logger.info(f"{name} - Compression ratio: {compression_ratio:.2f}x, "
                       f"Output shape: {outputs.shape}")
    
    # Example 3: Save compressed model
    if 'quantized' in compressed_models:
        save_compressed_model(
            compressed_models['quantized']['model'],
            'quantized_fashion_model.pth',
            metadata={'compression_ratio': compressed_models['quantized']['compression_ratio']}
        )
        logger.info("Compressed model saved successfully")


def example_training_loop():
    """Demonstrate a simple training loop with custom components."""
    logger.info("=== Training Loop Example ===")
    
    # Create sample data
    images, labels, dataloader = create_sample_data()
    
    # Create model
    model = FashionResNet(num_classes=50, pretrained=False)
    
    # Create optimizer
    optimizer = OptimizationFactory.create_optimizer(
        model,
        optimizer_type='adamw',
        learning_rate=0.001,
        weight_decay=0.0001
    )
    
    # Create scheduler
    scheduler = OptimizationFactory.create_scheduler(
        optimizer,
        scheduler_type='cosine',
        T_max=10
    )
    
    # Create loss function
    criterion = create_fashion_loss(
        loss_type='focal',
        num_classes=50,
        alpha=0.25,
        gamma=2.0
    )
    
    # Training loop
    model.train()
    num_epochs = 3
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # Forward pass
            output = model(data)
            if isinstance(output, tuple):  # Handle auxiliary outputs
                output = output[0]
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx == 0:  # Only log first batch
                logger.info(f"Epoch {epoch+1}/{num_epochs}, "
                           f"Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        
        # Update scheduler
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed, "
                   f"Average Loss: {avg_loss:.4f}")
    
    logger.info("Training completed successfully!")


def main():
    """Run all examples."""
    logger.info("Starting Fashion Classification Models Examples")
    
    try:
        example_single_model_usage()
        example_ensemble_usage()
        example_loss_functions()
        example_model_factory()
        example_quantization()
        example_training_loop()
        
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()