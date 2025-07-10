# Fashion Classification Models

This directory contains comprehensive deep learning models for fashion classification tasks, supporting the 50 fashion categories from the DeepFashion dataset.

## Overview

The fashion classification system includes:

- **Individual Classification Models**: ResNet, MobileNet, ConvNeXt, Vision Transformer, EfficientNet, and Multi-Scale models
- **Ensemble Methods**: Soft/Hard voting, weighted ensembles, stacked ensembles, and dynamic ensembles
- **Custom Loss Functions**: Focal loss, label smoothing, triplet loss, contrastive loss, center loss, and ArcFace loss
- **Model Factory**: Easy creation and configuration of models with different architectures
- **Quantization & Compression**: Post-training quantization, quantization-aware training, and model pruning

## File Structure

```
models/
├── classifiers.py              # Individual classification models
├── ensemble.py                 # Ensemble methods
├── losses.py                   # Custom loss functions
├── model_factory.py            # Model creation utilities
├── quantization.py             # Quantization and compression
├── classification_example.py   # Usage examples
├── classification_requirements.txt  # Dependencies
└── CLASSIFICATION_README.md    # This file
```

## Supported Models

### 1. Individual Classifiers

#### FashionResNet
- **Architecture**: ResNet-50/101/152 with custom fashion classification head
- **Features**: Transfer learning, auxiliary heads, configurable dropout
- **Use case**: Balanced performance and accuracy

```python
from models.classifiers import FashionResNet

model = FashionResNet(
    num_classes=50,
    pretrained=True,
    dropout_rate=0.1,
    variant='resnet50'
)
```

#### FashionMobileNet
- **Architecture**: MobileNet-V3 Large/Small for efficient deployment
- **Features**: Optimized for mobile and edge devices
- **Use case**: Resource-constrained environments

```python
from models.classifiers import FashionMobileNet

model = FashionMobileNet(
    num_classes=50,
    pretrained=True,
    variant='mobilenetv3_large_100'
)
```

#### FashionConvNeXt
- **Architecture**: ConvNeXt-Tiny/Small/Base for state-of-the-art performance
- **Features**: Modern CNN architecture with competitive accuracy
- **Use case**: High-accuracy requirements

```python
from models.classifiers import FashionConvNeXt

model = FashionConvNeXt(
    num_classes=50,
    pretrained=True,
    variant='convnext_tiny'
)
```

#### FashionViT
- **Architecture**: Vision Transformer with attention mechanisms
- **Features**: Patch-based processing, attention visualization
- **Use case**: Research and experimentation

```python
from models.classifiers import FashionViT

model = FashionViT(
    num_classes=50,
    pretrained=True,
    variant='vit_base_patch16_224'
)
```

#### FashionEfficientNet
- **Architecture**: EfficientNet-B0 to B7 with balanced efficiency
- **Features**: Compound scaling, optimal accuracy-efficiency trade-off
- **Use case**: Production deployment with size constraints

```python
from models.classifiers import FashionEfficientNet

model = FashionEfficientNet(
    num_classes=50,
    pretrained=True,
    variant='efficientnet_b0'
)
```

#### FashionMultiScale
- **Architecture**: Multi-scale feature extraction for varying item sizes
- **Features**: Processes images at multiple resolutions
- **Use case**: Fashion items with varying scales and details

```python
from models.classifiers import FashionMultiScale

model = FashionMultiScale(
    num_classes=50,
    backbone_type='resnet50',
    scales=[224, 288, 384]
)
```

### 2. Ensemble Methods

#### Soft Voting Ensemble
- **Method**: Averages class probabilities from multiple models
- **Features**: Weighted combination, temperature scaling
- **Use case**: Improved accuracy through model diversity

```python
from models.ensemble import create_ensemble

ensemble = create_ensemble(
    models=[model1, model2, model3],
    method='soft_voting',
    weights=[0.4, 0.3, 0.3]
)
```

#### Hard Voting Ensemble
- **Method**: Majority vote from model predictions
- **Features**: Discrete voting, interpretable decisions
- **Use case**: Robust predictions with clear consensus

```python
ensemble = create_ensemble(
    models=[model1, model2, model3],
    method='hard_voting',
    weights=[1.0, 1.0, 1.0]
)
```

#### Weighted Ensemble
- **Method**: Learnable weights optimized during training
- **Features**: Automatic weight optimization
- **Use case**: Adaptive model combination

```python
ensemble = create_ensemble(
    models=[model1, model2, model3],
    method='weighted',
    learning_rate=0.001
)
```

#### Stacked Ensemble
- **Method**: Meta-learner combines model predictions
- **Features**: Second-level learning, feature combination
- **Use case**: Complex prediction patterns

```python
ensemble = create_ensemble(
    models=[model1, model2, model3],
    method='stacked',
    use_original_features=True
)
```

#### Dynamic Ensemble
- **Method**: Input-dependent model selection
- **Features**: Adaptive model choosing based on input
- **Use case**: Optimal model selection per input

```python
ensemble = create_ensemble(
    models=[model1, model2, model3],
    method='dynamic',
    top_k=2
)
```

### 3. Custom Loss Functions

#### Focal Loss
- **Purpose**: Addresses class imbalance by focusing on hard examples
- **Parameters**: `alpha` (class weighting), `gamma` (focusing parameter)

```python
from models.losses import FocalLoss

criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

#### Label Smoothing Loss
- **Purpose**: Regularization to prevent overfitting
- **Parameters**: `smoothing` (smoothing factor)

```python
from models.losses import LabelSmoothingLoss

criterion = LabelSmoothingLoss(
    num_classes=50,
    smoothing=0.1
)
```

#### Triplet Loss
- **Purpose**: Learns embeddings for similarity tasks
- **Parameters**: `margin`, `mining_strategy`

```python
from models.losses import FashionTripletLoss

criterion = FashionTripletLoss(
    margin=1.0,
    mining_strategy='hard'
)
```

#### Center Loss
- **Purpose**: Intra-class compactness
- **Parameters**: `alpha` (center update rate)

```python
from models.losses import FashionCenterLoss

criterion = FashionCenterLoss(
    num_classes=50,
    feature_dim=512,
    alpha=0.5
)
```

#### ArcFace Loss
- **Purpose**: Angular margin for improved separability
- **Parameters**: `margin` (angular margin), `scale` (feature scale)

```python
from models.losses import FashionArcFaceLoss

criterion = FashionArcFaceLoss(
    in_features=512,
    out_features=50,
    margin=0.5,
    scale=64.0
)
```

### 4. Model Factory

#### Easy Model Creation
```python
from models.model_factory import ModelFactory

# Create individual model
model = ModelFactory.create_model(
    model_type='resnet50',
    config={
        'num_classes': 50,
        'pretrained': True,
        'dropout_rate': 0.1
    }
)

# Create ensemble
model_configs = [
    {'model_type': 'resnet', 'pretrained': True},
    {'model_type': 'mobilenet', 'pretrained': True},
    {'model_type': 'convnext', 'pretrained': True}
]

ensemble = ModelFactory.create_ensemble(
    model_configs,
    ensemble_method='soft_voting'
)
```

#### Complete Training Setup
```python
from models.model_factory import create_complete_training_setup

training_setup = create_complete_training_setup(
    model_config={
        'type': 'resnet',
        'num_classes': 50,
        'pretrained': True
    },
    training_config={
        'optimizer': {'type': 'adamw', 'learning_rate': 0.001},
        'scheduler': {'type': 'cosine', 'params': {'T_max': 100}},
        'loss': {'type': 'focal', 'params': {'alpha': 0.25, 'gamma': 2.0}}
    }
)
```

### 5. Quantization and Compression

#### Post-Training Quantization
```python
from models.quantization import PostTrainingQuantizer, QuantizationConfig

# Dynamic quantization
quantizer = PostTrainingQuantizer(
    QuantizationConfig(quantization_type='dynamic')
)
quantized_model = quantizer.quantize_dynamic(model)

# Static quantization
quantizer = PostTrainingQuantizer(
    QuantizationConfig(quantization_type='static')
)
quantized_model = quantizer.quantize_static(model, calibration_dataloader)
```

#### Model Compression
```python
from models.quantization import ModelCompressor

compressor = ModelCompressor()
compressed_models = compressor.compress_model(
    model,
    compression_config={
        'pruning': {'enabled': True, 'ratio': 0.5},
        'quantization': {'enabled': True, 'type': 'dynamic'}
    },
    calibration_dataloader=dataloader
)
```

## Fashion Categories

The models support 50 fashion categories from the DeepFashion dataset:

**Clothing**: shirt, dress, pants, skirt, jacket, coat, sweater, t-shirt, jeans, shorts, blazer, hoodie, cardigan, vest, suit, jumpsuit, romper, leggings, sweatpants, tank_top, blouse, tunic, kimono, poncho, overalls, tracksuit, swimwear, underwear

**Footwear**: shoes, boots, sandals, sneakers, heels, flats

**Accessories**: bag, backpack, purse, wallet, belt, hat, cap, scarf, gloves, sunglasses, watch, necklace, bracelet, earrings, ring, socks

## Usage Examples

### Basic Classification
```python
import torch
from models.classifiers import FashionResNet

# Create model
model = FashionResNet(num_classes=50, pretrained=True)

# Load image (3, 224, 224)
image = torch.randn(1, 3, 224, 224)

# Inference
model.eval()
with torch.no_grad():
    outputs = model(image)
    predicted_class = torch.argmax(outputs, dim=1)
```

### Ensemble Prediction
```python
from models.ensemble import create_ensemble
from models.classifiers import FashionResNet, FashionMobileNet

# Create models
models = [
    FashionResNet(num_classes=50, pretrained=True),
    FashionMobileNet(num_classes=50, pretrained=True)
]

# Create ensemble
ensemble = create_ensemble(models, method='soft_voting')

# Prediction
with torch.no_grad():
    outputs = ensemble(image)
    predicted_class = torch.argmax(outputs, dim=1)
```

### Custom Training Loop
```python
from models.model_factory import create_complete_training_setup

# Create complete setup
setup = create_complete_training_setup(
    model_config={'type': 'resnet', 'num_classes': 50},
    training_config={
        'optimizer': {'type': 'adamw', 'learning_rate': 0.001},
        'loss': {'type': 'focal', 'params': {'alpha': 0.25, 'gamma': 2.0}}
    }
)

model = setup['model']
optimizer = setup['optimizer']
criterion = setup['loss_fn']

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

## Performance Considerations

### Model Sizes and Speeds (Approximate)
- **FashionMobileNet**: ~15MB, 10ms inference
- **FashionResNet**: ~100MB, 25ms inference
- **FashionConvNeXt**: ~120MB, 30ms inference
- **FashionViT**: ~350MB, 50ms inference
- **FashionEfficientNet**: ~20MB, 15ms inference

### Memory Usage
- **Training**: 8-16GB GPU memory recommended
- **Inference**: 2-4GB GPU memory for most models
- **Quantized models**: 50-75% memory reduction

## Best Practices

1. **Model Selection**:
   - Use MobileNet for mobile/edge deployment
   - Use ResNet for balanced performance
   - Use ConvNeXt for highest accuracy
   - Use ViT for research/experimentation

2. **Training**:
   - Start with pretrained models
   - Use focal loss for imbalanced datasets
   - Apply label smoothing for regularization
   - Use ensemble methods for production

3. **Deployment**:
   - Quantize models for production
   - Use dynamic quantization for CPU inference
   - Use static quantization for optimal performance
   - Consider model pruning for size constraints

4. **Ensemble Strategy**:
   - Combine diverse architectures
   - Use soft voting for probability outputs
   - Consider computational budget
   - Evaluate diversity metrics

## Installation

```bash
# Install requirements
pip install -r classification_requirements.txt

# Install the package
pip install -e .
```

## Contributing

When adding new models or features:
1. Follow the existing code structure
2. Add comprehensive docstrings
3. Include type hints
4. Add examples and tests
5. Update documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.