# Fashion Detection Data Module

This module provides comprehensive data loading, preprocessing, and augmentation utilities for fashion detection tasks. It supports both DeepFashion and DeepFashion2 datasets with local and S3 storage options.

## Features

- **Multiple Dataset Support**: DeepFashion and DeepFashion2 datasets
- **Flexible Storage**: Local filesystem and AWS S3 support
- **Advanced Preprocessing**: Fashion-specific transforms and augmentations
- **Efficient Loading**: Balanced sampling, distributed training, and caching
- **Comprehensive Tools**: Data validation, statistics, and visualization

## Module Structure

```
data/
├── __init__.py           # Main module exports
├── dataset.py            # Dataset classes
├── transforms.py         # Image transforms and augmentations
├── dataloader.py         # Custom data loaders and samplers
├── s3_loader.py          # S3 integration utilities
├── utils.py              # Helper functions and utilities
├── example_usage.py      # Usage examples
└── README.md             # This file
```

## Quick Start

### Basic Usage

```python
from data import create_dataset, create_transform_pipeline, create_dataloader

# Create transforms
transform = create_transform_pipeline(
    mode='train',
    size=224,
    augment_level='medium'
)

# Create dataset
dataset = create_dataset(
    'deepfashion',
    root_dir='/path/to/deepfashion',
    split='train',
    transform=transform
)

# Create dataloader
dataloader = create_dataloader(
    dataset,
    batch_size=32,
    balanced_sampling=True
)
```

### Advanced Usage

```python
from data import DeepFashionDataset, BalancedBatchSampler, S3DataLoader

# S3 dataset loading
s3_loader = S3DataLoader(
    bucket_name='my-fashion-bucket',
    cache_dir='./cache'
)

dataset = DeepFashionDataset(
    root_dir='s3://my-bucket/deepfashion',
    split='train',
    use_s3=True,
    s3_bucket='my-fashion-bucket',
    cache_dir='./cache'
)

# Balanced sampling
sampler = BalancedBatchSampler(
    dataset,
    batch_size=32,
    num_classes=50,
    samples_per_class=2
)
```

## Dataset Classes

### DeepFashionDataset

Loads the DeepFashion dataset with support for:
- Bounding boxes
- Fashion landmarks
- Clothing attributes
- Category labels

```python
dataset = DeepFashionDataset(
    root_dir='/path/to/deepfashion',
    split='train',
    categories=['dress', 'shirt'],  # Optional category filter
    load_landmarks=True,
    load_attributes=True,
    transform=transform
)
```

### DeepFashion2Dataset

Loads the DeepFashion2 dataset with support for:
- Instance segmentation masks
- Keypoints
- Bounding boxes
- Category labels

```python
dataset = DeepFashion2Dataset(
    root_dir='/path/to/deepfashion2',
    split='train',
    load_masks=True,
    load_keypoints=True,
    categories=[1, 2, 3],  # Category IDs
    transform=transform
)
```

## Transform Pipeline

### Standard Transforms

```python
from data import create_transform_pipeline

# Training transforms with augmentation
train_transform = create_transform_pipeline(
    mode='train',
    size=224,
    augment_level='heavy'  # 'none', 'light', 'medium', 'heavy'
)

# Validation transforms
val_transform = create_transform_pipeline(
    mode='val',
    size=224,
    augment_level='none'
)
```

### Custom Transforms

```python
from data import FashionResize, FashionColorJitter, CutMix, MixUp

# Fashion-specific transforms
transforms = [
    FashionResize(size=224),
    FashionColorJitter(brightness=0.2, contrast=0.2),
    CutMix(alpha=1.0, p=0.5),
    MixUp(alpha=0.2, p=0.5)
]
```

### CLIP Preprocessing

```python
from data import get_clip_transform

clip_transform = get_clip_transform()
dataset = DeepFashionDataset(
    root_dir='/path/to/data',
    transform=clip_transform
)
```

## Data Loaders

### Basic DataLoader

```python
from data import create_dataloader

dataloader = create_dataloader(
    dataset,
    batch_size=32,
    num_workers=4,
    shuffle=True
)
```

### Balanced Sampling

```python
dataloader = create_dataloader(
    dataset,
    batch_size=32,
    balanced_sampling=True,
    num_classes=50,
    samples_per_class=2
)
```

### Distributed Training

```python
dataloader = create_dataloader(
    dataset,
    batch_size=32,
    distributed=True,
    num_workers=4
)
```

### Train/Val Split

```python
from data import create_train_val_dataloaders

train_loader, val_loader = create_train_val_dataloaders(
    train_dataset,
    val_dataset,
    batch_size=32,
    num_workers=4,
    balanced_sampling=True
)
```

## S3 Integration

### S3 Data Loader

```python
from data import S3DataLoader

s3_loader = S3DataLoader(
    bucket_name='my-fashion-bucket',
    cache_dir='./cache',
    max_workers=8
)

# Load individual files
image = s3_loader.load_image('path/to/image.jpg')
annotations = s3_loader.load_json('path/to/annotations.json')

# Batch download
files = s3_loader.batch_download([
    'image1.jpg',
    'image2.jpg',
    'annotations.json'
])
```

### S3 Dataset Wrapper

```python
from data import S3DatasetWrapper

# Wrap existing dataset for S3 usage
s3_dataset = S3DatasetWrapper(
    dataset=base_dataset,
    s3_loader=s3_loader
)
```

## Utilities

### Dataset Validation

```python
from data import validate_dataset_structure

report = validate_dataset_structure(
    root_dir='/path/to/dataset',
    dataset_type='deepfashion'
)

if report['valid']:
    print("Dataset structure is valid")
else:
    print("Validation errors:", report['errors'])
```

### Dataset Statistics

```python
from data import calculate_dataset_statistics, visualize_dataset_statistics

# Calculate statistics
stats = calculate_dataset_statistics(
    dataset,
    num_samples=1000,
    save_path='./stats.json'
)

# Create visualizations
figures = visualize_dataset_statistics(
    stats,
    save_dir='./plots',
    show=True
)
```

### Data Validation

```python
from data import verify_image_integrity

# Check single image
is_valid = verify_image_integrity('/path/to/image.jpg')

# Check with detailed info
is_valid, info = verify_image_integrity('/path/to/image.jpg', return_info=True)
```

### Train/Val Split

```python
from data import create_train_val_split

train_indices, val_indices = create_train_val_split(
    dataset,
    val_ratio=0.2,
    stratify=True,
    random_seed=42
)
```

## Configuration Examples

### Training Configuration

```python
# Configuration for training
config = {
    'dataset': {
        'name': 'deepfashion',
        'root_dir': '/path/to/deepfashion',
        'use_s3': False,
        'load_landmarks': True,
        'load_attributes': True
    },
    'transform': {
        'size': 224,
        'augment_level': 'medium',
        'normalize': True
    },
    'dataloader': {
        'batch_size': 32,
        'num_workers': 4,
        'balanced_sampling': True,
        'distributed': True
    }
}
```

### Inference Configuration

```python
# Configuration for inference
config = {
    'dataset': {
        'name': 'deepfashion',
        'root_dir': '/path/to/test_images',
        'split': 'test'
    },
    'transform': {
        'size': 224,
        'augment_level': 'none',
        'normalize': True
    },
    'dataloader': {
        'batch_size': 1,
        'num_workers': 1,
        'shuffle': False
    }
}
```

## Error Handling

The module includes comprehensive error handling:

- **Missing Files**: Graceful handling of missing images or annotations
- **Corrupted Images**: Automatic detection and skipping of corrupted files
- **S3 Errors**: Robust error handling for network issues and access problems
- **Validation**: Dataset structure validation with detailed error messages

## Performance Tips

1. **Caching**: Use S3 caching for faster repeated access
2. **Parallel Loading**: Increase `num_workers` for faster data loading
3. **Memory Management**: Use appropriate batch sizes to avoid OOM errors
4. **Preprocessing**: Pre-compute transforms for static augmentations

## Dependencies

Required packages:
- `torch`
- `torchvision`
- `numpy`
- `PIL` (Pillow)
- `pandas`
- `matplotlib`
- `seaborn`
- `tqdm`
- `boto3` (for S3 functionality)
- `scikit-learn` (for stratified splits)

Install with:
```bash
pip install torch torchvision numpy pillow pandas matplotlib seaborn tqdm boto3 scikit-learn
```

## Examples

See `example_usage.py` for complete working examples of all functionality.

## License

This module is part of the fashion detection system and follows the same license terms.