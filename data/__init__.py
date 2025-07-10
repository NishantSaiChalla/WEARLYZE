"""
Data loading and preprocessing module for fashion detection.

This module provides comprehensive data loading, preprocessing, and augmentation
utilities for fashion detection tasks.
"""

from .dataset import (
    BaseDataset,
    DeepFashionDataset,
    DeepFashion2Dataset,
    create_dataset
)

from .transforms import (
    FashionResize,
    RandomHorizontalFlip,
    FashionColorJitter,
    CutMix,
    MixUp,
    RandAugment,
    CLIPProcessor,
    SegmentationTransform,
    create_transform_pipeline,
    get_clip_transform,
    get_segmentation_transform
)

from .dataloader import (
    BalancedBatchSampler,
    WeightedSampler,
    DistributedSamplerWrapper,
    InfiniteDataLoader,
    fashion_collate_fn,
    create_dataloader,
    create_train_val_dataloaders
)

from .s3_loader import (
    S3DataLoader,
    S3DatasetWrapper
)

from .utils import (
    validate_dataset_structure,
    calculate_dataset_statistics,
    visualize_dataset_statistics,
    verify_image_integrity,
    create_train_val_split,
    compute_dataset_mean_std,
    save_dataset_sample,
    get_dataset_info
)

__all__ = [
    # Dataset classes
    'BaseDataset',
    'DeepFashionDataset',
    'DeepFashion2Dataset',
    'create_dataset',
    
    # Transform classes
    'FashionResize',
    'RandomHorizontalFlip',
    'FashionColorJitter',
    'CutMix',
    'MixUp',
    'RandAugment',
    'CLIPProcessor',
    'SegmentationTransform',
    'create_transform_pipeline',
    'get_clip_transform',
    'get_segmentation_transform',
    
    # DataLoader utilities
    'BalancedBatchSampler',
    'WeightedSampler',
    'DistributedSamplerWrapper',
    'InfiniteDataLoader',
    'fashion_collate_fn',
    'create_dataloader',
    'create_train_val_dataloaders',
    
    # S3 utilities
    'S3DataLoader',
    'S3DatasetWrapper',
    
    # Utility functions
    'validate_dataset_structure',
    'calculate_dataset_statistics',
    'visualize_dataset_statistics',
    'verify_image_integrity',
    'create_train_val_split',
    'compute_dataset_mean_std',
    'save_dataset_sample',
    'get_dataset_info'
]