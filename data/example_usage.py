"""
Example usage of the fashion detection data modules.

This script demonstrates how to use the data loading and preprocessing
components for fashion detection tasks.
"""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Import our data modules
from dataset import DeepFashionDataset, DeepFashion2Dataset, create_dataset
from transforms import create_transform_pipeline, get_clip_transform
from dataloader import create_dataloader, create_train_val_dataloaders
from utils import (
    validate_dataset_structure, 
    calculate_dataset_statistics,
    visualize_dataset_statistics
)


def example_deepfashion_loading():
    """Example of loading DeepFashion dataset."""
    print("=== DeepFashion Dataset Example ===")
    
    # Replace with your actual dataset path
    dataset_path = "/path/to/deepfashion"
    
    # Validate dataset structure
    validation_report = validate_dataset_structure(dataset_path, 'deepfashion')
    if not validation_report['valid']:
        print("Dataset validation failed:")
        for error in validation_report['errors']:
            print(f"  - {error}")
        return None
    
    # Create transforms
    train_transform = create_transform_pipeline(
        mode='train',
        size=224,
        augment_level='medium'
    )
    
    val_transform = create_transform_pipeline(
        mode='val',
        size=224,
        augment_level='none'
    )
    
    # Create datasets
    train_dataset = DeepFashionDataset(
        root_dir=dataset_path,
        split='train',
        transform=train_transform,
        load_landmarks=True,
        load_attributes=True
    )
    
    val_dataset = DeepFashionDataset(
        root_dir=dataset_path,
        split='val',
        transform=val_transform,
        load_landmarks=True,
        load_attributes=True
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader, val_loader = create_train_val_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=32,
        num_workers=4,
        balanced_sampling=True
    )
    
    # Test loading a batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    if 'images' in batch:
        print(f"Image batch shape: {batch['images'].shape}")
    if 'labels' in batch:
        print(f"Label batch shape: {batch['labels'].shape}")
    
    return train_dataset, val_dataset


def example_deepfashion2_loading():
    """Example of loading DeepFashion2 dataset."""
    print("\n=== DeepFashion2 Dataset Example ===")
    
    # Replace with your actual dataset path
    dataset_path = "/path/to/deepfashion2"
    
    # Validate dataset structure
    validation_report = validate_dataset_structure(dataset_path, 'deepfashion2')
    if not validation_report['valid']:
        print("Dataset validation failed:")
        for error in validation_report['errors']:
            print(f"  - {error}")
        return None
    
    # Create dataset with segmentation transforms
    from transforms import get_segmentation_transform
    
    transform = get_segmentation_transform(mode='train', size=512)
    
    dataset = DeepFashion2Dataset(
        root_dir=dataset_path,
        split='train',
        load_masks=True,
        load_keypoints=True,
        categories=[1, 2, 3],  # Only load specific categories
        transform=transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=16,
        num_workers=4,
        shuffle=True
    )
    
    # Test loading a batch
    batch = next(iter(dataloader))
    print(f"Batch keys: {batch.keys()}")
    if 'images' in batch:
        print(f"Image batch shape: {batch['images'].shape}")
    if 'targets' in batch:
        print(f"Number of targets: {len(batch['targets'])}")
    
    return dataset


def example_s3_loading():
    """Example of loading data from S3."""
    print("\n=== S3 Loading Example ===")
    
    try:
        from s3_loader import S3DataLoader, S3DatasetWrapper
        
        # Initialize S3 loader
        s3_loader = S3DataLoader(
            bucket_name='your-fashion-dataset-bucket',
            cache_dir='./cache',
            max_workers=4
        )
        
        # Test loading an image
        try:
            image = s3_loader.load_image('path/to/image.jpg')
            print(f"Loaded image size: {image.size}")
        except Exception as e:
            print(f"Error loading from S3: {e}")
        
        # Wrap existing dataset for S3 usage
        # This would require modifying the dataset to return S3 keys
        # dataset = S3DatasetWrapper(base_dataset, s3_loader)
        
    except ImportError:
        print("S3 functionality requires boto3: pip install boto3")


def example_statistics_and_visualization():
    """Example of calculating and visualizing dataset statistics."""
    print("\n=== Dataset Statistics Example ===")
    
    # This would use an actual dataset from the previous examples
    # For demonstration, we'll show the API
    dataset_path = "/path/to/dataset"
    
    if not os.path.exists(dataset_path):
        print("Dataset path not found, skipping statistics example")
        return
    
    # Create a simple dataset
    dataset = create_dataset(
        'deepfashion',
        root_dir=dataset_path,
        split='train'
    )
    
    # Calculate statistics
    stats = calculate_dataset_statistics(
        dataset,
        num_samples=1000,  # Sample 1000 images for statistics
        save_path='./dataset_stats.json'
    )
    
    print("Dataset statistics calculated:")
    print(f"  - Total samples: {stats['dataset_info']['total_samples']}")
    print(f"  - Samples analyzed: {stats['dataset_info']['samples_analyzed']}")
    
    if 'image_statistics' in stats:
        img_stats = stats['image_statistics']
        print(f"  - Average image size: {img_stats['width']['mean']:.0f}x{img_stats['height']['mean']:.0f}")
        print(f"  - Average aspect ratio: {img_stats['aspect_ratio']['mean']:.2f}")
    
    if 'label_statistics' in stats:
        print(f"  - Number of classes: {stats['label_statistics']['num_classes']}")
    
    # Create visualizations
    figures = visualize_dataset_statistics(
        stats,
        save_dir='./plots',
        show=False  # Set to True to display plots
    )
    
    print(f"Created {len(figures)} visualization plots")


def example_clip_preprocessing():
    """Example of CLIP-specific preprocessing."""
    print("\n=== CLIP Preprocessing Example ===")
    
    # Create CLIP transform
    clip_transform = get_clip_transform()
    
    # Create dataset with CLIP preprocessing
    dataset_path = "/path/to/dataset"
    
    if os.path.exists(dataset_path):
        dataset = DeepFashionDataset(
            root_dir=dataset_path,
            split='train',
            transform=clip_transform
        )
        
        # Test loading
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"CLIP preprocessed image shape: {sample['image'].shape}")
            print(f"Image tensor range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    else:
        print("Dataset path not found, skipping CLIP example")


def main():
    """Run all examples."""
    print("Fashion Detection Data Module Examples")
    print("=" * 40)
    
    # Note: These examples require actual dataset paths
    # Replace the paths with your actual dataset locations
    
    # Example 1: DeepFashion dataset
    try:
        example_deepfashion_loading()
    except Exception as e:
        print(f"DeepFashion example failed: {e}")
    
    # Example 2: DeepFashion2 dataset
    try:
        example_deepfashion2_loading()
    except Exception as e:
        print(f"DeepFashion2 example failed: {e}")
    
    # Example 3: S3 loading
    try:
        example_s3_loading()
    except Exception as e:
        print(f"S3 example failed: {e}")
    
    # Example 4: Statistics and visualization
    try:
        example_statistics_and_visualization()
    except Exception as e:
        print(f"Statistics example failed: {e}")
    
    # Example 5: CLIP preprocessing
    try:
        example_clip_preprocessing()
    except Exception as e:
        print(f"CLIP example failed: {e}")
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main()