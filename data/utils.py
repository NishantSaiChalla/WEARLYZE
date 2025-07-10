"""
Utility functions for data handling, statistics, and validation.

This module provides helper functions for dataset management, data validation,
statistics calculation, and common data processing operations.
"""

import os
import json
import logging
import hashlib
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, desc=None: x

logger = logging.getLogger(__name__)


def validate_dataset_structure(
    root_dir: Union[str, Path],
    dataset_type: str = 'deepfashion'
) -> Dict[str, Any]:
    """
    Validate dataset directory structure and return validation report.
    
    Args:
        root_dir: Root directory of the dataset
        dataset_type: Type of dataset ('deepfashion' or 'deepfashion2')
        
    Returns:
        Dictionary containing validation results
    """
    root_dir = Path(root_dir)
    report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'structure': {}
    }
    
    if not root_dir.exists():
        report['valid'] = False
        report['errors'].append(f"Root directory does not exist: {root_dir}")
        return report
    
    if dataset_type == 'deepfashion':
        # Check DeepFashion structure
        required_dirs = ['Anno', 'Eval', 'Img']
        for dir_name in required_dirs:
            dir_path = root_dir / dir_name
            if not dir_path.exists():
                report['valid'] = False
                report['errors'].append(f"Missing required directory: {dir_name}")
            else:
                report['structure'][dir_name] = str(dir_path)
        
        # Check annotation files
        anno_files = [
            'Anno/list_bbox.txt',
            'Anno/list_landmarks.txt',
            'Anno/list_attr_cloth.txt',
            'Anno/list_category_cloth.txt',
            'Anno/list_category_img.txt'
        ]
        
        for file_path in anno_files:
            full_path = root_dir / file_path
            if not full_path.exists():
                report['warnings'].append(f"Missing annotation file: {file_path}")
            else:
                report['structure'][file_path] = str(full_path)
                
    elif dataset_type == 'deepfashion2':
        # Check DeepFashion2 structure
        splits = ['train', 'validation', 'test']
        for split in splits:
            split_dir = root_dir / split
            if split_dir.exists():
                # Check for image and annotation subdirectories
                image_dir = split_dir / 'image'
                anno_dir = split_dir / 'annos'
                
                if not image_dir.exists():
                    report['warnings'].append(f"Missing image directory for {split}")
                else:
                    # Count images
                    num_images = len(list(image_dir.glob('*.jpg')))
                    report['structure'][f'{split}/images'] = num_images
                
                if split != 'test' and not anno_dir.exists():
                    report['warnings'].append(f"Missing annotation directory for {split}")
                elif anno_dir.exists():
                    # Count annotations
                    num_annos = len(list(anno_dir.glob('*.json')))
                    report['structure'][f'{split}/annotations'] = num_annos
            else:
                if split != 'test':  # Test split is optional
                    report['warnings'].append(f"Missing split directory: {split}")
    else:
        report['valid'] = False
        report['errors'].append(f"Unknown dataset type: {dataset_type}")
    
    return report


def calculate_dataset_statistics(
    dataset: Any,
    num_samples: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a fashion dataset.
    
    Args:
        dataset: Dataset instance
        num_samples: Number of samples to analyze (None for all)
        save_path: Path to save statistics report
        
    Returns:
        Dictionary containing dataset statistics
    """
    stats = {
        'total_samples': len(dataset),
        'samples_analyzed': 0,
        'image_stats': {
            'width': [],
            'height': [],
            'aspect_ratios': [],
            'channels': []
        },
        'label_distribution': defaultdict(int),
        'attribute_distribution': defaultdict(int),
        'bbox_stats': {
            'widths': [],
            'heights': [],
            'areas': [],
            'aspect_ratios': []
        },
        'keypoint_stats': {
            'num_visible': [],
            'visibility_rate': []
        }
    }
    
    # Determine number of samples to analyze
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))
    
    # Analyze samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    iter_indices = tqdm(indices, desc="Calculating statistics") if HAS_TQDM else indices
    for idx in iter_indices:
        try:
            sample = dataset[idx]
            stats['samples_analyzed'] += 1
            
            # Image statistics
            if 'image' in sample:
                if isinstance(sample['image'], Image.Image):
                    img = sample['image']
                    width, height = img.size
                    channels = len(img.getbands())
                elif isinstance(sample['image'], torch.Tensor):
                    if sample['image'].dim() == 3:
                        channels, height, width = sample['image'].shape
                    else:
                        height, width = sample['image'].shape
                        channels = 1
                elif isinstance(sample['image'], np.ndarray):
                    if sample['image'].ndim == 3:
                        height, width, channels = sample['image'].shape
                    else:
                        height, width = sample['image'].shape
                        channels = 1
                else:
                    continue
                
                stats['image_stats']['width'].append(width)
                stats['image_stats']['height'].append(height)
                stats['image_stats']['aspect_ratios'].append(width / height)
                stats['image_stats']['channels'].append(channels)
            
            # Label statistics
            if 'label' in sample:
                stats['label_distribution'][sample['label']] += 1
            elif 'category_id' in sample:
                stats['label_distribution'][sample['category_id']] += 1
            
            # Attribute statistics
            if 'attributes' in sample and 'values' in sample['attributes']:
                for i, attr_val in enumerate(sample['attributes']['values']):
                    if attr_val == 1:  # Positive attribute
                        attr_name = (sample['attributes']['names'][i] 
                                   if i < len(sample['attributes']['names']) 
                                   else f'attr_{i}')
                        stats['attribute_distribution'][attr_name] += 1
            
            # Bounding box statistics
            if 'bbox' in sample:
                bbox = sample['bbox']
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    stats['bbox_stats']['widths'].append(bbox_width)
                    stats['bbox_stats']['heights'].append(bbox_height)
                    stats['bbox_stats']['areas'].append(bbox_width * bbox_height)
                    if bbox_height > 0:
                        stats['bbox_stats']['aspect_ratios'].append(bbox_width / bbox_height)
            
            # Keypoint statistics
            if 'keypoints' in sample or ('landmarks' in sample and 'landmarks' in sample['landmarks']):
                keypoints = (sample.get('keypoints') or 
                           sample['landmarks'].get('landmarks', []))
                if keypoints:
                    num_visible = sum(1 for kpt in keypoints 
                                    if isinstance(kpt, dict) and kpt.get('visibility', 0) > 0)
                    stats['keypoint_stats']['num_visible'].append(num_visible)
                    stats['keypoint_stats']['visibility_rate'].append(
                        num_visible / len(keypoints) if len(keypoints) > 0 else 0
                    )
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            continue
    
    # Calculate summary statistics
    summary = {
        'dataset_info': {
            'total_samples': stats['total_samples'],
            'samples_analyzed': stats['samples_analyzed']
        },
        'image_statistics': {},
        'label_statistics': {},
        'bbox_statistics': {},
        'keypoint_statistics': {}
    }
    
    # Image summary
    if stats['image_stats']['width']:
        summary['image_statistics'] = {
            'width': {
                'mean': np.mean(stats['image_stats']['width']),
                'std': np.std(stats['image_stats']['width']),
                'min': np.min(stats['image_stats']['width']),
                'max': np.max(stats['image_stats']['width'])
            },
            'height': {
                'mean': np.mean(stats['image_stats']['height']),
                'std': np.std(stats['image_stats']['height']),
                'min': np.min(stats['image_stats']['height']),
                'max': np.max(stats['image_stats']['height'])
            },
            'aspect_ratio': {
                'mean': np.mean(stats['image_stats']['aspect_ratios']),
                'std': np.std(stats['image_stats']['aspect_ratios']),
                'min': np.min(stats['image_stats']['aspect_ratios']),
                'max': np.max(stats['image_stats']['aspect_ratios'])
            }
        }
    
    # Label summary
    if stats['label_distribution']:
        summary['label_statistics'] = {
            'num_classes': len(stats['label_distribution']),
            'samples_per_class': dict(stats['label_distribution']),
            'class_balance': {
                'min_samples': min(stats['label_distribution'].values()),
                'max_samples': max(stats['label_distribution'].values()),
                'mean_samples': np.mean(list(stats['label_distribution'].values())),
                'std_samples': np.std(list(stats['label_distribution'].values()))
            }
        }
    
    # BBox summary
    if stats['bbox_stats']['widths']:
        summary['bbox_statistics'] = {
            'area': {
                'mean': np.mean(stats['bbox_stats']['areas']),
                'std': np.std(stats['bbox_stats']['areas']),
                'min': np.min(stats['bbox_stats']['areas']),
                'max': np.max(stats['bbox_stats']['areas'])
            },
            'aspect_ratio': {
                'mean': np.mean(stats['bbox_stats']['aspect_ratios']),
                'std': np.std(stats['bbox_stats']['aspect_ratios'])
            }
        }
    
    # Keypoint summary
    if stats['keypoint_stats']['num_visible']:
        summary['keypoint_statistics'] = {
            'avg_visible_keypoints': np.mean(stats['keypoint_stats']['num_visible']),
            'avg_visibility_rate': np.mean(stats['keypoint_stats']['visibility_rate'])
        }
    
    # Save statistics if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Statistics saved to {save_path}")
    
    return summary


def visualize_dataset_statistics(
    stats: Dict[str, Any],
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = True
) -> Dict[str, Any]:
    """
    Create visualizations for dataset statistics.
    
    Args:
        stats: Statistics dictionary from calculate_dataset_statistics
        save_dir: Directory to save plots
        show: Whether to display plots
        
    Returns:
        Dictionary of matplotlib figures or None if matplotlib not available
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib not available. Install with: pip install matplotlib seaborn")
        return {}
    
    figures = {}
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Image size distribution
    if 'image_statistics' in stats and stats['image_statistics']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Width/Height distribution
        img_stats = stats['image_statistics']
        if 'width' in img_stats and 'height' in img_stats:
            ax1.bar(['Width', 'Height'], 
                   [img_stats['width']['mean'], img_stats['height']['mean']],
                   yerr=[img_stats['width']['std'], img_stats['height']['std']])
            ax1.set_ylabel('Pixels')
            ax1.set_title('Average Image Dimensions')
            
            # Aspect ratio distribution
            ax2.hist([img_stats['aspect_ratio']['mean']], bins=30, alpha=0.7)
            ax2.set_xlabel('Aspect Ratio')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Aspect Ratio Distribution')
        
        figures['image_stats'] = fig
    
    # 2. Class distribution
    if 'label_statistics' in stats and 'samples_per_class' in stats['label_statistics']:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        class_counts = stats['label_statistics']['samples_per_class']
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        # Sort by count
        sorted_indices = np.argsort(counts)[::-1]
        classes = [classes[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        # Show top 20 classes
        ax.bar(range(min(20, len(classes))), counts[:20])
        ax.set_xticks(range(min(20, len(classes))))
        ax.set_xticklabels([str(c) for c in classes[:20]], rotation=45, ha='right')
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Class Distribution (Top 20)')
        
        figures['class_distribution'] = fig
    
    # 3. Bounding box statistics
    if 'bbox_statistics' in stats and stats['bbox_statistics']:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bbox_stats = stats['bbox_statistics']
        if 'area' in bbox_stats:
            # Create box plot for bbox areas
            area_data = {
                'Mean': bbox_stats['area']['mean'],
                'Min': bbox_stats['area']['min'],
                'Max': bbox_stats['area']['max']
            }
            ax.bar(area_data.keys(), area_data.values())
            ax.set_ylabel('Area (pixels²)')
            ax.set_title('Bounding Box Area Statistics')
        
        figures['bbox_stats'] = fig
    
    # Save figures if requested
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for name, fig in figures.items():
            fig.savefig(save_dir / f'{name}.png', dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot: {save_dir / f'{name}.png'}")
    
    # Show plots if requested
    if show:
        plt.show()
    else:
        plt.close('all')
    
    return figures


def verify_image_integrity(
    image_path: Union[str, Path],
    return_info: bool = False
) -> Union[bool, Tuple[bool, Dict[str, Any]]]:
    """
    Verify that an image file is valid and can be loaded.
    
    Args:
        image_path: Path to image file
        return_info: Whether to return additional image information
        
    Returns:
        Boolean indicating validity, optionally with image info
    """
    image_path = Path(image_path)
    info = {
        'valid': False,
        'error': None,
        'size': None,
        'mode': None,
        'format': None
    }
    
    try:
        # Try to open the image
        with Image.open(image_path) as img:
            # Verify it's not truncated
            img.verify()
            
        # Re-open to get info (verify() closes the file)
        with Image.open(image_path) as img:
            info['valid'] = True
            info['size'] = img.size
            info['mode'] = img.mode
            info['format'] = img.format
            
    except Exception as e:
        info['error'] = str(e)
        logger.debug(f"Invalid image {image_path}: {e}")
    
    if return_info:
        return info['valid'], info
    return info['valid']


def create_train_val_split(
    dataset: Any,
    val_ratio: float = 0.2,
    stratify: bool = True,
    random_seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Create train/validation split indices for a dataset.
    
    Args:
        dataset: Dataset instance
        val_ratio: Ratio of validation samples
        stratify: Whether to stratify by class labels
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices)
    """
    np.random.seed(random_seed)
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    
    if stratify:
        # Get labels for stratification
        labels = []
        for i in range(n_samples):
            sample = dataset[i]
            if 'label' in sample:
                labels.append(sample['label'])
            elif 'category_id' in sample:
                labels.append(sample['category_id'])
            else:
                # Can't stratify without labels
                stratify = False
                break
        
        if stratify and labels:
            # Stratified split
            try:
                from sklearn.model_selection import train_test_split
                train_indices, val_indices = train_test_split(
                    indices,
                    test_size=val_ratio,
                    stratify=labels,
                    random_state=random_seed
                )
                return train_indices.tolist(), val_indices.tolist()
            except ImportError:
                logger.warning("scikit-learn not available for stratified split. Using random split.")
                stratify = False
    
    # Random split
    np.random.shuffle(indices)
    split_idx = int(n_samples * (1 - val_ratio))
    train_indices = indices[:split_idx].tolist()
    val_indices = indices[split_idx:].tolist()
    
    return train_indices, val_indices


def compute_dataset_mean_std(
    dataset: Any,
    num_samples: Optional[int] = None,
    batch_size: int = 32
) -> Tuple[List[float], List[float]]:
    """
    Compute channel-wise mean and standard deviation for normalization.
    
    Args:
        dataset: Dataset instance
        num_samples: Number of samples to use (None for all)
        batch_size: Batch size for computation
        
    Returns:
        Tuple of (mean, std) lists
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for this function. Install with: pip install torch")
    
    from torch.utils.data import DataLoader
    
    # Create a temporary transform that only converts to tensor
    original_transform = None
    if hasattr(dataset, 'transform'):
        original_transform = dataset.transform
        # Note: This assumes torchvision.transforms is available
        try:
            import torchvision.transforms as transforms
            dataset.transform = transforms.ToTensor()
        except ImportError:
            logger.warning("torchvision not available. Skipping transform modification.")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Initialize running stats
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    # Determine number of batches to process
    if num_samples:
        num_batches = min(len(dataloader), num_samples // batch_size)
    else:
        num_batches = len(dataloader)
    
    # Compute stats
    iter_dataloader = tqdm(dataloader, desc="Computing mean/std") if HAS_TQDM else dataloader
    for i, batch in enumerate(iter_dataloader):
        if i >= num_batches:
            break
        
        if isinstance(batch, dict) and 'image' in batch:
            images = batch['image']
        elif isinstance(batch, (list, tuple)) and len(batch) > 0:
            images = batch[0]
        else:
            images = batch
        
        # Ensure images are in [0, 1] range
        if images.max() > 1:
            images = images / 255.0
        
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    # Restore original transform
    if hasattr(dataset, 'transform') and original_transform is not None:
        dataset.transform = original_transform
    
    return mean.tolist(), std.tolist()


def save_dataset_sample(
    dataset: Any,
    indices: List[int],
    save_dir: Union[str, Path],
    max_samples: int = 10
):
    """
    Save sample images and annotations from dataset for inspection.
    
    Args:
        dataset: Dataset instance
        indices: List of sample indices to save
        save_dir: Directory to save samples
        max_samples: Maximum number of samples to save
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i, idx in enumerate(indices[:max_samples]):
        try:
            sample = dataset[idx]
            
            # Save image
            if 'image' in sample:
                if isinstance(sample['image'], Image.Image):
                    img = sample['image']
                elif isinstance(sample['image'], torch.Tensor):
                    # Convert tensor to PIL
                    img_array = sample['image'].numpy()
                    if img_array.shape[0] == 3:  # CHW -> HWC
                        img_array = np.transpose(img_array, (1, 2, 0))
                    if img_array.max() <= 1:
                        img_array = (img_array * 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
                else:
                    continue
                
                img_path = save_dir / f'sample_{idx}_image.jpg'
                img.save(img_path)
                
                # Save annotations
                anno_data = {
                    'index': idx,
                    'image_path': str(img_path.name)
                }
                
                # Add other annotations
                for key in ['label', 'category_id', 'bbox', 'keypoints', 'attributes']:
                    if key in sample:
                        if isinstance(sample[key], torch.Tensor):
                            anno_data[key] = sample[key].tolist()
                        else:
                            anno_data[key] = sample[key]
                
                anno_path = save_dir / f'sample_{idx}_annotations.json'
                with open(anno_path, 'w') as f:
                    json.dump(anno_data, f, indent=2)
                
                logger.info(f"Saved sample {idx} to {save_dir}")
                
        except Exception as e:
            logger.error(f"Error saving sample {idx}: {e}")
            continue


def get_dataset_info(dataset: Any) -> Dict[str, Any]:
    """
    Get comprehensive information about a dataset.
    
    Args:
        dataset: Dataset instance
        
    Returns:
        Dictionary containing dataset information
    """
    info = {
        'type': type(dataset).__name__,
        'length': len(dataset),
        'attributes': [],
        'sample_keys': set(),
        'has_transform': hasattr(dataset, 'transform') and dataset.transform is not None
    }
    
    # Get dataset attributes
    for attr in dir(dataset):
        if not attr.startswith('_') and not callable(getattr(dataset, attr)):
            info['attributes'].append(attr)
    
    # Sample a few items to get keys
    sample_indices = np.random.choice(
        len(dataset), 
        min(5, len(dataset)), 
        replace=False
    )
    
    for idx in sample_indices:
        try:
            sample = dataset[idx]
            if isinstance(sample, dict):
                info['sample_keys'].update(sample.keys())
        except Exception:
            pass
    
    info['sample_keys'] = list(info['sample_keys'])
    
    return info