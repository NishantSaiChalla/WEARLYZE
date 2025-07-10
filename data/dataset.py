"""
Dataset classes for fashion detection system.

This module provides dataset classes for loading and processing fashion datasets,
including DeepFashion and DeepFashion2, with support for both local and S3 storage.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """Base dataset class with common functionality for fashion datasets."""
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        use_s3: bool = False,
        s3_bucket: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize base dataset.
        
        Args:
            root_dir: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Image transformation pipeline
            use_s3: Whether to load data from S3
            s3_bucket: S3 bucket name if use_s3 is True
            cache_dir: Local cache directory for S3 data
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.use_s3 = use_s3
        self.s3_bucket = s3_bucket
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.use_s3 and not self.s3_bucket:
            raise ValueError("S3 bucket name must be provided when use_s3=True")
        
        self.samples = []
        self.annotations = {}
        
    def _load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load image from local path or S3.
        
        Args:
            image_path: Path to the image
            
        Returns:
            PIL Image object
        """
        try:
            if self.use_s3:
                from .s3_loader import S3DataLoader
                s3_loader = S3DataLoader(self.s3_bucket, self.cache_dir)
                image = s3_loader.load_image(str(image_path))
            else:
                image_path = self.root_dir / image_path
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")
                image = Image.open(image_path).convert('RGB')
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)


class DeepFashionDataset(BaseDataset):
    """
    Dataset class for DeepFashion dataset.
    
    Supports loading images with bounding boxes, landmarks, and attributes
    from the DeepFashion dataset structure.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = 'train',
        categories: Optional[List[str]] = None,
        load_landmarks: bool = True,
        load_attributes: bool = True,
        transform: Optional[transforms.Compose] = None,
        use_s3: bool = False,
        s3_bucket: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize DeepFashion dataset.
        
        Args:
            root_dir: Root directory of DeepFashion dataset
            split: Dataset split ('train', 'val', 'test')
            categories: List of categories to load (None for all)
            load_landmarks: Whether to load fashion landmarks
            load_attributes: Whether to load clothing attributes
            transform: Image transformation pipeline
            use_s3: Whether to load data from S3
            s3_bucket: S3 bucket name if use_s3 is True
            cache_dir: Local cache directory for S3 data
        """
        super().__init__(root_dir, split, transform, use_s3, s3_bucket, cache_dir)
        
        self.categories = categories
        self.load_landmarks = load_landmarks
        self.load_attributes = load_attributes
        
        # DeepFashion specific paths
        self.anno_dir = self.root_dir / 'Anno'
        self.eval_dir = self.root_dir / 'Eval'
        self.img_dir = self.root_dir / 'Img'
        
        # Load dataset
        self._load_dataset()
        
    def _load_dataset(self):
        """Load DeepFashion dataset annotations and file lists."""
        try:
            # Load list_eval_partition.txt for train/val/test split
            partition_file = self.eval_dir / 'list_eval_partition.txt'
            if partition_file.exists():
                df_partition = pd.read_csv(
                    partition_file, 
                    sep=r'\s+', 
                    skiprows=1,
                    names=['image_name', 'evaluation_status']
                )
                
                # Filter by split (train=1, val=2, test=3)
                split_map = {'train': 1, 'val': 2, 'test': 3}
                if self.split in split_map:
                    df_partition = df_partition[
                        df_partition['evaluation_status'] == split_map[self.split]
                    ]
                    
                self.image_list = df_partition['image_name'].tolist()
            else:
                logger.warning(f"Partition file not found: {partition_file}")
                self.image_list = []
            
            # Load bounding boxes
            self._load_bboxes()
            
            # Load landmarks if requested
            if self.load_landmarks:
                self._load_landmarks()
            
            # Load attributes if requested
            if self.load_attributes:
                self._load_attributes()
                
            # Create samples list
            self._create_samples()
            
        except Exception as e:
            logger.error(f"Error loading DeepFashion dataset: {e}")
            raise
    
    def _load_bboxes(self):
        """Load bounding box annotations."""
        bbox_file = self.anno_dir / 'list_bbox.txt'
        if bbox_file.exists():
            df_bbox = pd.read_csv(
                bbox_file,
                sep=r'\s+',
                skiprows=1,
                names=['image_name', 'x1', 'y1', 'x2', 'y2']
            )
            
            self.bboxes = {}
            for _, row in df_bbox.iterrows():
                self.bboxes[row['image_name']] = {
                    'bbox': [row['x1'], row['y1'], row['x2'], row['y2']]
                }
        else:
            logger.warning(f"Bounding box file not found: {bbox_file}")
            self.bboxes = {}
    
    def _load_landmarks(self):
        """Load fashion landmark annotations."""
        landmark_file = self.anno_dir / 'list_landmarks.txt'
        if landmark_file.exists():
            # Read landmark file with dynamic columns
            with open(landmark_file, 'r') as f:
                lines = f.readlines()
                
            # Parse header to get number of landmarks
            num_landmarks = int(lines[0].strip())
            
            # Create column names
            columns = ['image_name', 'clothes_type']
            for i in range(1, num_landmarks + 1):
                columns.extend([f'x_{i}', f'y_{i}', f'visibility_{i}'])
            
            # Parse data
            self.landmarks = {}
            for line in lines[2:]:  # Skip header lines
                parts = line.strip().split()
                if len(parts) >= 2:
                    image_name = parts[0]
                    clothes_type = int(parts[1])
                    
                    landmarks = []
                    for i in range(num_landmarks):
                        idx = 2 + i * 3
                        if idx + 2 < len(parts):
                            x = float(parts[idx])
                            y = float(parts[idx + 1])
                            vis = int(parts[idx + 2])
                            landmarks.append({'x': x, 'y': y, 'visibility': vis})
                    
                    self.landmarks[image_name] = {
                        'clothes_type': clothes_type,
                        'landmarks': landmarks
                    }
        else:
            logger.warning(f"Landmarks file not found: {landmark_file}")
            self.landmarks = {}
    
    def _load_attributes(self):
        """Load clothing attribute annotations."""
        attr_file = self.anno_dir / 'list_attr_cloth.txt'
        if attr_file.exists():
            # Read attribute file
            with open(attr_file, 'r') as f:
                lines = f.readlines()
            
            # Parse header to get attribute names
            num_attributes = int(lines[0].strip())
            attr_names = []
            for i in range(1, num_attributes + 1):
                if i < len(lines):
                    attr_names.append(lines[i].strip())
            
            # Parse attribute data
            self.attributes = {}
            self.attribute_names = attr_names
            
            for line in lines[num_attributes + 1:]:
                parts = line.strip().split()
                if len(parts) > 1:
                    image_name = parts[0]
                    attrs = [int(x) for x in parts[1:] if x.isdigit() or x.lstrip('-').isdigit()]
                    
                    self.attributes[image_name] = {
                        'values': attrs,
                        'names': attr_names[:len(attrs)]
                    }
        else:
            logger.warning(f"Attributes file not found: {attr_file}")
            self.attributes = {}
            self.attribute_names = []
    
    def _create_samples(self):
        """Create samples list with all annotations."""
        self.samples = []
        
        for image_name in self.image_list:
            sample = {
                'image_path': image_name,
                'image_id': len(self.samples)
            }
            
            # Add bbox if available
            if image_name in self.bboxes:
                sample['bbox'] = self.bboxes[image_name]['bbox']
            
            # Add landmarks if available
            if self.load_landmarks and image_name in self.landmarks:
                sample['landmarks'] = self.landmarks[image_name]
            
            # Add attributes if available
            if self.load_attributes and image_name in self.attributes:
                sample['attributes'] = self.attributes[image_name]
            
            # Filter by categories if specified
            if self.categories:
                # Extract category from image path
                category = image_name.split('/')[1] if '/' in image_name else None
                if category and category in self.categories:
                    self.samples.append(sample)
            else:
                self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} samples for {self.split} split")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image and annotations
        """
        sample = self.samples[idx].copy()
        
        # Load image
        image = self._load_image(sample['image_path'])
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        sample['image'] = image
        
        return sample


class DeepFashion2Dataset(BaseDataset):
    """
    Dataset class for DeepFashion2 dataset.
    
    Supports loading images with instance segmentation masks, bounding boxes,
    keypoints, and category labels from the DeepFashion2 dataset.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = 'train',
        load_masks: bool = True,
        load_keypoints: bool = True,
        categories: Optional[List[int]] = None,
        transform: Optional[transforms.Compose] = None,
        use_s3: bool = False,
        s3_bucket: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize DeepFashion2 dataset.
        
        Args:
            root_dir: Root directory of DeepFashion2 dataset
            split: Dataset split ('train', 'val', 'test')
            load_masks: Whether to load segmentation masks
            load_keypoints: Whether to load keypoints
            categories: List of category IDs to load (None for all)
            transform: Image transformation pipeline
            use_s3: Whether to load data from S3
            s3_bucket: S3 bucket name if use_s3 is True
            cache_dir: Local cache directory for S3 data
        """
        super().__init__(root_dir, split, transform, use_s3, s3_bucket, cache_dir)
        
        self.load_masks = load_masks
        self.load_keypoints = load_keypoints
        self.categories = categories
        
        # DeepFashion2 category mapping
        self.category_map = {
            1: 'short_sleeved_shirt',
            2: 'long_sleeved_shirt',
            3: 'short_sleeved_outwear',
            4: 'long_sleeved_outwear',
            5: 'vest',
            6: 'sling',
            7: 'shorts',
            8: 'trousers',
            9: 'skirt',
            10: 'short_sleeved_dress',
            11: 'long_sleeved_dress',
            12: 'vest_dress',
            13: 'sling_dress'
        }
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Load DeepFashion2 dataset annotations."""
        try:
            # Determine paths based on split
            if self.split == 'train':
                self.image_dir = self.root_dir / 'train' / 'image'
                self.anno_dir = self.root_dir / 'train' / 'annos'
            elif self.split == 'val':
                self.image_dir = self.root_dir / 'validation' / 'image'
                self.anno_dir = self.root_dir / 'validation' / 'annos'
            else:  # test
                self.image_dir = self.root_dir / 'test' / 'image'
                self.anno_dir = None  # Test set doesn't have annotations
            
            # Load annotations
            if self.anno_dir and self.anno_dir.exists():
                self._load_annotations()
            else:
                # For test set, just load image list
                if self.image_dir.exists():
                    self.samples = [
                        {'image_path': img_path.name, 'image_id': idx}
                        for idx, img_path in enumerate(sorted(self.image_dir.glob('*.jpg')))
                    ]
                else:
                    logger.warning(f"Image directory not found: {self.image_dir}")
                    self.samples = []
            
            logger.info(f"Loaded {len(self.samples)} samples for {self.split} split")
            
        except Exception as e:
            logger.error(f"Error loading DeepFashion2 dataset: {e}")
            raise
    
    def _load_annotations(self):
        """Load DeepFashion2 annotations from JSON files."""
        self.samples = []
        
        # Get all annotation files
        anno_files = sorted(self.anno_dir.glob('*.json'))
        
        for anno_file in anno_files:
            try:
                with open(anno_file, 'r') as f:
                    anno_data = json.load(f)
                
                # Get corresponding image name
                image_name = anno_file.stem + '.jpg'
                
                # Create sample
                sample = {
                    'image_path': image_name,
                    'image_id': len(self.samples),
                    'annotations': []
                }
                
                # Process each item annotation
                for item_id, item_data in anno_data.items():
                    if item_id == 'source' or item_id == 'pair_id':
                        continue
                    
                    # Filter by categories if specified
                    category_id = item_data.get('category_id', 0)
                    if self.categories and category_id not in self.categories:
                        continue
                    
                    annotation = {
                        'category_id': category_id,
                        'category_name': self.category_map.get(category_id, 'unknown'),
                        'bbox': item_data.get('bounding_box', []),
                        'scale': item_data.get('scale', 1),
                        'occlusion': item_data.get('occlusion', 0),
                        'zoom_in': item_data.get('zoom_in', 0),
                        'viewpoint': item_data.get('viewpoint', 0)
                    }
                    
                    # Load segmentation mask if requested
                    if self.load_masks and 'segmentation' in item_data:
                        annotation['segmentation'] = item_data['segmentation']
                    
                    # Load keypoints if requested
                    if self.load_keypoints and 'landmarks' in item_data:
                        annotation['keypoints'] = self._parse_keypoints(item_data['landmarks'])
                    
                    sample['annotations'].append(annotation)
                
                # Only add sample if it has annotations (after filtering)
                if sample['annotations']:
                    self.samples.append(sample)
                    
            except Exception as e:
                logger.error(f"Error loading annotation file {anno_file}: {e}")
                continue
    
    def _parse_keypoints(self, landmarks: List) -> List[Dict[str, Any]]:
        """Parse keypoints from landmarks data."""
        keypoints = []
        
        # DeepFashion2 has 294 keypoints (14 keypoints per clothing item)
        for i in range(0, len(landmarks), 3):
            if i + 2 < len(landmarks):
                keypoints.append({
                    'x': landmarks[i],
                    'y': landmarks[i + 1],
                    'visibility': landmarks[i + 2]
                })
        
        return keypoints
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image and annotations
        """
        sample = self.samples[idx].copy()
        
        # Load image
        image_path = os.path.join(
            'train' if self.split == 'train' else 'validation',
            'image',
            sample['image_path']
        )
        image = self._load_image(image_path)
        
        # Store original image size
        sample['original_size'] = image.size
        
        # Apply transformations
        if self.transform:
            # For datasets with bounding boxes/masks, we need to handle transforms carefully
            # This should be done in conjunction with the transforms module
            image = self.transform(image)
        
        sample['image'] = image
        
        # Convert annotations to tensors if needed
        if 'annotations' in sample:
            for anno in sample['annotations']:
                if 'bbox' in anno:
                    anno['bbox'] = torch.tensor(anno['bbox'], dtype=torch.float32)
                if 'keypoints' in anno:
                    # Flatten keypoints to [x1, y1, v1, x2, y2, v2, ...]
                    kpts = []
                    for kpt in anno['keypoints']:
                        kpts.extend([kpt['x'], kpt['y'], kpt['visibility']])
                    anno['keypoints'] = torch.tensor(kpts, dtype=torch.float32)
        
        return sample


def create_dataset(
    dataset_name: str,
    root_dir: Union[str, Path],
    split: str = 'train',
    **kwargs
) -> Union[DeepFashionDataset, DeepFashion2Dataset]:
    """
    Factory function to create dataset instances.
    
    Args:
        dataset_name: Name of the dataset ('deepfashion' or 'deepfashion2')
        root_dir: Root directory of the dataset
        split: Dataset split ('train', 'val', 'test')
        **kwargs: Additional arguments for dataset initialization
        
    Returns:
        Dataset instance
    """
    dataset_map = {
        'deepfashion': DeepFashionDataset,
        'deepfashion2': DeepFashion2Dataset
    }
    
    if dataset_name.lower() not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(dataset_map.keys())}")
    
    dataset_class = dataset_map[dataset_name.lower()]
    return dataset_class(root_dir, split, **kwargs)