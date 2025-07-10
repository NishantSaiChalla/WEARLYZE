"""
Image transformation and augmentation utilities for fashion detection.

This module provides various image preprocessing and augmentation strategies
including standard transforms, fashion-specific augmentations, and CLIP preprocessing.
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
import cv2

try:
    from torchvision.transforms import RandAugment as _RandAugment
except ImportError:
    _RandAugment = None


class FashionResize:
    """
    Resize transform that maintains aspect ratio and handles bounding boxes.
    """
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        max_size: Optional[int] = None,
        interpolation: int = Image.BILINEAR
    ):
        """
        Initialize resize transform.
        
        Args:
            size: Target size (height, width) or single int for short edge
            max_size: Maximum size for the longer edge
            interpolation: PIL interpolation method
        """
        self.size = size if isinstance(size, tuple) else (size, size)
        self.max_size = max_size
        self.interpolation = interpolation
    
    def __call__(self, img: Image.Image, target: Optional[Dict] = None) -> Union[Image.Image, Tuple]:
        """
        Apply resize transform.
        
        Args:
            img: PIL Image
            target: Optional target dict with bboxes, masks, etc.
            
        Returns:
            Resized image and optionally resized target
        """
        h, w = img.height, img.width
        
        if isinstance(self.size, int):
            # Resize shorter edge to size
            if h < w:
                new_h = self.size
                new_w = int(w * self.size / h)
            else:
                new_w = self.size
                new_h = int(h * self.size / w)
            
            # Check max_size constraint
            if self.max_size is not None:
                if new_h > self.max_size:
                    ratio = self.max_size / new_h
                    new_h = self.max_size
                    new_w = int(new_w * ratio)
                elif new_w > self.max_size:
                    ratio = self.max_size / new_w
                    new_w = self.max_size
                    new_h = int(new_h * ratio)
        else:
            new_h, new_w = self.size
        
        # Resize image
        img = img.resize((new_w, new_h), self.interpolation)
        
        # Resize target if provided
        if target is not None:
            ratio_h = new_h / h
            ratio_w = new_w / w
            
            # Resize bounding boxes
            if 'bbox' in target:
                bbox = target['bbox']
                target['bbox'] = [
                    bbox[0] * ratio_w,
                    bbox[1] * ratio_h,
                    bbox[2] * ratio_w,
                    bbox[3] * ratio_h
                ]
            
            # Resize keypoints
            if 'keypoints' in target:
                keypoints = target['keypoints']
                scaled_keypoints = []
                for kpt in keypoints:
                    scaled_keypoints.append({
                        'x': kpt['x'] * ratio_w,
                        'y': kpt['y'] * ratio_h,
                        'visibility': kpt['visibility']
                    })
                target['keypoints'] = scaled_keypoints
            
            # Resize segmentation masks
            if 'segmentation' in target:
                # This would require more complex handling with cv2
                pass
            
            return img, target
        
        return img


class RandomHorizontalFlip:
    """
    Random horizontal flip that handles bounding boxes and keypoints.
    """
    
    def __init__(self, p: float = 0.5):
        """
        Initialize random flip.
        
        Args:
            p: Probability of flipping
        """
        self.p = p
    
    def __call__(self, img: Image.Image, target: Optional[Dict] = None) -> Union[Image.Image, Tuple]:
        """
        Apply random horizontal flip.
        
        Args:
            img: PIL Image
            target: Optional target dict with bboxes, keypoints, etc.
            
        Returns:
            Flipped image and optionally flipped target
        """
        if random.random() < self.p:
            img = F.hflip(img)
            
            if target is not None:
                w = img.width
                
                # Flip bounding boxes
                if 'bbox' in target:
                    bbox = target['bbox']
                    target['bbox'] = [
                        w - bbox[2],
                        bbox[1],
                        w - bbox[0],
                        bbox[3]
                    ]
                
                # Flip keypoints
                if 'keypoints' in target:
                    keypoints = target['keypoints']
                    flipped_keypoints = []
                    for kpt in keypoints:
                        flipped_keypoints.append({
                            'x': w - kpt['x'],
                            'y': kpt['y'],
                            'visibility': kpt['visibility']
                        })
                    target['keypoints'] = flipped_keypoints
                
                return img, target
        
        return img if target is None else (img, target)


class FashionColorJitter:
    """
    Color jittering specifically tuned for fashion images.
    """
    
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1
    ):
        """
        Initialize color jitter.
        
        Args:
            brightness: Brightness jitter range
            contrast: Contrast jitter range
            saturation: Saturation jitter range
            hue: Hue jitter range
        """
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply color jittering."""
        return self.jitter(img)


class CutMix:
    """
    CutMix augmentation for fashion images.
    
    Reference: https://arxiv.org/abs/1905.04899
    """
    
    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        """
        Initialize CutMix.
        
        Args:
            alpha: Beta distribution parameter
            p: Probability of applying CutMix
        """
        self.alpha = alpha
        self.p = p
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix augmentation.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B, num_classes)
            
        Returns:
            Mixed images, mixed labels, indices used for mixing, lambda value
        """
        if random.random() > self.p:
            return images, labels, None, 1.0
        
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get image dimensions
        _, _, h, w = images.shape
        
        # Sample bounding box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply CutMix
        images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        # Mix labels
        if labels.dim() == 1:
            # Convert to one-hot if needed
            num_classes = labels.max() + 1
            labels_onehot = torch.zeros(batch_size, num_classes)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels = labels_onehot
        
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        return images, mixed_labels, indices, lam


class MixUp:
    """
    MixUp augmentation for fashion images.
    
    Reference: https://arxiv.org/abs/1710.09412
    """
    
    def __init__(self, alpha: float = 0.2, p: float = 0.5):
        """
        Initialize MixUp.
        
        Args:
            alpha: Beta distribution parameter
            p: Probability of applying MixUp
        """
        self.alpha = alpha
        self.p = p
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp augmentation.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels
            
        Returns:
            Mixed images, mixed labels, indices used for mixing, lambda value
        """
        if random.random() > self.p:
            return images, labels, None, 1.0
        
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # Mix labels
        if labels.dim() == 1:
            # Convert to one-hot if needed
            num_classes = labels.max() + 1
            labels_onehot = torch.zeros(batch_size, num_classes)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels = labels_onehot
        
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
        return mixed_images, mixed_labels, indices, lam


class RandAugment:
    """
    RandAugment for fashion images with custom policies.
    
    Reference: https://arxiv.org/abs/1909.13719
    """
    
    def __init__(self, n: int = 2, m: int = 9):
        """
        Initialize RandAugment.
        
        Args:
            n: Number of augmentation transformations to apply
            m: Magnitude of transformations (0-30)
        """
        if _RandAugment is not None:
            self.augment = _RandAugment(num_ops=n, magnitude=m)
        else:
            # Fallback implementation
            self.n = n
            self.m = m
            self.augmentations = self._create_augmentations()
    
    def _create_augmentations(self) -> List:
        """Create augmentation pool for fallback implementation."""
        # Define augmentation policies
        return [
            lambda img, m: F.rotate(img, m * 3),
            lambda img, m: F.adjust_brightness(img, 1 + m * 0.05),
            lambda img, m: F.adjust_contrast(img, 1 + m * 0.05),
            lambda img, m: F.adjust_saturation(img, 1 + m * 0.05),
            lambda img, m: img.transform(img.size, Image.AFFINE, 
                                       (1, m * 0.02, 0, 0, 1, 0)),
            lambda img, m: img.transform(img.size, Image.AFFINE,
                                       (1, 0, 0, m * 0.02, 1, 0)),
        ]
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply RandAugment."""
        if hasattr(self, 'augment'):
            return self.augment(img)
        else:
            # Fallback implementation
            ops = random.choices(self.augmentations, k=self.n)
            for op in ops:
                img = op(img, self.m)
            return img


class CLIPProcessor:
    """
    CLIP-specific preprocessing pipeline.
    """
    
    def __init__(
        self,
        size: int = 224,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None
    ):
        """
        Initialize CLIP processor.
        
        Args:
            size: Input size for CLIP model
            mean: Normalization mean (default CLIP values)
            std: Normalization std (default CLIP values)
        """
        self.size = size
        self.mean = mean or [0.48145466, 0.4578275, 0.40821073]
        self.std = std or [0.26862954, 0.26130258, 0.27577711]
        
        self.transform = T.Compose([
            T.Resize(size, interpolation=Image.BICUBIC),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ])
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Apply CLIP preprocessing."""
        return self.transform(img)


class SegmentationTransform:
    """
    Transform pipeline for segmentation tasks.
    """
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        crop_size: Optional[Union[int, Tuple[int, int]]] = None,
        scale: Tuple[float, float] = (0.5, 2.0),
        ratio: Tuple[float, float] = (0.5, 2.0),
        augment: bool = True
    ):
        """
        Initialize segmentation transform.
        
        Args:
            size: Base size for resizing
            crop_size: Random crop size (None to skip cropping)
            scale: Scale range for random resized crop
            ratio: Aspect ratio range for random resized crop
            augment: Whether to apply augmentations
        """
        self.size = size if isinstance(size, tuple) else (size, size)
        self.crop_size = crop_size
        self.scale = scale
        self.ratio = ratio
        self.augment = augment
    
    def __call__(
        self,
        img: Image.Image,
        mask: Optional[Image.Image] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply segmentation transforms.
        
        Args:
            img: Input image
            mask: Segmentation mask
            
        Returns:
            Transformed image and optionally mask
        """
        # Random resized crop
        if self.crop_size and self.augment:
            i, j, h, w = T.RandomResizedCrop.get_params(
                img, scale=self.scale, ratio=self.ratio
            )
            img = F.resized_crop(img, i, j, h, w, self.crop_size, Image.BILINEAR)
            if mask is not None:
                mask = F.resized_crop(mask, i, j, h, w, self.crop_size, Image.NEAREST)
        else:
            img = F.resize(img, self.size, Image.BILINEAR)
            if mask is not None:
                mask = F.resize(mask, self.size, Image.NEAREST)
        
        # Random horizontal flip
        if self.augment and random.random() > 0.5:
            img = F.hflip(img)
            if mask is not None:
                mask = F.hflip(mask)
        
        # Color augmentation (only for image)
        if self.augment:
            img = FashionColorJitter()(img)
        
        # Convert to tensor
        img = F.to_tensor(img)
        if mask is not None:
            mask = torch.from_numpy(np.array(mask)).long()
            return img, mask
        
        return img


def create_transform_pipeline(
    mode: str = 'train',
    size: Union[int, Tuple[int, int]] = 224,
    normalize: bool = True,
    augment_level: str = 'medium',
    task: str = 'classification'
) -> T.Compose:
    """
    Create a transform pipeline for fashion images.
    
    Args:
        mode: 'train', 'val', or 'test'
        size: Target image size
        normalize: Whether to apply normalization
        augment_level: 'none', 'light', 'medium', or 'heavy'
        task: 'classification', 'detection', or 'segmentation'
        
    Returns:
        Composed transform pipeline
    """
    transforms = []
    
    # Base transforms
    if task == 'classification':
        if mode == 'train':
            transforms.append(T.RandomResizedCrop(size, scale=(0.7, 1.0)))
            transforms.append(T.RandomHorizontalFlip())
        else:
            transforms.append(T.Resize(int(size * 1.14)))
            transforms.append(T.CenterCrop(size))
    elif task in ['detection', 'segmentation']:
        transforms.append(FashionResize(size))
        if mode == 'train':
            transforms.append(RandomHorizontalFlip())
    
    # Augmentation based on level
    if mode == 'train' and augment_level != 'none':
        if augment_level == 'light':
            transforms.append(FashionColorJitter(0.1, 0.1, 0.1, 0.05))
        elif augment_level == 'medium':
            transforms.append(FashionColorJitter(0.2, 0.2, 0.2, 0.1))
            if task == 'classification':
                transforms.append(T.RandomRotation(10))
        elif augment_level == 'heavy':
            transforms.append(RandAugment(n=3, m=12))
            transforms.append(FashionColorJitter(0.3, 0.3, 0.3, 0.15))
    
    # Convert to tensor
    transforms.append(T.ToTensor())
    
    # Normalize
    if normalize:
        # ImageNet statistics
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transforms.append(T.Normalize(mean=mean, std=std))
    
    return T.Compose(transforms)


def get_clip_transform() -> CLIPProcessor:
    """Get CLIP-specific transform pipeline."""
    return CLIPProcessor()


def get_segmentation_transform(
    mode: str = 'train',
    size: Union[int, Tuple[int, int]] = 512
) -> SegmentationTransform:
    """Get segmentation-specific transform pipeline."""
    augment = mode == 'train'
    crop_size = size if augment else None
    return SegmentationTransform(size, crop_size, augment=augment)


def get_transforms(
    mode: str = 'train',
    image_size: Union[int, Tuple[int, int]] = 640,
    augmentation_config: Optional[Dict[str, Any]] = None
) -> T.Compose:
    """
    Get transform pipeline based on mode and configuration.
    
    Args:
        mode: 'train' or 'val'
        image_size: Target image size
        augmentation_config: Augmentation configuration dict
        
    Returns:
        Transform pipeline
    """
    transforms = []
    
    # Resize
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    transforms.append(T.Resize(image_size))
    
    # Augmentations for training
    if mode == 'train' and augmentation_config:
        if augmentation_config.get('horizontal_flip', 0) > 0:
            transforms.append(T.RandomHorizontalFlip(p=augmentation_config['horizontal_flip']))
        if augmentation_config.get('rotation', 0) > 0:
            transforms.append(T.RandomRotation(degrees=augmentation_config['rotation']))
        if augmentation_config.get('brightness', 0) > 0:
            brightness = augmentation_config['brightness']
            contrast = augmentation_config.get('contrast', 0)
            saturation = augmentation_config.get('saturation', 0)
            hue = augmentation_config.get('hue', 0)
            transforms.append(T.ColorJitter(brightness=brightness, contrast=contrast, 
                                          saturation=saturation, hue=hue))
    
    # Convert to tensor
    transforms.append(T.ToTensor())
    
    # Normalize with ImageNet stats
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]))
    
    return T.Compose(transforms)