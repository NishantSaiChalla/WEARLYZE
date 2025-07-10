"""
Custom data loaders for fashion detection with balanced sampling and multi-GPU support.

This module provides efficient data loading utilities with support for
balanced sampling, distributed training, and custom batch collation.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler
import torch.distributed as dist
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class BalancedBatchSampler(Sampler):
    """
    Sampler that ensures balanced representation of classes in each batch.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_classes: int,
        samples_per_class: int = 2,
        iterations: Optional[int] = None
    ):
        """
        Initialize balanced batch sampler.
        
        Args:
            dataset: Dataset to sample from
            batch_size: Total batch size
            num_classes: Number of classes in dataset
            samples_per_class: Number of samples per class in each batch
            iterations: Number of iterations per epoch (None for full dataset)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.iterations = iterations
        
        # Validate batch configuration
        assert batch_size % samples_per_class == 0, \
            "batch_size must be divisible by samples_per_class"
        self.classes_per_batch = batch_size // samples_per_class
        
        # Build class-to-indices mapping
        self.class_indices = defaultdict(list)
        
        # For S3 datasets, we need to handle this differently
        if hasattr(dataset, 'samples'):
            # Use pre-loaded sample metadata instead of accessing dataset items
            for idx, sample in enumerate(dataset.samples):
                if 'label' in sample:
                    label = sample['label']
                elif 'category_id' in sample:
                    label = sample['category_id']
                elif 'annotations' in sample and sample['annotations']:
                    # For detection datasets, use first annotation's category
                    label = sample['annotations'][0]['category_id']
                else:
                    # Skip samples without labels
                    continue
                
                self.class_indices[label].append(idx)
        else:
            # Fallback to direct access for non-S3 datasets
            for idx in range(len(dataset)):
                sample = dataset[idx]
                if 'label' in sample:
                    label = sample['label']
                elif 'category_id' in sample:
                    label = sample['category_id']
                elif 'annotations' in sample and sample['annotations']:
                    # For detection datasets, use first annotation's category
                    label = sample['annotations'][0]['category_id']
                else:
                    raise ValueError("Dataset samples must have 'label' or 'category_id'")
                
                self.class_indices[label].append(idx)
        
        # Calculate number of iterations
        if self.iterations is None:
            self.iterations = len(dataset) // batch_size
    
    def __iter__(self):
        """Generate indices for balanced batches."""
        for _ in range(self.iterations):
            batch_indices = []
            
            # Sample classes for this batch
            if len(self.class_indices) >= self.classes_per_batch:
                selected_classes = np.random.choice(
                    list(self.class_indices.keys()),
                    self.classes_per_batch,
                    replace=False
                )
            else:
                # If we have fewer classes than needed, sample with replacement
                selected_classes = np.random.choice(
                    list(self.class_indices.keys()),
                    self.classes_per_batch,
                    replace=True
                )
            
            # Sample instances from each selected class
            for class_id in selected_classes:
                class_samples = self.class_indices[class_id]
                if len(class_samples) >= self.samples_per_class:
                    selected_samples = np.random.choice(
                        class_samples,
                        self.samples_per_class,
                        replace=False
                    )
                else:
                    # Sample with replacement if not enough samples
                    selected_samples = np.random.choice(
                        class_samples,
                        self.samples_per_class,
                        replace=True
                    )
                batch_indices.extend(selected_samples)
            
            yield batch_indices
    
    def __len__(self):
        """Return number of batches per epoch."""
        return self.iterations


class WeightedSampler:
    """
    Create weighted sampler for imbalanced datasets.
    """
    
    @staticmethod
    def create_sampler(
        dataset: Dataset,
        num_samples: Optional[int] = None,
        replacement: bool = True
    ) -> WeightedRandomSampler:
        """
        Create weighted random sampler based on class frequencies.
        
        Args:
            dataset: Dataset to sample from
            num_samples: Number of samples to draw (None for len(dataset))
            replacement: Whether to sample with replacement
            
        Returns:
            WeightedRandomSampler instance
        """
        # Count class frequencies
        class_counts = defaultdict(int)
        labels = []
        
        # Handle S3 datasets differently
        if hasattr(dataset, 'samples'):
            # Use pre-loaded sample metadata
            for sample in dataset.samples:
                if 'label' in sample:
                    label = sample['label']
                elif 'category_id' in sample:
                    label = sample['category_id']
                elif 'annotations' in sample and sample['annotations']:
                    label = sample['annotations'][0]['category_id']
                else:
                    # Skip samples without labels
                    labels.append(0)  # Default label
                    continue
                
                class_counts[label] += 1
                labels.append(label)
        else:
            # Fallback to direct access for non-S3 datasets
            for idx in range(len(dataset)):
                sample = dataset[idx]
                if 'label' in sample:
                    label = sample['label']
                elif 'category_id' in sample:
                    label = sample['category_id']
                elif 'annotations' in sample and sample['annotations']:
                    label = sample['annotations'][0]['category_id']
                else:
                    raise ValueError("Dataset samples must have 'label' or 'category_id'")
                
                class_counts[label] += 1
                labels.append(label)
        
        # Calculate weights (inverse frequency)
        class_weights = {}
        total_samples = len(dataset)
        num_classes = len(class_counts)
        
        for class_id, count in class_counts.items():
            # Inverse frequency weighting
            class_weights[class_id] = total_samples / (num_classes * count)
        
        # Assign weight to each sample
        sample_weights = [class_weights[label] for label in labels]
        
        # Create sampler
        if num_samples is None:
            num_samples = len(dataset)
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=replacement
        )


def fashion_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for fashion datasets.
    
    Handles variable-sized annotations (bboxes, keypoints, masks) and
    creates padded tensors for batch processing.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched data dictionary
    """
    # Separate different types of data
    images = []
    labels = []
    bboxes = []
    keypoints = []
    masks = []
    image_ids = []
    has_annotations = False
    
    for sample in batch:
        # Images
        if 'image' in sample:
            if isinstance(sample['image'], torch.Tensor):
                images.append(sample['image'])
            else:
                # Convert PIL to tensor if needed
                images.append(torch.from_numpy(np.array(sample['image'])))
        
        # Labels
        if 'label' in sample:
            labels.append(sample['label'])
        elif 'category_id' in sample:
            labels.append(sample['category_id'])
        
        # Image IDs
        if 'image_id' in sample:
            image_ids.append(sample['image_id'])
        
        # Annotations (for detection/segmentation)
        if 'annotations' in sample:
            has_annotations = True
            sample_bboxes = []
            sample_keypoints = []
            sample_labels = []
            
            for anno in sample['annotations']:
                if 'bbox' in anno:
                    sample_bboxes.append(anno['bbox'])
                    sample_labels.append(anno['category_id'])
                if 'keypoints' in anno:
                    sample_keypoints.append(anno['keypoints'])
            
            if sample_bboxes:
                bboxes.append({
                    'boxes': torch.stack(sample_bboxes) if isinstance(sample_bboxes[0], torch.Tensor) 
                             else torch.tensor(sample_bboxes),
                    'labels': torch.tensor(sample_labels)
                })
            
            if sample_keypoints:
                keypoints.append(torch.stack(sample_keypoints))
    
    # Stack data
    collated = {}
    
    if images:
        collated['images'] = torch.stack(images)
    
    if labels:
        collated['labels'] = torch.tensor(labels)
    
    if image_ids:
        collated['image_ids'] = image_ids
    
    if has_annotations and bboxes:
        collated['targets'] = bboxes
    
    if keypoints:
        collated['keypoints'] = keypoints
    
    # Add any other fields that don't need special handling
    for key in batch[0].keys():
        if key not in ['image', 'label', 'category_id', 'annotations', 'image_id']:
            values = [sample[key] for sample in batch]
            # Try to convert to tensor if possible
            try:
                collated[key] = torch.tensor(values)
            except:
                collated[key] = values
    
    return collated


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper for DistributedSampler to work with custom samplers.
    """
    
    def __init__(
        self,
        sampler: Sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0
    ):
        """
        Initialize distributed sampler wrapper.
        
        Args:
            sampler: Base sampler to wrap
            num_replicas: Number of processes (GPUs)
            rank: Rank of current process
            shuffle: Whether to shuffle indices
            seed: Random seed
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
    
    def __iter__(self):
        """Generate indices for distributed sampling."""
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(self.epoch)
        
        indices = list(self.sampler)
        
        # Add extra samples to make it evenly divisible
        num_samples = len(indices)
        total_size = num_samples * self.num_replicas
        indices += indices[:(total_size - num_samples)]
        assert len(indices) == total_size
        
        # Subsample for this rank
        indices = indices[self.rank:total_size:self.num_replicas]
        
        return iter(indices)
    
    def __len__(self):
        """Return number of samples for this rank."""
        return len(self.sampler) // self.num_replicas
    
    def set_epoch(self, epoch: int):
        """Set epoch for shuffling."""
        self.epoch = epoch
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    sampler: Optional[Sampler] = None,
    collate_fn: Optional[Callable] = None,
    distributed: bool = False,
    balanced_sampling: bool = False,
    weighted_sampling: bool = False,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader with appropriate settings for fashion datasets.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data (ignored if sampler is provided)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        drop_last: Whether to drop incomplete last batch
        sampler: Custom sampler (overrides shuffle)
        collate_fn: Custom collate function
        distributed: Whether to use distributed sampler
        balanced_sampling: Whether to use balanced batch sampling
        weighted_sampling: Whether to use weighted sampling
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader instance
    """
    # Determine sampler
    if sampler is None:
        if balanced_sampling:
            # For balanced sampling, we need number of classes
            # This is a simplified version - in practice, you'd get this from dataset
            num_classes = kwargs.pop('num_classes', 10)
            samples_per_class = kwargs.pop('samples_per_class', 2)
            sampler = BalancedBatchSampler(
                dataset, batch_size, num_classes, samples_per_class
            )
            # BalancedBatchSampler returns batches, so we need batch_sampler
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn or fashion_collate_fn,
                **kwargs
            )
        elif weighted_sampling:
            sampler = WeightedSampler.create_sampler(dataset)
        elif distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    
    # Wrap sampler for distributed training if needed
    if distributed and not isinstance(sampler, DistributedSampler):
        sampler = DistributedSamplerWrapper(sampler)
    
    # Use custom collate function if not provided
    if collate_fn is None:
        collate_fn = fashion_collate_fn
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kwargs
    )


class InfiniteDataLoader:
    """
    DataLoader wrapper that cycles through the dataset infinitely.
    Useful for training with a fixed number of iterations.
    """
    
    def __init__(self, dataloader: DataLoader):
        """
        Initialize infinite dataloader.
        
        Args:
            dataloader: Base DataLoader to wrap
        """
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self.sampler = dataloader.sampler
        self.num_workers = dataloader.num_workers
        self.collate_fn = dataloader.collate_fn
        self.pin_memory = dataloader.pin_memory
        self.drop_last = dataloader.drop_last
        self.timeout = dataloader.timeout
        self.worker_init_fn = dataloader.worker_init_fn
        self._iterator = None
    
    def __iter__(self):
        """Create infinite iterator."""
        while True:
            if self._iterator is None:
                self._iterator = iter(self.dataloader)
            try:
                batch = next(self._iterator)
                yield batch
            except StopIteration:
                self._iterator = iter(self.dataloader)
    
    def __len__(self):
        """Return length of underlying dataloader."""
        return len(self.dataloader)


def create_train_val_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    val_batch_size: Optional[int] = None,
    num_workers: int = 4,
    distributed: bool = False,
    balanced_sampling: bool = False,
    **kwargs
) -> tuple:
    """
    Create training and validation dataloaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Training batch size
        val_batch_size: Validation batch size (defaults to batch_size)
        num_workers: Number of workers for data loading
        distributed: Whether to use distributed training
        balanced_sampling: Whether to use balanced sampling for training
        **kwargs: Additional arguments for dataloaders
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    if val_batch_size is None:
        val_batch_size = batch_size
    
    # Training dataloader
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        distributed=distributed,
        balanced_sampling=balanced_sampling,
        **kwargs
    )
    
    # Validation dataloader
    val_loader = create_dataloader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        distributed=distributed,
        balanced_sampling=False,  # No balanced sampling for validation
        **kwargs
    )
    
    return train_loader, val_loader