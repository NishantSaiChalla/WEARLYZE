#!/usr/bin/env python3
"""
Simple local training example for Fashion Detection System.
This script demonstrates training without S3 dependencies.
"""

import os
import logging
from pathlib import Path
# import yaml  # Not needed for this test
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DummyFashionDataset(Dataset):
    """Dummy dataset for testing training pipeline."""
    
    def __init__(self, num_samples=100, num_classes=13, image_size=640):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Generate random samples
        self.samples = []
        for i in range(num_samples):
            self.samples.append({
                'image_id': i,
                'category_id': np.random.randint(0, num_classes),
                'bbox': [
                    np.random.randint(0, image_size//2),
                    np.random.randint(0, image_size//2),
                    np.random.randint(image_size//2, image_size),
                    np.random.randint(image_size//2, image_size)
                ]
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Create dummy image
        image = torch.randn(3, self.image_size, self.image_size)
        
        # Create target dict for YOLO
        target = {
            'boxes': torch.tensor([sample['bbox']], dtype=torch.float32),
            'labels': torch.tensor([sample['category_id']], dtype=torch.int64),
            'image_id': torch.tensor([sample['image_id']])
        }
        
        return {
            'images': image,
            'targets': [target],  # List of targets for batch
            'image_id': sample['image_id']
        }


def collate_fn(batch):
    """Custom collate function for detection."""
    images = torch.stack([item['images'] for item in batch])
    targets = [item['targets'][0] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'images': images,
        'targets': targets,
        'image_ids': image_ids
    }


def test_training_pipeline():
    """Test the training pipeline with dummy data."""
    
    # Create dummy datasets
    train_dataset = DummyFashionDataset(num_samples=100)
    val_dataset = DummyFashionDataset(num_samples=20)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Test data loading
    for i, batch in enumerate(train_loader):
        logger.info(f"Batch {i}: images shape = {batch['images'].shape}")
        logger.info(f"Batch {i}: num targets = {len(batch['targets'])}")
        if i >= 2:  # Just test first 3 batches
            break
    
    # Test with unified trainer
    try:
        from training.trainer import UnifiedTrainer, TrainingConfig
        from models.classifiers import FashionResNet
        
        # Create simple classifier model
        model = FashionResNet(
            num_classes=13,
            variant='resnet18',  # Use smaller model for testing
            pretrained=False  # Don't download pretrained weights
        )
        
        # Create training config
        config = TrainingConfig(
            epochs=2,
            batch_size=4,
            learning_rate=0.001,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_wandb=False,
            checkpoint_dir='test_checkpoints',
            log_every=1
        )
        
        # Create trainer
        trainer = UnifiedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        
        logger.info("Starting training...")
        history = trainer.train()
        logger.info(f"Training completed. History: {history}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with YOLO trainer
    try:
        from models.yolo_config import YOLOConfig
        from models.yolo_trainer import YOLOTrainer
        from models.yolo_segmentation import FashionYOLOv8
        
        logger.info("\n\nTesting YOLO trainer...")
        
        # Create YOLO config
        yolo_config = YOLOConfig()
        yolo_config.training.epochs = 2
        yolo_config.training.batch_size = 4
        yolo_config.training.save_period = 10
        yolo_config.experiment.wandb_log = False
        yolo_config.experiment.checkpoint_dir = 'test_yolo_checkpoints'
        
        # Create YOLO model
        yolo_model = FashionYOLOv8(
            config=yolo_config,
            model_path=None,  # Don't load pretrained
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Create YOLO trainer
        yolo_trainer = YOLOTrainer(
            config=yolo_config,
            model=yolo_model,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        logger.info("Starting YOLO training...")
        yolo_history = yolo_trainer.train()
        logger.info(f"YOLO training completed. History keys: {list(yolo_history.keys())}")
        
    except Exception as e:
        logger.error(f"Error during YOLO training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    logger.info("Testing Fashion Detection Training Pipeline")
    test_training_pipeline()
    logger.info("Test completed!")