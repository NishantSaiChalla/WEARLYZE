#!/usr/bin/env python3
"""
Main training script for Fashion Detection System.

This script provides a unified interface for training all model types
(YOLO, classifiers, CLIP) with support for S3 data loading and distributed training.
"""

import os
import argparse
import logging
from pathlib import Path
import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# Import modules
# Config loading handled directly with yaml
from data.dataset import create_dataset
from data.dataloader import create_train_val_dataloaders
from data.s3_loader import S3DataLoader, S3DatasetWrapper
from data.transforms import get_transforms
from training.trainer import create_trainer, setup_distributed_training
from models.yolo_trainer import create_trainer as create_yolo_trainer
from models.yolo_config import YOLOConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Fashion Detection Models')
    
    # Model selection
    parser.add_argument('--model', type=str, default='yolo',
                        choices=['yolo', 'classifier', 'clip', 'ensemble'],
                        help='Model type to train')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='deepfashion2',
                        choices=['deepfashion', 'deepfashion2'],
                        help='Dataset to use')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory of dataset')
    parser.add_argument('--use-s3', action='store_true',
                        help='Load data from S3')
    parser.add_argument('--s3-bucket', type=str,
                        help='S3 bucket name')
    parser.add_argument('--s3-region', type=str, default='us-east-1',
                        help='S3 region')
    
    # Training arguments
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained model')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to checkpoint to resume from')
    
    # Training options
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--balanced-sampling', action='store_true',
                        help='Use balanced batch sampling')
    
    # Experiment tracking
    parser.add_argument('--experiment-name', type=str, default='fashion_detection',
                        help='Experiment name')
    parser.add_argument('--wandb-project', type=str,
                        help='Weights & Biases project name')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for checkpoints and logs')
    
    return parser.parse_args()


def setup_s3_data_loading(args, dataset):
    """Setup S3 data loading if enabled."""
    if args.use_s3:
        if not args.s3_bucket:
            raise ValueError("--s3-bucket must be specified when using S3")
        
        # Create S3 loader
        s3_loader = S3DataLoader(
            bucket_name=args.s3_bucket,
            region_name=args.s3_region,
            cache_dir=Path(args.output_dir) / 'cache' / args.s3_bucket,
            max_workers=args.workers
        )
        
        # Wrap dataset with S3 loader
        dataset = S3DatasetWrapper(dataset, s3_loader)
        
        logger.info(f"Enabled S3 data loading from bucket: {args.s3_bucket}")
    
    return dataset


def create_datasets(args, config):
    """Create training and validation datasets."""
    # Get transforms
    train_transform = get_transforms(
        mode='train',
        image_size=config['data']['image_size'],
        augmentation_config=config['data']['augmentation']['train']
    )
    
    val_transform = get_transforms(
        mode='val',
        image_size=config['data']['image_size'],
        augmentation_config=config['data']['augmentation']['val']
    )
    
    # Create datasets
    dataset_kwargs = {
        'transform': train_transform,
        'use_s3': args.use_s3,
        's3_bucket': args.s3_bucket if args.use_s3 else None,
        'cache_dir': Path(args.output_dir) / 'cache' if args.use_s3 else None
    }
    
    if args.dataset == 'deepfashion2':
        dataset_kwargs.update({
            'load_masks': args.model == 'yolo',
            'load_keypoints': True,
            'categories': config['fashion']['categories'][:13]  # DeepFashion2 categories
        })
    
    train_dataset = create_dataset(
        args.dataset,
        args.data_root,
        split='train',
        **dataset_kwargs
    )
    
    dataset_kwargs['transform'] = val_transform
    val_dataset = create_dataset(
        args.dataset,
        args.data_root,
        split='val',
        **dataset_kwargs
    )
    
    # Setup S3 loading if needed
    train_dataset = setup_s3_data_loading(args, train_dataset)
    val_dataset = setup_s3_data_loading(args, val_dataset)
    
    logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def train_yolo_model(args, config, train_dataset, val_dataset):
    """Train YOLOv8 model."""
    # Create YOLO config
    yolo_config = YOLOConfig.from_dict(config)
    
    # Override with command line arguments
    yolo_config.training.epochs = args.epochs
    yolo_config.training.batch_size = args.batch_size
    yolo_config.training.learning_rate = args.lr
    yolo_config.training.num_workers = args.workers
    yolo_config.experiment.run_name = args.experiment_name
    yolo_config.experiment.wandb_project = args.wandb_project
    yolo_config.experiment.wandb_log = not args.no_wandb
    
    # Create data loaders
    train_loader, val_loader = create_train_val_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        distributed=args.distributed,
        balanced_sampling=args.balanced_sampling
    )
    
    # Create trainer
    trainer = create_yolo_trainer(
        config=yolo_config,
        train_loader=train_loader,
        val_loader=val_loader,
        model_path=args.checkpoint,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train model
    history = trainer.train()
    
    # Save final model
    final_model_path = Path(args.output_dir) / 'models' / f'{args.experiment_name}_final.pth'
    trainer.save_model(str(final_model_path))
    
    return history


def train_classifier_model(args, config, train_dataset, val_dataset):
    """Train classifier model."""
    from models.classifiers import FashionClassifier
    
    # Create model
    model = FashionClassifier(
        num_classes=len(config['fashion']['categories']),
        model_name=config['model'].get('classifier', {}).get('backbone', 'resnet50'),
        pretrained=args.pretrained
    )
    
    # Create data loaders
    train_loader, val_loader = create_train_val_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        distributed=args.distributed,
        balanced_sampling=args.balanced_sampling
    )
    
    # Update config with command line arguments
    training_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'use_wandb': not args.no_wandb,
        'wandb_project': args.wandb_project,
        'mixed_precision': args.mixed_precision,
        'distributed': args.distributed,
        'checkpoint_dir': str(Path(args.output_dir) / 'checkpoints')
    }
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        **training_config
    )
    
    # Train model
    history = trainer.train()
    
    # Save final model
    final_model_path = Path(args.output_dir) / 'models' / f'{args.experiment_name}_final.pth'
    trainer.save_model(str(final_model_path))
    
    return history


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup distributed training if requested
    if args.distributed:
        setup_distributed_training()
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(args, config)
    
    # Train model based on type
    if args.model == 'yolo':
        history = train_yolo_model(args, config, train_dataset, val_dataset)
    elif args.model == 'classifier':
        history = train_classifier_model(args, config, train_dataset, val_dataset)
    elif args.model == 'clip':
        raise NotImplementedError("CLIP training not yet implemented")
    elif args.model == 'ensemble':
        raise NotImplementedError("Ensemble training not yet implemented")
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Save training history
    history_path = output_dir / 'logs' / f'{args.experiment_name}_history.json'
    import json
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training completed. Results saved to {output_dir}")
    
    # Cleanup distributed training
    if args.distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()