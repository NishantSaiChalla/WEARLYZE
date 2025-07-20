#!/usr/bin/env python3
"""
Training script for DeepFashion2 dataset with YOLOv8 segmentation.

This script provides comprehensive training for fashion detection and segmentation
with support for multiple instances, colored masks, and evaluation metrics.
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import DeepFashion2Dataset
from data.transforms import get_augmentation_pipeline
from models.yolo_segmentation import FashionYOLOv8
from models.yolo_config import YOLOConfig
from utils.visualization import visualize_predictions_with_masks
from utils.metrics import calculate_map, calculate_segmentation_metrics
from training.experiment_manager import ExperimentManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepFashion2Trainer:
    """Trainer class for DeepFashion2 dataset."""
    
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize experiment manager (disabled for now)
        # self.exp_manager = ExperimentManager(
        #     experiment_name=self.config['experiment']['name'],
        #     output_dir=self.config['experiment']['output_dir']
        # )
        
        # Initialize datasets
        self._init_datasets()
        
        # Initialize model
        self._init_model()
        
        # Initialize training components
        self._init_training()
        
        # Initialize metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_map50': [],
            'val_map50_95': [],
            'val_miou': [],
            'val_dice': []
        }
    
    def _init_datasets(self):
        """Initialize training and validation datasets."""
        # Get augmentation pipelines
        train_transform = get_augmentation_pipeline(
            'train',
            image_size=self.config['model']['input_size'],
            augmentation_config=self.config['augmentation']
        )
        
        val_transform = get_augmentation_pipeline(
            'val',
            image_size=self.config['model']['input_size']
        )
        
        # Create datasets
        self.train_dataset = DeepFashion2Dataset(
            root_dir=self.config['dataset']['root_dir'],
            split='train',
            load_masks=True,
            load_keypoints=self.config['dataset'].get('load_keypoints', True),
            categories=self.config['dataset'].get('categories', None),
            transform=train_transform
        )
        
        self.val_dataset = DeepFashion2Dataset(
            root_dir=self.config['dataset']['root_dir'],
            split='validation',
            load_masks=True,
            load_keypoints=self.config['dataset'].get('load_keypoints', True),
            categories=self.config['dataset'].get('categories', None),
            transform=val_transform
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
    
    def _init_model(self):
        """Initialize YOLOv8 model for segmentation."""
        # Create YOLO config
        yolo_config = YOLOConfig()
        yolo_config.model.model_size = self.config['model']['model_size']
        yolo_config.model.num_classes = 13  # DeepFashion2 has 13 categories
        yolo_config.model.input_size = self.config['model']['input_size']
        
        # Update training parameters
        yolo_config.training.box_loss_gain = self.config['loss']['box_weight']
        yolo_config.training.cls_loss_gain = self.config['loss']['cls_weight']
        yolo_config.training.seg_loss_gain = self.config['loss']['seg_weight']
        
        # Initialize model
        self.model = FashionYOLOv8(
            config=yolo_config,
            model_path=self.config['model'].get('pretrained_path', None),
            device=self.device
        )
        
        logger.info(f"Initialized {yolo_config.model.model_size} model")
    
    def _init_training(self):
        """Initialize training components."""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config['training']['learning_rate'],
            epochs=self.config['training']['epochs'],
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if self.config['training'].get('mixed_precision', True) else None
        
        # Initialize wandb if enabled
        if WANDB_AVAILABLE and self.config['experiment'].get('use_wandb', False):
            wandb.init(
                project=self.config['experiment']['wandb_project'],
                name=self.config['experiment']['name'],
                config=self.config
            )
    
    def collate_fn(self, batch):
        """
        Custom collate function for DeepFashion2 dataset.
        
        Handles variable number of instances per image.
        """
        images = []
        targets = []
        
        for sample in batch:
            images.append(sample['image'])
            
            # Convert annotations to target format
            target = {
                'boxes': [],
                'labels': [],
                'masks': [],
                'keypoints': []
            }
            
            for anno in sample.get('annotations', []):
                if 'bbox' in anno:
                    target['boxes'].append(anno['bbox'])
                    target['labels'].append(anno['category_id'])
                
                if 'segmentation' in anno:
                    # Convert polygon to mask
                    mask = self._polygon_to_mask(
                        anno['segmentation'],
                        sample['original_size']
                    )
                    target['masks'].append(mask)
                
                if 'keypoints' in anno:
                    target['keypoints'].append(anno['keypoints'])
            
            # Convert to tensors
            if target['boxes']:
                target['boxes'] = torch.stack(target['boxes'])
                target['labels'] = torch.tensor(target['labels'], dtype=torch.long)
            
            if target['masks']:
                target['masks'] = torch.stack(target['masks'])
            
            targets.append(target)
        
        # Stack images
        images = torch.stack(images)
        
        return images, targets
    
    def _polygon_to_mask(self, polygons, image_size):
        """Convert polygon annotations to binary mask."""
        import cv2
        
        h, w = image_size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for polygon in polygons:
            if len(polygon) > 0:
                pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
        
        return torch.from_numpy(mask).float()
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss_dict = self.compute_loss(outputs, targets)
                    loss = loss_dict['total_loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss_dict = self.compute_loss(outputs, targets)
                loss = loss_dict['total_loss']
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })
            
            # Log to wandb
            if WANDB_AVAILABLE and self.config['experiment'].get('use_wandb', False) and batch_idx % 100 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/box_loss': loss_dict.get('box_loss', 0).item(),
                    'train/cls_loss': loss_dict.get('cls_loss', 0).item(),
                    'train/seg_loss': loss_dict.get('seg_loss', 0).item(),
                    'train/lr': current_lr
                })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                # Forward pass
                outputs = self.model(images)
                loss_dict = self.compute_loss(outputs, targets)
                total_loss += loss_dict['total_loss'].item()
                
                # Collect predictions
                all_predictions.extend(self.process_outputs(outputs))
                all_targets.extend(targets)
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        
        # Detection metrics
        map_metrics = calculate_map(all_predictions, all_targets)
        
        # Segmentation metrics
        seg_metrics = calculate_segmentation_metrics(all_predictions, all_targets)
        
        # Log metrics
        metrics = {
            'val_loss': avg_loss,
            'val_map50': map_metrics['map50'],
            'val_map50_95': map_metrics['map50_95'],
            'val_miou': seg_metrics['miou'],
            'val_dice': seg_metrics['dice']
        }
        
        # Log to wandb
        if self.config['experiment'].get('use_wandb', False):
            if WANDB_AVAILABLE:
                wandb.log(metrics)
        
        # Save visualizations
        if epoch % self.config['experiment'].get('viz_interval', 5) == 0:
            self.visualize_predictions(images[:4], outputs[:4], epoch)
        
        return metrics
    
    def compute_loss(self, outputs, targets):
        """Compute loss for batch."""
        # Use model's loss function
        predictions = {
            'boxes': outputs.get('boxes', None),
            'classes': outputs.get('classes', None),
            'masks': outputs.get('masks', None)
        }
        
        # Prepare targets in correct format
        target_dict = {
            'boxes': torch.cat([t['boxes'] for t in targets if 'boxes' in t]),
            'classes': torch.cat([t['labels'] for t in targets if 'labels' in t]),
            'masks': torch.cat([t['masks'] for t in targets if 'masks' in t])
        }
        
        return self.model.loss_fn(predictions, target_dict)
    
    def process_outputs(self, outputs):
        """Process model outputs for metric calculation."""
        processed = []
        
        # Handle different output formats from YOLOv8
        if isinstance(outputs, list):
            # outputs is already a list of predictions per image
            for output in outputs:
                pred = {}
                if isinstance(output, dict):
                    # Direct dictionary format
                    if 'boxes' in output:
                        pred['boxes'] = output['boxes'].detach().cpu() if isinstance(output['boxes'], torch.Tensor) else output['boxes']
                    if 'scores' in output:
                        pred['scores'] = output['scores'].detach().cpu() if isinstance(output['scores'], torch.Tensor) else output['scores']
                    if 'classes' in output:
                        pred['labels'] = output['classes'].detach().cpu() if isinstance(output['classes'], torch.Tensor) else output['classes']
                    elif 'labels' in output:
                        pred['labels'] = output['labels'].detach().cpu() if isinstance(output['labels'], torch.Tensor) else output['labels']
                    if 'masks' in output:
                        pred['masks'] = output['masks'].detach().cpu() if isinstance(output['masks'], torch.Tensor) else output['masks']
                elif hasattr(output, 'boxes'):
                    # Object with attributes
                    if hasattr(output, 'boxes') and output.boxes is not None:
                        pred['boxes'] = output.boxes.xyxy.detach().cpu()
                        pred['scores'] = output.boxes.conf.detach().cpu()
                        pred['labels'] = output.boxes.cls.detach().cpu()
                    if hasattr(output, 'masks') and output.masks is not None:
                        pred['masks'] = output.masks.data.detach().cpu()
                processed.append(pred)
        elif isinstance(outputs, dict):
            # Single batch dictionary format
            batch_size = outputs['boxes'].shape[0] if 'boxes' in outputs else 0
            
            for i in range(batch_size):
                pred = {}
                if 'boxes' in outputs:
                    pred['boxes'] = outputs['boxes'][i].detach().cpu()
                if 'scores' in outputs:
                    pred['scores'] = outputs['scores'][i].detach().cpu()
                if 'classes' in outputs:
                    pred['labels'] = outputs['classes'][i].detach().cpu()
                elif 'labels' in outputs:
                    pred['labels'] = outputs['labels'][i].detach().cpu()
                if 'masks' in outputs:
                    pred['masks'] = outputs['masks'][i].detach().cpu()
                
                processed.append(pred)
        else:
            # Handle YOLO results format
            if hasattr(outputs, '__iter__'):
                for output in outputs:
                    pred = {}
                    if hasattr(output, 'boxes') and output.boxes is not None:
                        pred['boxes'] = output.boxes.xyxy.detach().cpu()
                        pred['scores'] = output.boxes.conf.detach().cpu()
                        pred['labels'] = output.boxes.cls.detach().cpu()
                    if hasattr(output, 'masks') and output.masks is not None:
                        pred['masks'] = output.masks.data.detach().cpu()
                    processed.append(pred)
        
        return processed
    
    def visualize_predictions(self, images, outputs, epoch):
        """Visualize predictions with colored masks."""
        save_dir = Path(self.exp_manager.output_dir) / 'visualizations'
        save_dir.mkdir(exist_ok=True)
        
        # Visualize batch
        vis_path = save_dir / f'epoch_{epoch}_predictions.png'
        visualize_predictions_with_masks(
            images.cpu(),
            outputs,
            save_path=vis_path,
            class_names=self.model.config.training.fashion_categories
        )
        
        # Log to wandb
        if WANDB_AVAILABLE and self.config['experiment'].get('use_wandb', False):
            wandb.log({
                'val/predictions': wandb.Image(str(vis_path))
            })
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        best_map = 0
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            self.metrics_history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self.validate(epoch)
            for key, value in val_metrics.items():
                if key in self.metrics_history:
                    self.metrics_history[key].append(value)
            
            # Log epoch summary
            logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}, "
                f"mAP@50: {val_metrics['val_map50']:.4f}, mIoU: {val_metrics['val_miou']:.4f}"
            )
            
            # Save checkpoint
            is_best = val_metrics['val_map50'] > best_map
            if is_best:
                best_map = val_metrics['val_map50']
                patience_counter = 0
            else:
                patience_counter += 1
            
            checkpoint_path = os.path.join(
                self.exp_manager.checkpoints_dir,
                f'epoch_{epoch+1}.pth'
            )
            
            self.model.save_checkpoint(
                checkpoint_path,
                epoch + 1,
                self.optimizer,
                self.scheduler,
                is_best
            )
            
            # Early stopping
            if patience_counter >= self.config['training'].get('patience', 10):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save final results
        self.save_results()
        logger.info("Training completed!")
    
    def save_results(self):
        """Save training results and plots."""
        results_dir = Path(self.exp_manager.output_dir) / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(self.metrics_history['train_loss'], label='Train')
        axes[0, 0].plot(self.metrics_history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        
        # mAP curves
        axes[0, 1].plot(self.metrics_history['val_map50'], label='mAP@50')
        axes[0, 1].plot(self.metrics_history['val_map50_95'], label='mAP@50:95')
        axes[0, 1].set_title('Detection mAP')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        
        # Segmentation metrics
        axes[1, 0].plot(self.metrics_history['val_miou'], label='mIoU')
        axes[1, 0].set_title('Mean IoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        
        axes[1, 1].plot(self.metrics_history['val_dice'], label='Dice')
        axes[1, 1].set_title('Dice Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(results_dir / 'training_curves.png')
        
        # Save metrics history
        import json
        with open(results_dir / 'metrics_history.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Save configuration
        with open(results_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on DeepFashion2')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/deepfashion2_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = DeepFashion2Trainer(args.config)
    
    # Resume if checkpoint provided
    if args.resume:
        epoch = trainer.model.load_checkpoint(
            args.resume,
            trainer.optimizer,
            trainer.scheduler
        )
        logger.info(f"Resumed from epoch {epoch}")
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()