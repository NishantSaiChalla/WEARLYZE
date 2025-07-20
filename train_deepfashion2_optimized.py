#!/usr/bin/env python3
"""
Optimized training script for DeepFashion2 dataset with memory management.
Prevents PC crashes by implementing proper memory cleanup and monitoring.
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
import gc
import psutil
import GPUtil
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


class MemoryMonitor:
    """Monitor system and GPU memory usage."""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        
    def get_memory_stats(self):
        """Get current memory usage statistics."""
        stats = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'ram_used_gb': psutil.virtual_memory().used / (1024**3),
            'ram_available_gb': psutil.virtual_memory().available / (1024**3),
            'ram_percent': psutil.virtual_memory().percent
        }
        
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Assuming single GPU
                    stats.update({
                        'gpu_memory_used_gb': gpu.memoryUsed / 1024,
                        'gpu_memory_total_gb': gpu.memoryTotal / 1024,
                        'gpu_memory_percent': gpu.memoryUtil * 100,
                        'gpu_temperature': gpu.temperature
                    })
                    
                # PyTorch specific GPU memory
                stats.update({
                    'torch_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                    'torch_reserved_gb': torch.cuda.memory_reserved() / (1024**3)
                })
            except Exception as e:
                logger.warning(f"Could not get GPU stats: {e}")
                
        return stats
    
    def log_memory_stats(self, phase=""):
        """Log current memory statistics."""
        stats = self.get_memory_stats()
        logger.info(f"\n{phase} Memory Stats:")
        logger.info(f"CPU: {stats['cpu_percent']:.1f}%")
        logger.info(f"RAM: {stats['ram_used_gb']:.1f}GB used / {stats['ram_percent']:.1f}%")
        
        if self.gpu_available and 'gpu_memory_used_gb' in stats:
            logger.info(f"GPU Memory: {stats['gpu_memory_used_gb']:.1f}GB / {stats['gpu_memory_total_gb']:.1f}GB ({stats['gpu_memory_percent']:.1f}%)")
            logger.info(f"GPU Temp: {stats.get('gpu_temperature', 'N/A')}°C")
            logger.info(f"PyTorch Allocated: {stats['torch_allocated_gb']:.2f}GB")


def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class DeepFashion2Trainer:
    """Optimized trainer class with memory management."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Memory optimization settings
        self._apply_memory_optimizations()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor()
        self.memory_monitor.log_memory_stats("Initial")
        
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
    
    def _apply_memory_optimizations(self):
        """Apply memory optimization settings."""
        # Reduce batch size for safety
        original_batch_size = self.config['training']['batch_size']
        self.config['training']['batch_size'] = min(16, original_batch_size)
        
        # Reduce number of workers to prevent memory buildup
        self.config['training']['num_workers'] = min(4, self.config['training']['num_workers'])
        
        # Enable gradient accumulation if batch size was reduced
        if original_batch_size > self.config['training']['batch_size']:
            self.config['training']['gradient_accumulation_steps'] = original_batch_size // self.config['training']['batch_size']
        else:
            self.config['training']['gradient_accumulation_steps'] = 1
            
        logger.info(f"Memory optimizations applied:")
        logger.info(f"- Batch size: {original_batch_size} -> {self.config['training']['batch_size']}")
        logger.info(f"- Num workers: {self.config['training']['num_workers']}")
        logger.info(f"- Gradient accumulation steps: {self.config['training']['gradient_accumulation_steps']}")
    
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
        
        # Create data loaders with memory-efficient settings
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=False,  # Disable pin_memory to save RAM
            persistent_workers=False,  # Disable persistent workers
            collate_fn=self.collate_fn,
            drop_last=True  # Drop last incomplete batch
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=2,  # Fewer workers for validation
            pin_memory=False,
            persistent_workers=False,
            collate_fn=self.collate_fn,
            drop_last=False
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
        total_steps = len(self.train_loader) * self.config['training']['epochs']
        total_steps = total_steps // self.config['training']['gradient_accumulation_steps']
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config['training']['learning_rate'],
            total_steps=total_steps,
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
        """Custom collate function for DeepFashion2 dataset."""
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
        """Train for one epoch with memory management."""
        self.model.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        # Log memory before epoch
        self.memory_monitor.log_memory_stats(f"Epoch {epoch+1} Start")
        
        accumulation_steps = self.config['training']['gradient_accumulation_steps']
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            try:
                images = images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                # Forward pass with mixed precision
                if self.scaler:
                    with autocast():
                        outputs = self.model(images)
                        loss_dict = self.compute_loss(outputs, targets)
                        loss = loss_dict['total_loss'] / accumulation_steps
                    
                    # Backward pass
                    self.scaler.scale(loss).backward()
                    
                    # Gradient accumulation
                    if (batch_idx + 1) % accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        self.scheduler.step()
                else:
                    outputs = self.model(images)
                    loss_dict = self.compute_loss(outputs, targets)
                    loss = loss_dict['total_loss'] / accumulation_steps
                    
                    loss.backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.scheduler.step()
                
                # Update metrics
                total_loss += loss.item() * accumulation_steps
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * accumulation_steps:.4f}',
                    'lr': f'{current_lr:.6f}',
                    'gpu_mem': f'{torch.cuda.memory_allocated() / (1024**3):.1f}GB'
                })
                
                # Periodic memory cleanup
                if batch_idx % 50 == 0:
                    clear_memory()
                
                # Log to wandb
                if WANDB_AVAILABLE and self.config['experiment'].get('use_wandb', False) and batch_idx % 100 == 0:
                    wandb.log({
                        'train/loss': loss.item() * accumulation_steps,
                        'train/box_loss': loss_dict.get('box_loss', 0).item(),
                        'train/cls_loss': loss_dict.get('cls_loss', 0).item(),
                        'train/seg_loss': loss_dict.get('seg_loss', 0).item(),
                        'train/lr': current_lr,
                        'system/gpu_memory_gb': torch.cuda.memory_allocated() / (1024**3),
                        'system/cpu_percent': psutil.cpu_percent()
                    })
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error("GPU OOM! Clearing cache and skipping batch...")
                    if hasattr(self, 'optimizer'):
                        self.optimizer.zero_grad()
                    clear_memory()
                    continue
                else:
                    raise e
        
        # Clear memory after epoch
        clear_memory()
        self.memory_monitor.log_memory_stats(f"Epoch {epoch+1} End")
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model with memory management."""
        self.model.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        # Clear memory before validation
        clear_memory()
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(self.val_loader, desc='Validation')):
                try:
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
                    
                    # Periodic cleanup
                    if batch_idx % 20 == 0:
                        clear_memory()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error("GPU OOM during validation! Clearing cache...")
                        clear_memory()
                        continue
                    else:
                        raise e
        
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
        
        # Clear memory after validation
        clear_memory()
        
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
        
        # Convert outputs to list of dictionaries
        batch_size = outputs['boxes'].shape[0] if 'boxes' in outputs else 0
        
        for i in range(batch_size):
            pred = {}
            if 'boxes' in outputs:
                pred['boxes'] = outputs['boxes'][i]
            if 'scores' in outputs:
                pred['scores'] = outputs['scores'][i]
            if 'classes' in outputs:
                pred['labels'] = outputs['classes'][i]
            if 'masks' in outputs:
                pred['masks'] = outputs['masks'][i]
            
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
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics_history': self.metrics_history,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = f"checkpoints/epoch_{epoch}.pth"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            torch.save(checkpoint, "checkpoints/best_model.pth")
            
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop with crash prevention."""
        logger.info("Starting optimized training...")
        best_map = 0
        patience_counter = 0
        
        try:
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
                
                self.save_checkpoint(epoch + 1, is_best)
                
                # Early stopping
                if patience_counter >= self.config['training'].get('patience', 10):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
                
                # Force garbage collection after each epoch
                clear_memory()
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint(epoch, is_best=False)
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            self.save_checkpoint(epoch, is_best=False)
            raise e
        
        # Save final results
        self.save_results()
        logger.info("Training completed!")
    
    def save_results(self):
        """Save training results and plots."""
        results_dir = Path("results")
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
        plt.close()
        
        # Save metrics history
        import json
        with open(results_dir / 'metrics_history.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Save configuration
        with open(results_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on DeepFashion2 (Optimized)')
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
        checkpoint = torch.load(args.resume)
        trainer.model.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.metrics_history = checkpoint['metrics_history']
        start_epoch = checkpoint['epoch']
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()