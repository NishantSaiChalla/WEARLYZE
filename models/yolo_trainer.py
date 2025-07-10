"""
YOLOv8 Trainer for Fashion Detection and Segmentation.

This module provides a comprehensive training pipeline for YOLOv8 models on fashion datasets
with support for distributed training, validation, early stopping, and experiment tracking.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from .yolo_config import YOLOConfig
from .yolo_segmentation import FashionYOLOv8
from .yolo_utils import YOLOVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility to stop training when validation metric stops improving."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max',
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if early stopping condition is met.
        
        Args:
            score: Current validation score
            model: Model to save weights from
            
        Returns:
            True if should stop training
        """
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    logger.info("Restored best weights")
        
        return self.early_stop


class MetricsTracker:
    """Utility class to track and log training metrics."""
    
    def __init__(self, log_dir: str, use_tensorboard: bool = True, use_wandb: bool = False):
        """
        Initialize metrics tracker.
        
        Args:
            log_dir: Directory to save logs
            use_tensorboard: Whether to use TensorBoard logging
            use_wandb: Whether to use Weights & Biases logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        # Initialize loggers
        self.tb_writer = None
        if use_tensorboard:
            self.tb_writer = SummaryWriter(str(self.log_dir))
        
        # Storage for metrics
        self.metrics_history = defaultdict(list)
        self.current_epoch = 0
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = '') -> None:
        """
        Log metrics to all configured loggers.
        
        Args:
            metrics: Dictionary of metric values
            step: Current step/epoch
            prefix: Prefix for metric names
        """
        # Store in history
        for name, value in metrics.items():
            full_name = f"{prefix}{name}" if prefix else name
            self.metrics_history[full_name].append(value)
        
        # Log to TensorBoard
        if self.tb_writer is not None:
            for name, value in metrics.items():
                full_name = f"{prefix}{name}" if prefix else name
                self.tb_writer.add_scalar(full_name, value, step)
        
        # Log to Weights & Biases
        if self.use_wandb:
            wandb_metrics = {}
            for name, value in metrics.items():
                full_name = f"{prefix}{name}" if prefix else name
                wandb_metrics[full_name] = value
            wandb.log(wandb_metrics, step=step)
    
    def log_images(self, images: Dict[str, Any], step: int) -> None:
        """
        Log images to loggers.
        
        Args:
            images: Dictionary of images
            step: Current step/epoch
        """
        # Log to TensorBoard
        if self.tb_writer is not None:
            for name, image in images.items():
                if isinstance(image, torch.Tensor):
                    self.tb_writer.add_image(name, image, step)
                elif isinstance(image, np.ndarray):
                    self.tb_writer.add_image(name, image, step, dataformats='HWC')
        
        # Log to Weights & Biases
        if self.use_wandb:
            wandb_images = {}
            for name, image in images.items():
                if isinstance(image, (torch.Tensor, np.ndarray)):
                    wandb_images[name] = wandb.Image(image)
            wandb.log(wandb_images, step=step)
    
    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Plot training metrics.
        
        Args:
            save_path: Path to save plot
        """
        if not self.metrics_history:
            return
        
        # Create subplots
        metrics_to_plot = ['train_loss', 'val_loss', 'val_map', 'val_miou']
        available_metrics = [m for m in metrics_to_plot if m in self.metrics_history]
        
        if not available_metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics):
            if i < len(axes):
                values = self.metrics_history[metric]
                axes[i].plot(values)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel('Value')
                axes[i].grid(True)
        
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.log_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def save_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Save metrics history to JSON file.
        
        Args:
            save_path: Path to save metrics
        """
        if save_path is None:
            save_path = self.log_dir / 'metrics_history.json'
        
        with open(save_path, 'w') as f:
            json.dump(dict(self.metrics_history), f, indent=2)
    
    def close(self) -> None:
        """Close all loggers."""
        if self.tb_writer is not None:
            self.tb_writer.close()


class YOLOTrainer:
    """
    Comprehensive trainer for YOLOv8 fashion detection and segmentation.
    
    This class provides a complete training pipeline with support for distributed training,
    validation, early stopping, learning rate scheduling, and experiment tracking.
    """
    
    def __init__(
        self,
        config: YOLOConfig,
        model: FashionYOLOv8,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize YOLO trainer.
        
        Args:
            config: Training configuration
            model: Fashion YOLOv8 model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            callbacks: List of callback functions
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.callbacks = callbacks or []
        
        # Setup distributed training
        self.is_distributed = self._setup_distributed()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        
        # Wrap model for distributed training
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.rank])
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            mode='max',  # For mAP
            restore_best_weights=True
        )
        
        # Setup metrics tracking
        self.metrics_tracker = MetricsTracker(
            log_dir=config.experiment.log_dir,
            use_tensorboard=config.experiment.tensorboard_log,
            use_wandb=config.experiment.wandb_log
        )
        
        # Initialize Weights & Biases if enabled
        if config.experiment.wandb_log and self.rank == 0:
            wandb.init(
                project=config.experiment.wandb_project,
                entity=config.experiment.wandb_entity,
                name=config.experiment.run_name,
                config=config.to_dict(),
                tags=config.experiment.tags,
                notes=config.experiment.notes
            )
        
        # Setup visualizer
        self.visualizer = YOLOVisualizer(config.training.fashion_categories)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.training_history = defaultdict(list)
        
        # Create output directories
        self.checkpoint_dir = Path(config.experiment.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized with {self.world_size} processes")
    
    def _setup_distributed(self) -> bool:
        """Setup distributed training if available."""
        if 'WORLD_SIZE' in os.environ:
            world_size = int(os.environ['WORLD_SIZE'])
            rank = int(os.environ['RANK'])
            
            # Initialize distributed training
            dist.init_process_group(
                backend='nccl',
                world_size=world_size,
                rank=rank
            )
            
            # Set device
            torch.cuda.set_device(rank)
            
            return True
        
        return False
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        if hasattr(self.model, 'module'):
            model_params = self.model.module.parameters()
        else:
            model_params = self.model.parameters()
        
        optimizer = torch.optim.AdamW(
            model_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        return optimizer
    
    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.epochs,
            eta_min=self.config.training.learning_rate * 0.01
        )
        
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        if hasattr(self.model, 'module'):
            self.model.module.train()
        else:
            # For YOLO models, don't call train() as it conflicts with the training method
            pass
        
        # Setup distributed sampler
        if self.is_distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.epoch)
        
        total_loss = 0.0
        total_samples = 0
        
        # Progress bar for rank 0
        if self.rank == 0:
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        else:
            pbar = self.train_loader
        
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch structures
            if isinstance(batch, tuple) and len(batch) == 2:
                images, targets = batch
            elif isinstance(batch, dict):
                if 'images' in batch:
                    images = batch['images']
                    targets = batch.get('targets', batch.get('labels', None))
                else:
                    raise ValueError("Batch dict must contain 'images' key")
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")
            
            # Move to device
            images = images.to(self.model.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'module'):
                outputs = self.model.module(images)
            else:
                outputs = self.model(images)
            
            # Compute loss
            if hasattr(self.model, 'module'):
                loss_dict = self.model.module.loss_fn(outputs, targets)
            else:
                loss_dict = self.model.loss_fn(outputs, targets)
            
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            self.global_step += 1
            
            # Update progress bar
            if self.rank == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
            
            # Log batch metrics
            if self.global_step % 100 == 0 and self.rank == 0:
                self.metrics_tracker.log_metrics({
                    'batch_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, self.global_step, 'train/')
        
        # Aggregate metrics across processes
        if self.is_distributed:
            total_loss = self._reduce_metric(total_loss)
            total_samples = self._reduce_metric(total_samples)
        
        avg_loss = total_loss / total_samples
        
        metrics = {
            'train_loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        # Update scheduler
        self.scheduler.step()
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        if hasattr(self.model, 'module'):
            self.model.module.eval()
        else:
            # For YOLO models, don't call eval() directly
            pass
        
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", disable=self.rank != 0):
                # Handle different batch structures
                if isinstance(batch, tuple) and len(batch) == 2:
                    images, targets = batch
                elif isinstance(batch, dict):
                    if 'images' in batch:
                        images = batch['images']
                        targets = batch.get('targets', batch.get('labels', None))
                    else:
                        raise ValueError("Batch dict must contain 'images' key")
                else:
                    raise ValueError(f"Unexpected batch type: {type(batch)}")
                
                # Move to device
                images = images.to(self.model.device)
                
                # Forward pass
                if hasattr(self.model, 'module'):
                    outputs = self.model.module(images)
                else:
                    outputs = self.model(images)
                
                # Compute loss
                if hasattr(self.model, 'module'):
                    loss_dict = self.model.module.loss_fn(outputs, targets)
                else:
                    loss_dict = self.model.loss_fn(outputs, targets)
                
                loss = loss_dict['total_loss']
                
                # Update metrics
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Collect predictions and targets
                all_predictions.append(outputs)
                all_targets.append(targets)
        
        # Aggregate metrics across processes
        if self.is_distributed:
            total_loss = self._reduce_metric(total_loss)
            total_samples = self._reduce_metric(total_samples)
        
        avg_loss = total_loss / total_samples
        
        # Compute detailed metrics
        if hasattr(self.model, 'module'):
            detailed_metrics = self.model.module.compute_metrics(all_predictions, all_targets)
        else:
            detailed_metrics = self.model.compute_metrics(all_predictions, all_targets)
        
        metrics = {
            'val_loss': avg_loss,
            **detailed_metrics
        }
        
        return metrics
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.config.training.epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Log metrics
            if self.rank == 0:
                self.metrics_tracker.log_metrics(epoch_metrics, epoch)
                
                # Update history
                for key, value in epoch_metrics.items():
                    self.training_history[key].append(value)
                
                # Log epoch summary
                logger.info(f"Epoch {epoch+1}/{self.config.training.epochs} - "
                           f"Train Loss: {train_metrics['train_loss']:.4f}, "
                           f"Val Loss: {val_metrics['val_loss']:.4f}, "
                           f"Val mAP: {val_metrics.get('mAP@0.5', 0.0):.4f}, "
                           f"Val mIoU: {val_metrics.get('mIoU', 0.0):.4f}")
            
            # Save checkpoint
            if self.rank == 0:
                is_best = val_metrics.get('mAP@0.5', 0.0) > self.best_metric
                
                if is_best:
                    self.best_metric = val_metrics.get('mAP@0.5', 0.0)
                
                if (epoch + 1) % self.config.training.save_period == 0 or is_best:
                    checkpoint_path = self.checkpoint_dir / f"epoch_{epoch+1}.pth"
                    if hasattr(self.model, 'module'):
                        self.model.module.save_checkpoint(
                            str(checkpoint_path), epoch + 1, self.optimizer, self.scheduler, is_best
                        )
                    else:
                        self.model.save_checkpoint(
                            str(checkpoint_path), epoch + 1, self.optimizer, self.scheduler, is_best
                        )
            
            # Check early stopping
            if self.early_stopping(val_metrics.get('mAP@0.5', 0.0), self.model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Run callbacks
            for callback in self.callbacks:
                callback(epoch, epoch_metrics)
        
        # Final logging
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        if self.rank == 0:
            # Save final metrics and plots
            self.metrics_tracker.save_metrics()
            self.metrics_tracker.plot_metrics()
            
            # Save training history
            history_path = self.checkpoint_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(dict(self.training_history), f, indent=2)
            
            logger.info(f"Training history saved to {history_path}")
        
        return dict(self.training_history)
    
    def test(self) -> Dict[str, float]:
        """
        Test the model on test dataset.
        
        Returns:
            Dictionary of test metrics
        """
        if self.test_loader is None:
            logger.warning("No test loader provided")
            return {}
        
        logger.info("Running test evaluation")
        
        # Load best model
        best_model_path = self.checkpoint_dir / 'best_model.pth'
        if best_model_path.exists():
            if hasattr(self.model, 'module'):
                self.model.module.load_checkpoint(str(best_model_path))
            else:
                self.model.load_checkpoint(str(best_model_path))
            logger.info("Loaded best model for testing")
        
        # Run evaluation
        if hasattr(self.model, 'module'):
            self.model.module.eval()
        else:
            # For YOLO models, don't call eval() directly
            pass
        test_metrics = self.validate()  # Using validate method with test_loader
        
        if self.rank == 0:
            logger.info("Test Results:")
            for key, value in test_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
            
            # Save test results
            test_results_path = self.checkpoint_dir / 'test_results.json'
            with open(test_results_path, 'w') as f:
                json.dump(test_metrics, f, indent=2)
        
        return test_metrics
    
    def _reduce_metric(self, metric: float) -> float:
        """Reduce metric across all processes."""
        if not self.is_distributed:
            return metric
        
        metric_tensor = torch.tensor(metric, device=self.model.device)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        return metric_tensor.item()
    
    def save_model(self, save_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            save_path: Path to save the model
        """
        if hasattr(self.model, 'module'):
            model_to_save = self.model.module
        else:
            model_to_save = self.model
        
        torch.save(model_to_save.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            load_path: Path to load the model from
        """
        if hasattr(self.model, 'module'):
            model_to_load = self.model.module
        else:
            model_to_load = self.model
        
        model_to_load.load_state_dict(torch.load(load_path, map_location=self.model.device))
        logger.info(f"Model loaded from {load_path}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.metrics_tracker.close()
        
        if self.is_distributed:
            dist.destroy_process_group()
        
        if self.config.experiment.wandb_log and self.rank == 0:
            wandb.finish()
        
        logger.info("Trainer cleanup completed")


def create_trainer(
    config: YOLOConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    callbacks: Optional[List[Callable]] = None
) -> YOLOTrainer:
    """
    Create a YOLOTrainer instance.
    
    Args:
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader (optional)
        model_path: Path to pre-trained model (optional)
        device: Device to use (optional)
        callbacks: List of callback functions (optional)
    
    Returns:
        YOLOTrainer instance
    """
    # Create model
    model = FashionYOLOv8(config, model_path, device)
    
    # Create trainer
    trainer = YOLOTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        callbacks=callbacks
    )
    
    return trainer


def train_fashion_yolo(
    config: YOLOConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    callbacks: Optional[List[Callable]] = None
) -> Dict[str, List[float]]:
    """
    Train a YOLOv8 model for fashion detection.
    
    Args:
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader (optional)
        model_path: Path to pre-trained model (optional)
        device: Device to use (optional)
        callbacks: List of callback functions (optional)
    
    Returns:
        Training history
    """
    trainer = create_trainer(
        config, train_loader, val_loader, test_loader, model_path, device, callbacks
    )
    
    try:
        # Train model
        history = trainer.train()
        
        # Test model if test loader is provided
        if test_loader is not None:
            test_metrics = trainer.test()
            history['test_metrics'] = test_metrics
        
        return history
    
    finally:
        trainer.cleanup()