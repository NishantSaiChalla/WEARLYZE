"""
Unified Training Pipeline for Fashion Detection System

This module provides a comprehensive training framework that supports all model types
(YOLO, classifiers, CLIP) with distributed training, early stopping, checkpointing,
and experiment tracking capabilities.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    
    # Model settings
    model_type: str = "yolo"  # yolo, classifier, clip
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip_val: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Optimizer settings
    optimizer: str = "AdamW"  # Adam, AdamW, SGD
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduler settings
    scheduler: str = "cosine"  # cosine, step, exponential, plateau
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    warmup_epochs: int = 3
    warmup_lr: float = 0.0001
    
    # Regularization
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_delta: float = 0.001
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"  # min or max
    
    # Cross-validation
    cv_folds: int = 5
    stratified_cv: bool = True
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Mixed precision
    mixed_precision: bool = True
    
    # Checkpointing
    save_checkpoint_every: int = 5
    save_top_k: int = 3
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_every: int = 10
    val_every: int = 1
    
    # Experiment tracking
    use_wandb: bool = True
    wandb_project: str = "fashion-detection"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    
    # Device settings
    device: str = "cuda"
    num_workers: int = 8
    pin_memory: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False


class EarlyStopper:
    """Early stopping utility."""
    
    def __init__(
        self,
        patience: int = 5,
        delta: float = 0.001,
        mode: str = "min"
    ):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        self.is_better = self._get_is_better_fn()
    
    def _get_is_better_fn(self) -> Callable:
        """Get comparison function based on mode."""
        if self.mode == "min":
            return lambda current, best: current < best - self.delta
        else:
            return lambda current, best: current > best + self.delta
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop early."""
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ModelCheckpoint:
    """Model checkpointing utility."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_top_k: int = 3,
        monitor: str = "val_loss",
        mode: str = "min",
        save_last: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        
        self.best_k_models = []
        self.kth_best_model_path = None
        
        self.is_better = self._get_is_better_fn()
    
    def _get_is_better_fn(self) -> Callable:
        """Get comparison function based on mode."""
        if self.mode == "min":
            return lambda current, best: current < best
        else:
            return lambda current, best: current > best
    
    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[_LRScheduler],
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'is_best': is_best
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
        
        # Save top-k checkpoints
        if self.save_top_k > 0 and self.monitor in metrics:
            self._save_top_k_checkpoint(epoch, checkpoint, metrics[self.monitor])
    
    def _save_top_k_checkpoint(
        self,
        epoch: int,
        checkpoint: Dict[str, Any],
        score: float
    ) -> None:
        """Save top-k best checkpoints."""
        model_path = self.checkpoint_dir / f"epoch_{epoch:03d}_score_{score:.4f}.pth"
        
        # Add to best models list
        self.best_k_models.append((score, model_path))
        
        # Sort by score
        self.best_k_models.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        
        # Keep only top-k
        if len(self.best_k_models) > self.save_top_k:
            # Remove worst model
            _, worst_path = self.best_k_models.pop()
            if worst_path.exists():
                worst_path.unlink()
        
        # Save current checkpoint
        torch.save(checkpoint, model_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None
    ) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint


class UnifiedTrainer:
    """Unified trainer for all model types in fashion detection system."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        loss_fn: Optional[nn.Module] = None,
        metrics: Optional[Dict[str, Callable]] = None
    ):
        self.config = config or TrainingConfig()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn or nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        self.metrics = metrics or {}
        
        # Setup device
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)
        
        # Setup distributed training
        if self.config.distributed:
            self._setup_distributed()
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup mixed precision
        self.scaler = GradScaler() if self.config.mixed_precision else None
        
        # Setup early stopping and checkpointing
        self.early_stopper = EarlyStopper(
            patience=self.config.early_stopping_patience,
            delta=self.config.early_stopping_delta,
            mode=self.config.early_stopping_mode
        )
        
        self.checkpoint_manager = ModelCheckpoint(
            checkpoint_dir=self.config.checkpoint_dir,
            save_top_k=self.config.save_top_k,
            monitor=self.config.early_stopping_metric,
            mode=self.config.early_stopping_mode
        )
        
        # Setup experiment tracking
        if self.config.use_wandb and (not self.config.distributed or self.config.rank == 0):
            self._setup_wandb()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_score = None
        self.training_history = defaultdict(list)
        
        # Set random seed
        self._set_seed()
    
    def _setup_distributed(self) -> None:
        """Setup distributed training."""
        if 'WORLD_SIZE' in os.environ:
            self.config.world_size = int(os.environ['WORLD_SIZE'])
            self.config.rank = int(os.environ['RANK'])
            self.config.local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        # Set device
        torch.cuda.set_device(self.config.local_rank)
        self.device = torch.device(f'cuda:{self.config.local_rank}')
        
        # Wrap model with DDP
        self.model = DDP(self.model, device_ids=[self.config.local_rank])
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        optimizer_params = {
            'lr': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            **self.config.optimizer_params
        }
        
        if self.config.optimizer == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), **optimizer_params)
        elif self.config.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(), **optimizer_params)
        elif self.config.optimizer == 'SGD':
            optimizer_params['momentum'] = optimizer_params.get('momentum', 0.9)
            optimizer = optim.SGD(self.model.parameters(), **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.scheduler == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.scheduler_params.get('min_lr', 1e-6)
            )
        elif self.config.scheduler == 'step':
            from torch.optim.lr_scheduler import StepLR
            scheduler = StepLR(
                self.optimizer,
                step_size=self.config.scheduler_params.get('step_size', 30),
                gamma=self.config.scheduler_params.get('gamma', 0.1)
            )
        elif self.config.scheduler == 'exponential':
            from torch.optim.lr_scheduler import ExponentialLR
            scheduler = ExponentialLR(
                self.optimizer,
                gamma=self.config.scheduler_params.get('gamma', 0.95)
            )
        elif self.config.scheduler == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.scheduler_params.get('mode', 'min'),
                factor=self.config.scheduler_params.get('factor', 0.1),
                patience=self.config.scheduler_params.get('patience', 10)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _setup_wandb(self) -> None:
        """Setup Weights & Biases logging."""
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            tags=self.config.wandb_tags,
            config=self.config.__dict__
        )
        
        # Watch model
        wandb.watch(self.model, log='all', log_freq=self.config.log_every)
    
    def _set_seed(self) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
        
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        # Set distributed sampler epoch
        if self.config.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.current_epoch)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            with autocast(enabled=self.config.mixed_precision):
                # Handle different batch structures from collate_fn
                if 'images' in batch:
                    inputs = batch['images']
                elif 'input' in batch:
                    inputs = batch['input']
                else:
                    raise ValueError("Batch must contain 'images' or 'input' key")
                
                outputs = self.model(inputs)
                
                # Handle different target structures
                if 'targets' in batch:
                    targets = batch['targets']
                elif 'labels' in batch:
                    targets = batch['labels']
                elif 'target' in batch:
                    targets = batch['target']
                else:
                    raise ValueError("Batch must contain 'targets', 'labels', or 'target' key")
                
                loss = self.loss_fn(outputs, targets)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    if self.config.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val
                    )
                
                # Optimizer step
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update metrics
            epoch_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Calculate batch metrics
            if self.metrics:
                with torch.no_grad():
                    batch_metrics = self._calculate_metrics(outputs, batch['target'])
                    for name, value in batch_metrics.items():
                        epoch_metrics[name] += value
            
            num_batches += 1
            
            # Log batch metrics
            if batch_idx % self.config.log_every == 0:
                self._log_batch_metrics(batch_idx, loss.item(), batch_metrics)
        
        # Average metrics
        epoch_loss /= num_batches
        for name in epoch_metrics:
            epoch_metrics[name] /= num_batches
        
        # Add loss to metrics
        epoch_metrics['loss'] = epoch_loss
        
        return dict(epoch_metrics)
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        epoch_loss = 0.0
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                with autocast(enabled=self.config.mixed_precision):
                    # Handle different batch structures from collate_fn
                    if 'images' in batch:
                        inputs = batch['images']
                    elif 'input' in batch:
                        inputs = batch['input']
                    else:
                        raise ValueError("Batch must contain 'images' or 'input' key")
                    
                    outputs = self.model(inputs)
                    
                    # Handle different target structures
                    if 'targets' in batch:
                        targets = batch['targets']
                    elif 'labels' in batch:
                        targets = batch['labels']
                    elif 'target' in batch:
                        targets = batch['target']
                    else:
                        raise ValueError("Batch must contain 'targets', 'labels', or 'target' key")
                    
                    loss = self.loss_fn(outputs, targets)
                
                # Update metrics
                epoch_loss += loss.item()
                
                # Calculate batch metrics
                if self.metrics:
                    batch_metrics = self._calculate_metrics(outputs, batch['target'])
                    for name, value in batch_metrics.items():
                        epoch_metrics[name] += value
                
                num_batches += 1
        
        # Average metrics
        epoch_loss /= num_batches
        for name in epoch_metrics:
            epoch_metrics[name] /= num_batches
        
        # Add loss to metrics
        epoch_metrics['val_loss'] = epoch_loss
        
        return dict(epoch_metrics)
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop."""
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = {}
            if epoch % self.config.val_every == 0:
                val_metrics = self.validate_epoch()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('val_loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
            
            # Log epoch metrics
            self._log_epoch_metrics(epoch, epoch_metrics)
            
            # Update training history
            for name, value in epoch_metrics.items():
                self.training_history[name].append(value)
            
            # Check for best model
            monitor_metric = epoch_metrics.get(self.config.early_stopping_metric)
            is_best = False
            if monitor_metric is not None:
                if self.best_score is None:
                    self.best_score = monitor_metric
                    is_best = True
                elif self.config.early_stopping_mode == 'min' and monitor_metric < self.best_score:
                    self.best_score = monitor_metric
                    is_best = True
                elif self.config.early_stopping_mode == 'max' and monitor_metric > self.best_score:
                    self.best_score = monitor_metric
                    is_best = True
            
            # Save checkpoint
            if epoch % self.config.save_checkpoint_every == 0 or is_best:
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics=epoch_metrics,
                    is_best=is_best
                )
            
            # Early stopping
            if monitor_metric is not None and self.early_stopper(monitor_metric):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info("Training completed")
        return dict(self.training_history)
    
    def cross_validate(
        self,
        dataset: torch.utils.data.Dataset,
        labels: np.ndarray
    ) -> Dict[str, List[float]]:
        """Perform k-fold cross-validation."""
        logger.info(f"Starting {self.config.cv_folds}-fold cross-validation")
        
        # Create stratified k-fold
        if self.config.stratified_cv:
            kfold = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.seed
            )
        else:
            from sklearn.model_selection import KFold
            kfold = KFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.seed
            )
        
        cv_results = defaultdict(list)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset, labels)):
            logger.info(f"Training fold {fold + 1}/{self.config.cv_folds}")
            
            # Create data loaders for this fold
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            
            train_loader = DataLoader(
                train_subset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
            
            # Update data loaders
            self.train_loader = train_loader
            self.val_loader = val_loader
            
            # Reset model and optimizer for this fold
            self.model.apply(self._weight_reset)
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
            
            # Reset training state
            self.current_epoch = 0
            self.global_step = 0
            self.best_score = None
            self.training_history = defaultdict(list)
            
            # Train for this fold
            fold_history = self.train()
            
            # Store fold results
            for metric_name, values in fold_history.items():
                if values:  # Only store non-empty metrics
                    cv_results[f"fold_{fold}_{metric_name}"] = values
                    
                    # Store best value for this fold
                    if metric_name.startswith('val_'):
                        best_value = min(values) if 'loss' in metric_name else max(values)
                        cv_results[f"fold_{fold}_best_{metric_name}"].append(best_value)
        
        # Calculate cross-validation statistics
        self._calculate_cv_stats(cv_results)
        
        logger.info("Cross-validation completed")
        return dict(cv_results)
    
    def _calculate_cv_stats(self, cv_results: Dict[str, List[float]]) -> None:
        """Calculate cross-validation statistics."""
        # Find all unique metric names
        metric_names = set()
        for key in cv_results.keys():
            if key.startswith('fold_') and '_best_' in key:
                metric_name = key.split('_best_')[1]
                metric_names.add(metric_name)
        
        # Calculate statistics for each metric
        for metric_name in metric_names:
            values = []
            for fold in range(self.config.cv_folds):
                key = f"fold_{fold}_best_{metric_name}"
                if key in cv_results:
                    values.extend(cv_results[key])
            
            if values:
                cv_results[f"cv_mean_{metric_name}"] = [np.mean(values)]
                cv_results[f"cv_std_{metric_name}"] = [np.std(values)]
                cv_results[f"cv_min_{metric_name}"] = [np.min(values)]
                cv_results[f"cv_max_{metric_name}"] = [np.max(values)]
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            elif isinstance(value, list) and len(value) > 0:
                # Handle list of tensors or dicts (for targets in detection)
                if isinstance(value[0], torch.Tensor):
                    moved_batch[key] = [v.to(self.device) for v in value]
                elif isinstance(value[0], dict):
                    # For detection targets with boxes and labels
                    moved_batch[key] = []
                    for v in value:
                        moved_dict = {}
                        for k, val in v.items():
                            if isinstance(val, torch.Tensor):
                                moved_dict[k] = val.to(self.device)
                            else:
                                moved_dict[k] = val
                        moved_batch[key].append(moved_dict)
                else:
                    moved_batch[key] = value
            else:
                moved_batch[key] = value
        return moved_batch
    
    def _calculate_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate batch metrics."""
        batch_metrics = {}
        
        # Get predictions
        if outputs.dim() > 1:
            predictions = torch.argmax(outputs, dim=1)
        else:
            predictions = outputs
        
        # Convert to numpy for sklearn metrics
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Calculate metrics
        for name, metric_fn in self.metrics.items():
            try:
                if name == 'accuracy':
                    value = accuracy_score(targets_np, predictions_np)
                elif name == 'f1_score':
                    value = f1_score(targets_np, predictions_np, average='weighted', zero_division=0)
                elif name == 'precision':
                    value = precision_score(targets_np, predictions_np, average='weighted', zero_division=0)
                elif name == 'recall':
                    value = recall_score(targets_np, predictions_np, average='weighted', zero_division=0)
                else:
                    value = metric_fn(predictions_np, targets_np)
                
                batch_metrics[name] = value
            except Exception as e:
                logger.warning(f"Error calculating metric {name}: {e}")
                batch_metrics[name] = 0.0
        
        return batch_metrics
    
    def _log_batch_metrics(
        self,
        batch_idx: int,
        loss: float,
        metrics: Dict[str, float]
    ) -> None:
        """Log batch metrics."""
        log_str = f"Epoch {self.current_epoch:03d} | Batch {batch_idx:04d} | Loss: {loss:.4f}"
        
        for name, value in metrics.items():
            log_str += f" | {name}: {value:.4f}"
        
        logger.info(log_str)
    
    def _log_epoch_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Log epoch metrics."""
        log_str = f"Epoch {epoch:03d} completed"
        
        for name, value in metrics.items():
            log_str += f" | {name}: {value:.4f}"
        
        logger.info(log_str)
        
        # Log to wandb
        if self.config.use_wandb and wandb.run is not None:
            wandb.log({**metrics, 'epoch': epoch, 'lr': self.optimizer.param_groups[0]['lr']})
    
    def _weight_reset(self, module: nn.Module) -> None:
        """Reset module weights."""
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
    
    def save_model(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")
    
    def export_training_history(self, path: str) -> None:
        """Export training history to JSON."""
        with open(path, 'w') as f:
            json.dump(dict(self.training_history), f, indent=2)
        logger.info(f"Training history exported to {path}")


# Helper functions
def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config_path: Optional[str] = None,
    **kwargs
) -> UnifiedTrainer:
    """Factory function to create a trainer."""
    
    # Load configuration
    if config_path:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig(**kwargs)
    
    # Create trainer
    trainer = UnifiedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    return trainer


def setup_distributed_training() -> None:
    """Setup distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Multi-node training
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Single-node multi-GPU training
        rank = 0
        world_size = torch.cuda.device_count()
        local_rank = 0
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    logger.info(f"Distributed training setup: rank={rank}, world_size={world_size}, local_rank={local_rank}")


# Export main classes
__all__ = [
    'TrainingConfig',
    'UnifiedTrainer',
    'EarlyStopper',
    'ModelCheckpoint',
    'create_trainer',
    'setup_distributed_training'
]