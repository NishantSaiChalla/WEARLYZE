"""
Custom Learning Rate Schedulers for Fashion Detection Training

This module provides advanced learning rate scheduling strategies optimized for
fashion detection tasks, including warmup, cosine annealing with restarts, and
adaptive scheduling based on validation metrics.
"""

import math
import warnings
from typing import Dict, List, Optional, Union, Callable, Any
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WarmupScheduler(_LRScheduler):
    """
    Warmup scheduler that gradually increases learning rate from a small value
    to the target learning rate over a specified number of epochs.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        warmup_lr: float = 1e-6,
        target_lr: Optional[float] = None,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_epochs: Number of warmup epochs
            warmup_lr: Initial learning rate during warmup
            target_lr: Target learning rate after warmup (if None, uses optimizer's lr)
            last_epoch: The index of last epoch
            verbose: If True, prints a message to stdout for each update
        """
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        
        if target_lr is None:
            self.target_lr = optimizer.param_groups[0]['lr']
        else:
            self.target_lr = target_lr
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for each parameter group."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (self.last_epoch + 1) / self.warmup_epochs
            return [self.warmup_lr + (self.target_lr - self.warmup_lr) * lr_scale
                    for _ in self.base_lrs]
        else:
            # Return target learning rate
            return [self.target_lr for _ in self.base_lrs]


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing with warmup and restarts scheduler.
    
    This scheduler combines warmup with cosine annealing and periodic restarts
    to help escape local minima and improve convergence.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        warmup_epochs: int = 0,
        warmup_lr: float = 1e-6,
        gamma: float = 1.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize cosine annealing warmup restarts scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            T_0: Number of iterations for the first restart
            T_mult: A factor increases T_i after a restart
            eta_min: Minimum learning rate
            warmup_epochs: Number of warmup epochs
            warmup_lr: Initial learning rate during warmup
            gamma: Decrease factor for maximum learning rate after restart
            last_epoch: The index of last epoch
            verbose: If True, prints a message to stdout for each update
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.gamma = gamma
        
        self.T_cur = 0
        self.T_i = T_0
        self.restart_count = 0
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for each parameter group."""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            lr_scale = (self.last_epoch + 1) / self.warmup_epochs
            return [self.warmup_lr + (base_lr - self.warmup_lr) * lr_scale
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            adjusted_epoch = self.last_epoch - self.warmup_epochs
            
            if adjusted_epoch >= self.T_i:
                # Time for restart
                self.restart_count += 1
                self.T_cur = 0
                self.T_i = self.T_i * self.T_mult
                
                # Adjust base learning rates
                for i, group in enumerate(self.optimizer.param_groups):
                    self.base_lrs[i] = self.base_lrs[i] * self.gamma
            else:
                self.T_cur = adjusted_epoch % self.T_i
            
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                    for base_lr in self.base_lrs]


class AdaptiveScheduler(_LRScheduler):
    """
    Adaptive learning rate scheduler that adjusts learning rate based on
    validation metrics and training progress.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        metric_name: str = 'val_loss',
        mode: str = 'min',
        factor: float = 0.5,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = 'rel',
        cooldown: int = 0,
        min_lr: float = 0,
        eps: float = 1e-8,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize adaptive scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            metric_name: Name of the metric to monitor
            mode: 'min' or 'max' for the metric
            factor: Factor by which the learning rate will be reduced
            patience: Number of epochs with no improvement after which learning rate will be reduced
            threshold: Threshold for measuring the new optimum
            threshold_mode: 'rel' or 'abs'
            cooldown: Number of epochs to wait before resuming normal operation
            min_lr: Lower bound on the learning rate
            eps: Minimal decay applied to lr
            last_epoch: The index of last epoch
            verbose: If True, prints a message to stdout for each update
        """
        self.metric_name = metric_name
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        
        self.best_metric = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.metric_history = []
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def step(self, metrics: Dict[str, float]) -> None:
        """
        Step the scheduler with validation metrics.
        
        Args:
            metrics: Dictionary of metrics
        """
        if self.metric_name not in metrics:
            logger.warning(f"Metric '{self.metric_name}' not found in metrics")
            return
        
        current_metric = metrics[self.metric_name]
        self.metric_history.append(current_metric)
        
        # Check if we're in cooldown period
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return
        
        # Check if metric has improved
        if self.best_metric is None:
            self.best_metric = current_metric
        elif self._is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        # Reduce learning rate if patience exceeded
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
            self.cooldown_counter = self.cooldown
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold if self.mode == 'min' else self.threshold + 1.
            return (current < best * rel_epsilon) if self.mode == 'min' else (current > best * rel_epsilon)
        else:  # abs
            return (current < best - self.threshold) if self.mode == 'min' else (current > best + self.threshold)
    
    def _reduce_lr(self) -> None:
        """Reduce learning rate for all parameter groups."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    logger.info(f'Reducing learning rate of group {i} to {new_lr:.4e}')
    
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


class CyclicScheduler(_LRScheduler):
    """
    Cyclic learning rate scheduler that cycles learning rate between two bounds.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        gamma: float = 1.0,
        scale_fn: Optional[Callable] = None,
        scale_mode: str = 'cycle',
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize cyclic scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            base_lr: Lower learning rate bound
            max_lr: Upper learning rate bound
            step_size_up: Number of training iterations in the increasing half of a cycle
            step_size_down: Number of training iterations in the decreasing half of a cycle
            mode: 'triangular', 'triangular2', or 'exp_range'
            gamma: Constant in 'exp_range' scaling function
            scale_fn: Custom scaling function
            scale_mode: 'cycle' or 'iterations'
            last_epoch: The index of last epoch
            verbose: If True, prints a message to stdout for each update
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        
        self.total_size = self.step_size_up + self.step_size_down
        self.step_num = 0
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for each parameter group."""
        cycle = math.floor(1 + self.step_num / self.total_size)
        x = abs(self.step_num / self.step_size_up - 2 * cycle + 1)
        
        if self.step_num <= self.step_size_up:
            scale_factor = x
        else:
            scale_factor = 1 - x
        
        if self.scale_fn is None:
            if self.mode == 'triangular':
                scale_factor = 1.0
            elif self.mode == 'triangular2':
                scale_factor = 1 / (2 ** (cycle - 1))
            elif self.mode == 'exp_range':
                scale_factor = self.gamma ** self.step_num
        else:
            if self.scale_mode == 'cycle':
                scale_factor = self.scale_fn(cycle)
            else:
                scale_factor = self.scale_fn(self.step_num)
        
        lr = self.base_lr + (self.max_lr - self.base_lr) * scale_factor
        return [lr for _ in self.base_lrs]
    
    def step(self, epoch: Optional[int] = None) -> None:
        """Step the scheduler."""
        if epoch is None:
            self.step_num += 1
        else:
            self.step_num = epoch
        
        super().step(epoch)


class OneCycleScheduler(_LRScheduler):
    """
    One cycle learning rate scheduler that implements the 1cycle policy.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize one cycle scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            max_lr: Upper learning rate bound
            total_steps: Total number of training steps
            pct_start: Percentage of the cycle spent increasing the learning rate
            anneal_strategy: 'cos' or 'linear'
            div_factor: Initial learning rate will be max_lr/div_factor
            final_div_factor: Final learning rate will be max_lr/final_div_factor
            last_epoch: The index of last epoch
            verbose: If True, prints a message to stdout for each update
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        self.step_size_up = int(total_steps * pct_start)
        self.step_size_down = total_steps - self.step_size_up
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for each parameter group."""
        if self.last_epoch < self.step_size_up:
            # Increasing phase
            lr_scale = self.last_epoch / self.step_size_up
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * lr_scale
        else:
            # Decreasing phase
            step_down = self.last_epoch - self.step_size_up
            
            if self.anneal_strategy == 'cos':
                lr_scale = (1 + math.cos(math.pi * step_down / self.step_size_down)) / 2
            else:  # linear
                lr_scale = 1 - step_down / self.step_size_down
            
            lr = self.final_lr + (self.max_lr - self.final_lr) * lr_scale
        
        return [lr for _ in self.base_lrs]


class PolynomialScheduler(_LRScheduler):
    """
    Polynomial learning rate scheduler that decays learning rate polynomially.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_iters: int,
        power: float = 1.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize polynomial scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            total_iters: Total number of training iterations
            power: Power of the polynomial
            last_epoch: The index of last epoch
            verbose: If True, prints a message to stdout for each update
        """
        self.total_iters = total_iters
        self.power = power
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for each parameter group."""
        if self.last_epoch >= self.total_iters:
            return [0 for _ in self.base_lrs]
        
        decay_factor = (1 - self.last_epoch / self.total_iters) ** self.power
        return [base_lr * decay_factor for base_lr in self.base_lrs]


class MultiStepWarmupScheduler(_LRScheduler):
    """
    Multi-step scheduler with warmup that decays learning rate at specified milestones.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_epochs: int = 0,
        warmup_lr: float = 1e-6,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize multi-step warmup scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            milestones: List of epoch indices for learning rate decay
            gamma: Multiplicative factor of learning rate decay
            warmup_epochs: Number of warmup epochs
            warmup_lr: Initial learning rate during warmup
            last_epoch: The index of last epoch
            verbose: If True, prints a message to stdout for each update
        """
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for each parameter group."""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            lr_scale = (self.last_epoch + 1) / self.warmup_epochs
            return [self.warmup_lr + (base_lr - self.warmup_lr) * lr_scale
                    for base_lr in self.base_lrs]
        else:
            # Multi-step phase
            decay_count = sum(1 for milestone in self.milestones if milestone <= self.last_epoch)
            decay_factor = self.gamma ** decay_count
            return [base_lr * decay_factor for base_lr in self.base_lrs]


class ChainedScheduler:
    """
    Chains multiple schedulers together for complex scheduling strategies.
    """
    
    def __init__(self, schedulers: List[_LRScheduler]):
        """
        Initialize chained scheduler.
        
        Args:
            schedulers: List of schedulers to chain
        """
        self.schedulers = schedulers
        self.current_scheduler = 0
    
    def step(self, *args, **kwargs) -> None:
        """Step the current scheduler."""
        if self.current_scheduler < len(self.schedulers):
            self.schedulers[self.current_scheduler].step(*args, **kwargs)
    
    def switch_scheduler(self, index: int) -> None:
        """Switch to a different scheduler."""
        if 0 <= index < len(self.schedulers):
            self.current_scheduler = index
        else:
            raise IndexError(f"Scheduler index {index} out of range")
    
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        if self.current_scheduler < len(self.schedulers):
            return self.schedulers[self.current_scheduler].get_lr()
        return [0.0]


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    **kwargs
) -> _LRScheduler:
    """
    Factory function to create learning rate schedulers.
    
    Args:
        optimizer: Optimizer to wrap
        scheduler_type: Type of scheduler to create
        **kwargs: Additional arguments for the scheduler
    
    Returns:
        Learning rate scheduler instance
    
    Raises:
        ValueError: If scheduler_type is not supported
    """
    scheduler_registry = {
        'warmup': WarmupScheduler,
        'cosine_warmup_restarts': CosineAnnealingWarmupRestarts,
        'adaptive': AdaptiveScheduler,
        'cyclic': CyclicScheduler,
        'onecycle': OneCycleScheduler,
        'polynomial': PolynomialScheduler,
        'multistep_warmup': MultiStepWarmupScheduler,
        
        # Standard PyTorch schedulers
        'step': torch.optim.lr_scheduler.StepLR,
        'multistep': torch.optim.lr_scheduler.MultiStepLR,
        'exponential': torch.optim.lr_scheduler.ExponentialLR,
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'lambda': torch.optim.lr_scheduler.LambdaLR,
    }
    
    if scheduler_type not in scheduler_registry:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}. "
                        f"Supported types: {list(scheduler_registry.keys())}")
    
    scheduler_class = scheduler_registry[scheduler_type]
    return scheduler_class(optimizer, **kwargs)


def get_scheduler_config(scheduler_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a scheduler type.
    
    Args:
        scheduler_type: Type of scheduler
    
    Returns:
        Default configuration dictionary
    """
    configs = {
        'warmup': {
            'warmup_epochs': 3,
            'warmup_lr': 1e-6,
        },
        'cosine_warmup_restarts': {
            'T_0': 50,
            'T_mult': 1,
            'eta_min': 1e-6,
            'warmup_epochs': 3,
            'warmup_lr': 1e-6,
            'gamma': 0.9,
        },
        'adaptive': {
            'metric_name': 'val_loss',
            'mode': 'min',
            'factor': 0.5,
            'patience': 10,
            'threshold': 1e-4,
            'min_lr': 1e-6,
        },
        'cyclic': {
            'base_lr': 1e-4,
            'max_lr': 1e-2,
            'step_size_up': 2000,
            'mode': 'triangular',
        },
        'onecycle': {
            'max_lr': 1e-2,
            'total_steps': 10000,
            'pct_start': 0.3,
            'anneal_strategy': 'cos',
            'div_factor': 25.0,
            'final_div_factor': 1e4,
        },
        'polynomial': {
            'total_iters': 10000,
            'power': 1.0,
        },
        'multistep_warmup': {
            'milestones': [30, 60, 90],
            'gamma': 0.1,
            'warmup_epochs': 3,
            'warmup_lr': 1e-6,
        },
    }
    
    return configs.get(scheduler_type, {})


# Export all classes and functions
__all__ = [
    'WarmupScheduler',
    'CosineAnnealingWarmupRestarts',
    'AdaptiveScheduler',
    'CyclicScheduler',
    'OneCycleScheduler',
    'PolynomialScheduler',
    'MultiStepWarmupScheduler',
    'ChainedScheduler',
    'create_scheduler',
    'get_scheduler_config'
]