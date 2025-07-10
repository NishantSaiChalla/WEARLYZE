"""
Experiment Management System for Fashion Detection

This module provides comprehensive experiment management capabilities including
hyperparameter optimization, automated model comparison, and configuration management.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
import pickle
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, f1_score

# Bayesian optimization imports
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Bayesian optimization will be disabled.")

# Hyperopt imports (alternative to Optuna)
try:
    import hyperopt
    from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

# Ray Tune imports (for distributed hyperparameter optimization)
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.suggest.bayesopt import BayesOptSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from .trainer import UnifiedTrainer, TrainingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment management."""
    
    # Experiment settings
    experiment_name: str = "fashion_detection_experiment"
    experiment_dir: str = "experiments"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Hyperparameter optimization
    optimization_method: str = "grid"  # grid, random, bayesian, hyperopt, ray
    max_trials: int = 100
    max_concurrent_trials: int = 4
    optimization_timeout: int = 3600  # seconds
    
    # Bayesian optimization settings (Optuna)
    bayesian_sampler: str = "TPE"  # TPE, Random, CmaEs
    bayesian_pruner: str = "MedianPruner"  # MedianPruner, HyperbandPruner
    
    # Grid/Random search settings
    random_seed: int = 42
    
    # Model comparison settings
    comparison_metric: str = "val_f1_score"  # metric to compare models
    comparison_mode: str = "max"  # max or min
    
    # Cross-validation settings
    cv_folds: int = 5
    stratified_cv: bool = True
    
    # Resource management
    gpu_per_trial: float = 0.25
    cpu_per_trial: int = 2
    memory_per_trial: str = "2GB"
    
    # Logging and monitoring
    log_level: str = "INFO"
    save_all_models: bool = False
    save_top_k: int = 5
    
    # Early stopping for trials
    trial_early_stopping: bool = True
    trial_patience: int = 10
    
    # Reproducibility
    deterministic: bool = False


class HyperparameterSpace:
    """Hyperparameter search space definition."""
    
    def __init__(self):
        self.space = {}
        self.distributions = {}
    
    def add_categorical(self, name: str, choices: List[Any]) -> None:
        """Add categorical hyperparameter."""
        self.space[name] = choices
        self.distributions[name] = 'categorical'
    
    def add_uniform(self, name: str, low: float, high: float) -> None:
        """Add uniform continuous hyperparameter."""
        self.space[name] = (low, high)
        self.distributions[name] = 'uniform'
    
    def add_loguniform(self, name: str, low: float, high: float) -> None:
        """Add log-uniform continuous hyperparameter."""
        self.space[name] = (low, high)
        self.distributions[name] = 'loguniform'
    
    def add_int(self, name: str, low: int, high: int) -> None:
        """Add integer hyperparameter."""
        self.space[name] = (low, high)
        self.distributions[name] = 'int'
    
    def add_logint(self, name: str, low: int, high: int) -> None:
        """Add log-integer hyperparameter."""
        self.space[name] = (low, high)
        self.distributions[name] = 'logint'
    
    def get_optuna_space(self, trial) -> Dict[str, Any]:
        """Convert to Optuna search space."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization")
        
        params = {}
        for name, value in self.space.items():
            dist_type = self.distributions[name]
            
            if dist_type == 'categorical':
                params[name] = trial.suggest_categorical(name, value)
            elif dist_type == 'uniform':
                params[name] = trial.suggest_float(name, value[0], value[1])
            elif dist_type == 'loguniform':
                params[name] = trial.suggest_float(name, value[0], value[1], log=True)
            elif dist_type == 'int':
                params[name] = trial.suggest_int(name, value[0], value[1])
            elif dist_type == 'logint':
                params[name] = trial.suggest_int(name, value[0], value[1], log=True)
        
        return params
    
    def get_hyperopt_space(self) -> Dict[str, Any]:
        """Convert to Hyperopt search space."""
        if not HYPEROPT_AVAILABLE:
            raise ImportError("Hyperopt is required for hyperopt optimization")
        
        params = {}
        for name, value in self.space.items():
            dist_type = self.distributions[name]
            
            if dist_type == 'categorical':
                params[name] = hp.choice(name, value)
            elif dist_type == 'uniform':
                params[name] = hp.uniform(name, value[0], value[1])
            elif dist_type == 'loguniform':
                params[name] = hp.lognormal(name, np.log(value[0]), np.log(value[1]))
            elif dist_type == 'int':
                params[name] = hp.randint(name, value[0], value[1] + 1)
            elif dist_type == 'logint':
                params[name] = hp.lognormal(name, np.log(value[0]), np.log(value[1]))
        
        return params
    
    def get_ray_space(self) -> Dict[str, Any]:
        """Convert to Ray Tune search space."""
        if not RAY_AVAILABLE:
            raise ImportError("Ray is required for Ray Tune optimization")
        
        params = {}
        for name, value in self.space.items():
            dist_type = self.distributions[name]
            
            if dist_type == 'categorical':
                params[name] = tune.choice(value)
            elif dist_type == 'uniform':
                params[name] = tune.uniform(value[0], value[1])
            elif dist_type == 'loguniform':
                params[name] = tune.loguniform(value[0], value[1])
            elif dist_type == 'int':
                params[name] = tune.randint(value[0], value[1] + 1)
            elif dist_type == 'logint':
                params[name] = tune.lograndint(value[0], value[1] + 1)
        
        return params
    
    def sample_random(self, n_samples: int = 1) -> List[Dict[str, Any]]:
        """Sample random configurations."""
        samples = []
        np.random.seed(42)  # For reproducibility
        
        for _ in range(n_samples):
            sample = {}
            for name, value in self.space.items():
                dist_type = self.distributions[name]
                
                if dist_type == 'categorical':
                    sample[name] = np.random.choice(value)
                elif dist_type == 'uniform':
                    sample[name] = np.random.uniform(value[0], value[1])
                elif dist_type == 'loguniform':
                    sample[name] = np.exp(np.random.uniform(np.log(value[0]), np.log(value[1])))
                elif dist_type == 'int':
                    sample[name] = np.random.randint(value[0], value[1] + 1)
                elif dist_type == 'logint':
                    sample[name] = int(np.exp(np.random.uniform(np.log(value[0]), np.log(value[1]))))
            
            samples.append(sample)
        
        return samples
    
    def get_grid_search_space(self) -> List[Dict[str, Any]]:
        """Get grid search space."""
        # For grid search, we need discrete values
        grid_space = {}
        
        for name, value in self.space.items():
            dist_type = self.distributions[name]
            
            if dist_type == 'categorical':
                grid_space[name] = value
            elif dist_type in ['uniform', 'loguniform']:
                # Create discrete values for continuous parameters
                if dist_type == 'uniform':
                    grid_space[name] = np.linspace(value[0], value[1], 5).tolist()
                else:  # loguniform
                    grid_space[name] = np.logspace(np.log10(value[0]), np.log10(value[1]), 5).tolist()
            elif dist_type in ['int', 'logint']:
                if dist_type == 'int':
                    grid_space[name] = list(range(value[0], min(value[1] + 1, value[0] + 5)))
                else:  # logint
                    grid_space[name] = [int(x) for x in np.logspace(np.log10(value[0]), np.log10(value[1]), 5)]
        
        return list(ParameterGrid(grid_space))


class ExperimentResult:
    """Container for experiment results."""
    
    def __init__(self, trial_id: str, params: Dict[str, Any]):
        self.trial_id = trial_id
        self.params = params
        self.metrics = {}
        self.training_history = {}
        self.model_path = None
        self.config = None
        self.start_time = None
        self.end_time = None
        self.status = 'running'
        self.error = None
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update trial metrics."""
        self.metrics.update(metrics)
    
    def set_training_history(self, history: Dict[str, List[float]]) -> None:
        """Set training history."""
        self.training_history = history
    
    def complete(self, status: str = 'completed', error: str = None) -> None:
        """Mark trial as completed."""
        self.status = status
        self.error = error
        self.end_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trial_id': self.trial_id,
            'params': self.params,
            'metrics': self.metrics,
            'training_history': self.training_history,
            'model_path': self.model_path,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'status': self.status,
            'error': self.error
        }


class ExperimentManager:
    """Manages multiple experiments with hyperparameter optimization."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        model_factory: Callable[[Dict[str, Any]], nn.Module],
        data_loaders: Dict[str, DataLoader],
        hyperparameter_space: HyperparameterSpace,
        base_training_config: TrainingConfig
    ):
        self.config = config
        self.model_factory = model_factory
        self.data_loaders = data_loaders
        self.hyperparameter_space = hyperparameter_space
        self.base_training_config = base_training_config
        
        # Setup experiment directory
        self.experiment_dir = Path(config.experiment_dir) / config.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = []
        self.best_result = None
        self.study = None
        
        # Setup logging
        self._setup_logging()
        
        # Save experiment configuration
        self._save_config()
    
    def _setup_logging(self) -> None:
        """Setup experiment logging."""
        log_file = self.experiment_dir / "experiment.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, self.config.log_level))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    
    def _save_config(self) -> None:
        """Save experiment configuration."""
        config_file = self.experiment_dir / "config.yaml"
        
        config_dict = {
            'experiment_config': self.config.__dict__,
            'base_training_config': self.base_training_config.__dict__,
            'hyperparameter_space': {
                'space': self.hyperparameter_space.space,
                'distributions': self.hyperparameter_space.distributions
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def run_optimization(self) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        logger.info(f"Starting hyperparameter optimization with method: {self.config.optimization_method}")
        
        if self.config.optimization_method == 'grid':
            return self._run_grid_search()
        elif self.config.optimization_method == 'random':
            return self._run_random_search()
        elif self.config.optimization_method == 'bayesian':
            return self._run_bayesian_optimization()
        elif self.config.optimization_method == 'hyperopt':
            return self._run_hyperopt_optimization()
        elif self.config.optimization_method == 'ray':
            return self._run_ray_optimization()
        else:
            raise ValueError(f"Unsupported optimization method: {self.config.optimization_method}")
    
    def _run_grid_search(self) -> Dict[str, Any]:
        """Run grid search optimization."""
        logger.info("Running grid search optimization")
        
        param_grid = self.hyperparameter_space.get_grid_search_space()
        
        for i, params in enumerate(param_grid):
            if i >= self.config.max_trials:
                break
            
            trial_id = f"grid_{i:04d}"
            logger.info(f"Running trial {trial_id} with params: {params}")
            
            result = self._run_trial(trial_id, params)
            self.results.append(result)
            
            # Update best result
            self._update_best_result(result)
        
        return self._get_optimization_summary()
    
    def _run_random_search(self) -> Dict[str, Any]:
        """Run random search optimization."""
        logger.info("Running random search optimization")
        
        param_samples = self.hyperparameter_space.sample_random(self.config.max_trials)
        
        for i, params in enumerate(param_samples):
            trial_id = f"random_{i:04d}"
            logger.info(f"Running trial {trial_id} with params: {params}")
            
            result = self._run_trial(trial_id, params)
            self.results.append(result)
            
            # Update best result
            self._update_best_result(result)
        
        return self._get_optimization_summary()
    
    def _run_bayesian_optimization(self) -> Dict[str, Any]:
        """Run Bayesian optimization using Optuna."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization")
        
        logger.info("Running Bayesian optimization with Optuna")
        
        # Create study
        study_name = f"{self.config.experiment_name}_bayesian"
        storage = f"sqlite:///{self.experiment_dir}/optuna_study.db"
        
        # Create sampler
        if self.config.bayesian_sampler == 'TPE':
            sampler = TPESampler(seed=self.config.random_seed)
        else:
            sampler = optuna.samplers.RandomSampler(seed=self.config.random_seed)
        
        # Create pruner
        if self.config.bayesian_pruner == 'MedianPruner':
            pruner = MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Create study
        direction = 'maximize' if self.config.comparison_mode == 'max' else 'minimize'
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
        
        self.study = study
        
        # Run optimization
        study.optimize(
            self._optuna_objective,
            n_trials=self.config.max_trials,
            timeout=self.config.optimization_timeout
        )
        
        # Get best result
        best_trial = study.best_trial
        self.best_result = ExperimentResult(
            trial_id=f"bayesian_{best_trial.number:04d}",
            params=best_trial.params
        )
        self.best_result.metrics = {self.config.comparison_metric: best_trial.value}
        
        return self._get_optimization_summary()
    
    def _run_hyperopt_optimization(self) -> Dict[str, Any]:
        """Run optimization using Hyperopt."""
        if not HYPEROPT_AVAILABLE:
            raise ImportError("Hyperopt is required for hyperopt optimization")
        
        logger.info("Running optimization with Hyperopt")
        
        # Create search space
        space = self.hyperparameter_space.get_hyperopt_space()
        
        # Create trials object
        trials = Trials()
        
        # Run optimization
        best_params = fmin(
            fn=self._hyperopt_objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.config.max_trials,
            trials=trials,
            rstate=np.random.RandomState(self.config.random_seed)
        )
        
        # Get best result
        best_trial = trials.best_trial
        self.best_result = ExperimentResult(
            trial_id=f"hyperopt_{len(trials.trials):04d}",
            params=best_params
        )
        self.best_result.metrics = {self.config.comparison_metric: -best_trial['result']['loss']}
        
        return self._get_optimization_summary()
    
    def _run_ray_optimization(self) -> Dict[str, Any]:
        """Run optimization using Ray Tune."""
        if not RAY_AVAILABLE:
            raise ImportError("Ray is required for Ray Tune optimization")
        
        logger.info("Running optimization with Ray Tune")
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init()
        
        # Create search space
        config = self.hyperparameter_space.get_ray_space()
        
        # Create scheduler
        scheduler = ASHAScheduler(
            metric=self.config.comparison_metric,
            mode=self.config.comparison_mode,
            max_t=self.base_training_config.epochs,
            grace_period=10,
            reduction_factor=2
        )
        
        # Create search algorithm
        search_alg = BayesOptSearch(
            metric=self.config.comparison_metric,
            mode=self.config.comparison_mode,
            random_state=self.config.random_seed
        )
        
        # Run optimization
        analysis = tune.run(
            self._ray_trainable,
            config=config,
            num_samples=self.config.max_trials,
            scheduler=scheduler,
            search_alg=search_alg,
            resources_per_trial={
                'cpu': self.config.cpu_per_trial,
                'gpu': self.config.gpu_per_trial
            },
            local_dir=str(self.experiment_dir),
            name='ray_tune_optimization'
        )
        
        # Get best result
        best_trial = analysis.get_best_trial(
            metric=self.config.comparison_metric,
            mode=self.config.comparison_mode
        )
        
        self.best_result = ExperimentResult(
            trial_id=f"ray_{best_trial.trial_id}",
            params=best_trial.config
        )
        self.best_result.metrics = best_trial.last_result
        
        return self._get_optimization_summary()
    
    def _run_trial(self, trial_id: str, params: Dict[str, Any]) -> ExperimentResult:
        """Run a single trial."""
        result = ExperimentResult(trial_id, params)
        result.start_time = time.time()
        
        try:
            # Create model
            model = self.model_factory(params)
            
            # Update training config with hyperparameters
            training_config = self._update_training_config(params)
            
            # Create trainer
            trainer = UnifiedTrainer(
                model=model,
                train_loader=self.data_loaders['train'],
                val_loader=self.data_loaders.get('val'),
                config=training_config
            )
            
            # Train model
            if self.config.cv_folds > 1:
                # Cross-validation
                dataset = self.data_loaders['train'].dataset
                labels = self._get_labels_from_dataset(dataset)
                
                cv_results = trainer.cross_validate(dataset, labels)
                result.set_training_history(cv_results)
                
                # Extract final metrics
                final_metrics = {}
                for key, values in cv_results.items():
                    if key.startswith('cv_mean_'):
                        metric_name = key.replace('cv_mean_', '')
                        final_metrics[metric_name] = values[0]
                
                result.update_metrics(final_metrics)
            else:
                # Regular training
                training_history = trainer.train()
                result.set_training_history(training_history)
                
                # Extract final metrics
                final_metrics = {}
                for key, values in training_history.items():
                    if values:
                        final_metrics[key] = values[-1]
                
                result.update_metrics(final_metrics)
            
            # Save model if requested
            if self.config.save_all_models or len(self.results) < self.config.save_top_k:
                model_path = self.experiment_dir / f"model_{trial_id}.pth"
                trainer.save_model(str(model_path))
                result.model_path = str(model_path)
            
            result.complete('completed')
            
        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {str(e)}")
            result.complete('failed', str(e))
        
        # Save trial result
        self._save_trial_result(result)
        
        return result
    
    def _optuna_objective(self, trial) -> float:
        """Objective function for Optuna."""
        params = self.hyperparameter_space.get_optuna_space(trial)
        trial_id = f"bayesian_{trial.number:04d}"
        
        result = self._run_trial(trial_id, params)
        self.results.append(result)
        
        # Update best result
        self._update_best_result(result)
        
        # Return metric value
        if self.config.comparison_metric in result.metrics:
            return result.metrics[self.config.comparison_metric]
        else:
            # Return worst possible value if metric not available
            return float('-inf') if self.config.comparison_mode == 'max' else float('inf')
    
    def _hyperopt_objective(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Objective function for Hyperopt."""
        trial_id = f"hyperopt_{len(self.results):04d}"
        
        result = self._run_trial(trial_id, params)
        self.results.append(result)
        
        # Update best result
        self._update_best_result(result)
        
        # Return loss (hyperopt minimizes)
        if self.config.comparison_metric in result.metrics:
            loss = result.metrics[self.config.comparison_metric]
            if self.config.comparison_mode == 'max':
                loss = -loss
            return {'loss': loss, 'status': STATUS_OK}
        else:
            return {'loss': float('inf'), 'status': STATUS_OK}
    
    def _ray_trainable(self, config: Dict[str, Any]) -> None:
        """Trainable function for Ray Tune."""
        trial_id = f"ray_{tune.get_trial_id()}"
        
        result = self._run_trial(trial_id, config)
        
        # Report result to Ray Tune
        if self.config.comparison_metric in result.metrics:
            tune.report(**result.metrics)
        else:
            tune.report(**{self.config.comparison_metric: 0.0})
    
    def _update_training_config(self, params: Dict[str, Any]) -> TrainingConfig:
        """Update training configuration with hyperparameters."""
        config_dict = self.base_training_config.__dict__.copy()
        
        # Update with hyperparameters
        config_dict.update(params)
        
        # Create new config
        return TrainingConfig(**config_dict)
    
    def _get_labels_from_dataset(self, dataset) -> np.ndarray:
        """Extract labels from dataset for cross-validation."""
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        return np.array(labels)
    
    def _update_best_result(self, result: ExperimentResult) -> None:
        """Update best result based on comparison metric."""
        if self.config.comparison_metric not in result.metrics:
            return
        
        metric_value = result.metrics[self.config.comparison_metric]
        
        if self.best_result is None:
            self.best_result = result
        else:
            best_metric = self.best_result.metrics.get(self.config.comparison_metric)
            if best_metric is None:
                self.best_result = result
            elif self.config.comparison_mode == 'max' and metric_value > best_metric:
                self.best_result = result
            elif self.config.comparison_mode == 'min' and metric_value < best_metric:
                self.best_result = result
    
    def _save_trial_result(self, result: ExperimentResult) -> None:
        """Save trial result to file."""
        result_file = self.experiment_dir / f"trial_{result.trial_id}.json"
        
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def _get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        summary = {
            'experiment_name': self.config.experiment_name,
            'total_trials': len(self.results),
            'completed_trials': len([r for r in self.results if r.status == 'completed']),
            'failed_trials': len([r for r in self.results if r.status == 'failed']),
            'best_result': self.best_result.to_dict() if self.best_result else None,
            'optimization_method': self.config.optimization_method,
            'comparison_metric': self.config.comparison_metric,
            'comparison_mode': self.config.comparison_mode
        }
        
        # Save summary
        summary_file = self.experiment_dir / "optimization_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as pandas DataFrame."""
        data = []
        
        for result in self.results:
            row = {
                'trial_id': result.trial_id,
                'status': result.status,
                **result.params,
                **result.metrics
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_file = self.experiment_dir / "results.csv"
        df.to_csv(csv_file, index=False)
        
        return df
    
    def compare_models(self) -> Dict[str, Any]:
        """Compare all models and generate comparison report."""
        if not self.results:
            return {}
        
        # Get completed results
        completed_results = [r for r in self.results if r.status == 'completed']
        
        if not completed_results:
            return {}
        
        # Create comparison data
        comparison_data = []
        
        for result in completed_results:
            if self.config.comparison_metric in result.metrics:
                comparison_data.append({
                    'trial_id': result.trial_id,
                    'metric_value': result.metrics[self.config.comparison_metric],
                    'params': result.params,
                    'all_metrics': result.metrics
                })
        
        # Sort by metric value
        comparison_data.sort(
            key=lambda x: x['metric_value'],
            reverse=(self.config.comparison_mode == 'max')
        )
        
        # Create comparison report
        comparison_report = {
            'best_models': comparison_data[:self.config.save_top_k],
            'worst_models': comparison_data[-3:],
            'metric_statistics': {
                'mean': np.mean([d['metric_value'] for d in comparison_data]),
                'std': np.std([d['metric_value'] for d in comparison_data]),
                'min': np.min([d['metric_value'] for d in comparison_data]),
                'max': np.max([d['metric_value'] for d in comparison_data])
            }
        }
        
        # Save comparison report
        report_file = self.experiment_dir / "model_comparison.json"
        with open(report_file, 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        return comparison_report
    
    def cleanup(self) -> None:
        """Cleanup experiment resources."""
        # Close Optuna study
        if self.study:
            self.study.stop()
        
        # Shutdown Ray
        if RAY_AVAILABLE and ray.is_initialized():
            ray.shutdown()
        
        logger.info("Experiment cleanup completed")


def create_default_hyperparameter_space() -> HyperparameterSpace:
    """Create default hyperparameter search space for fashion detection."""
    space = HyperparameterSpace()
    
    # Model hyperparameters
    space.add_categorical('model_type', ['yolo', 'classifier', 'clip'])
    
    # Training hyperparameters
    space.add_loguniform('learning_rate', 1e-5, 1e-2)
    space.add_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])
    space.add_loguniform('weight_decay', 1e-6, 1e-2)
    space.add_int('batch_size', 16, 64)
    
    # Scheduler hyperparameters
    space.add_categorical('scheduler', ['cosine', 'step', 'exponential'])
    space.add_int('warmup_epochs', 1, 10)
    
    # Regularization hyperparameters
    space.add_uniform('label_smoothing', 0.0, 0.3)
    space.add_uniform('mixup_alpha', 0.0, 0.5)
    
    # Early stopping hyperparameters
    space.add_int('early_stopping_patience', 3, 15)
    
    return space


# Export main classes
__all__ = [
    'ExperimentConfig',
    'HyperparameterSpace',
    'ExperimentResult',
    'ExperimentManager',
    'create_default_hyperparameter_space'
]