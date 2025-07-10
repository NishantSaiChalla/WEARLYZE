"""
Configuration classes for YOLOv8 fashion detection and segmentation.

This module provides configuration classes for training YOLOv8 models on fashion datasets,
including hyperparameters, model architectures, and fashion-specific settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import yaml


@dataclass
class YOLOModelConfig:
    """Configuration for YOLO model architecture."""
    
    model_size: str = "yolov8n-seg.pt"  # nano, small, medium, large, xlarge
    num_classes: int = 13  # DeepFashion2 categories
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300
    
    # Model architecture parameters
    depth_multiple: float = 1.0
    width_multiple: float = 1.0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        valid_sizes = ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", 
                      "yolov8l-seg.pt", "yolov8x-seg.pt"]
        if self.model_size not in valid_sizes:
            raise ValueError(f"Model size must be one of {valid_sizes}")
        
        if self.num_classes <= 0:
            raise ValueError("Number of classes must be positive")
        
        if not (0 < self.confidence_threshold < 1):
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        if not (0 < self.iou_threshold < 1):
            raise ValueError("IoU threshold must be between 0 and 1")


@dataclass
class YOLOTrainingConfig:
    """Configuration for YOLO training pipeline."""
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    momentum: float = 0.937
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    # Data parameters
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    test_data_path: str = "data/test"
    workers: int = 8
    
    # Augmentation parameters
    augment: bool = True
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    
    # Loss function parameters
    box_loss_gain: float = 7.5
    cls_loss_gain: float = 0.5
    seg_loss_gain: float = 1.0
    focal_loss_gamma: float = 1.5
    label_smoothing: float = 0.0
    
    # Fashion-specific parameters
    fashion_categories: List[str] = field(default_factory=lambda: [
        "short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear",
        "long_sleeved_outwear", "vest", "sling", "shorts", "trousers",
        "skirt", "short_sleeved_dress", "long_sleeved_dress", "vest_dress", "sling_dress"
    ])
    
    # Early stopping
    patience: int = 50
    min_delta: float = 0.001
    
    # Checkpointing
    save_period: int = 10
    save_best: bool = True
    
    # Distributed training
    device: str = "0"  # GPU devices
    sync_bn: bool = False
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if len(self.fashion_categories) == 0:
            raise ValueError("Fashion categories cannot be empty")


@dataclass
class YOLOEvaluationConfig:
    """Configuration for YOLO model evaluation."""
    
    # Evaluation parameters
    iou_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    confidence_threshold: float = 0.001
    max_detections: int = 300
    
    # Metrics to compute
    compute_ap: bool = True
    compute_map: bool = True
    compute_miou: bool = True
    compute_dice: bool = True
    compute_precision: bool = True
    compute_recall: bool = True
    
    # Evaluation modes
    eval_detection: bool = True
    eval_segmentation: bool = True
    
    # Output settings
    save_confusion_matrix: bool = True
    save_pr_curves: bool = True
    save_results_json: bool = True
    
    def __post_init__(self):
        """Validate evaluation configuration."""
        if not all(0 <= iou <= 1 for iou in self.iou_thresholds):
            raise ValueError("IoU thresholds must be between 0 and 1")
        
        if not (0 <= self.confidence_threshold <= 1):
            raise ValueError("Confidence threshold must be between 0 and 1")


@dataclass
class YOLOExperimentConfig:
    """Configuration for experiment tracking and logging."""
    
    # Experiment settings
    experiment_name: str = "fashion_yolo_experiment"
    project_name: str = "fashion_detection"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Logging settings
    log_level: str = "INFO"
    log_dir: str = "logs"
    tensorboard_log: bool = True
    wandb_log: bool = False
    wandb_project: str = "fashion-detection"
    wandb_entity: Optional[str] = None
    
    # Output directories
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    
    # Visualization settings
    save_train_images: bool = True
    save_val_images: bool = True
    save_pred_images: bool = True
    max_images_to_save: int = 100
    
    def __post_init__(self):
        """Create output directories if they don't exist."""
        for dir_path in [self.log_dir, self.output_dir, self.checkpoint_dir, self.results_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class YOLOConfig:
    """Main configuration class combining all YOLO configurations."""
    
    model: YOLOModelConfig = field(default_factory=YOLOModelConfig)
    training: YOLOTrainingConfig = field(default_factory=YOLOTrainingConfig)
    evaluation: YOLOEvaluationConfig = field(default_factory=YOLOEvaluationConfig)
    experiment: YOLOExperimentConfig = field(default_factory=YOLOExperimentConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "YOLOConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=YOLOModelConfig(**config_dict.get('model', {})),
            training=YOLOTrainingConfig(**config_dict.get('training', {})),
            evaluation=YOLOEvaluationConfig(**config_dict.get('evaluation', {})),
            experiment=YOLOExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'experiment': self.experiment.__dict__
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'experiment': self.experiment.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "YOLOConfig":
        """Create configuration from dictionary."""
        return cls(
            model=YOLOModelConfig(**config_dict.get('model', {})),
            training=YOLOTrainingConfig(**config_dict.get('training', {})),
            evaluation=YOLOEvaluationConfig(**config_dict.get('evaluation', {})),
            experiment=YOLOExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(self.training, key):
                    setattr(self.training, key, value)
        
        if 'evaluation' in config_dict:
            for key, value in config_dict['evaluation'].items():
                if hasattr(self.evaluation, key):
                    setattr(self.evaluation, key, value)
        
        if 'experiment' in config_dict:
            for key, value in config_dict['experiment'].items():
                if hasattr(self.experiment, key):
                    setattr(self.experiment, key, value)


def get_default_config() -> YOLOConfig:
    """Get default YOLO configuration for fashion detection."""
    return YOLOConfig()


def get_fashion_config() -> YOLOConfig:
    """Get fashion-specific YOLO configuration."""
    config = YOLOConfig()
    
    # Fashion-specific model settings
    config.model.num_classes = 13
    config.model.input_size = (640, 640)
    config.model.confidence_threshold = 0.25
    config.model.iou_threshold = 0.45
    
    # Fashion-specific training settings
    config.training.epochs = 100
    config.training.batch_size = 16
    config.training.learning_rate = 0.01
    config.training.box_loss_gain = 7.5
    config.training.cls_loss_gain = 0.5
    config.training.seg_loss_gain = 1.0
    
    # Fashion-specific evaluation settings
    config.evaluation.compute_miou = True
    config.evaluation.compute_dice = True
    config.evaluation.eval_segmentation = True
    
    # Fashion-specific experiment settings
    config.experiment.project_name = "fashion_detection"
    config.experiment.experiment_name = "fashion_yolo_segmentation"
    
    return config