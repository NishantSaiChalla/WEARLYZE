# YOLOv8 Fashion Detection and Segmentation Module

A comprehensive YOLOv8-based module for fashion detection and segmentation tasks, specifically designed for the DeepFashion2 dataset with advanced training capabilities, custom loss functions, and extensive evaluation metrics.

## Features

- **Complete YOLOv8 Integration**: Seamless integration with ultralytics YOLOv8 for both detection and segmentation
- **Fashion-Specific Optimizations**: Custom loss functions and class weights tailored for fashion categories
- **Comprehensive Training Pipeline**: Full training pipeline with validation, early stopping, and learning rate scheduling
- **Advanced Evaluation Metrics**: AP@0.5, AP@0.5:0.95, mIoU, Dice coefficient, and per-class metrics
- **Distributed Training Support**: Multi-GPU and distributed training capabilities
- **Experiment Tracking**: Integration with TensorBoard and Weights & Biases
- **Data Conversion Utilities**: Convert between YOLO and DeepFashion2 formats
- **Visualization Tools**: Comprehensive visualization utilities for results analysis
- **Model Checkpointing**: Robust checkpointing and model resuming capabilities

## Module Structure

```
models/
├── __init__.py                 # Module initialization and exports
├── yolo_config.py             # Configuration classes and management
├── yolo_segmentation.py       # Main FashionYOLOv8 model implementation
├── yolo_trainer.py            # Training pipeline and utilities
├── yolo_utils.py              # Utility functions and data processing
├── example_usage.py           # Comprehensive usage examples
├── requirements.txt           # Module dependencies
└── README.md                  # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r models/requirements.txt
```

2. Ensure you have the ultralytics package installed:
```bash
pip install ultralytics
```

## Quick Start

### Basic Model Usage

```python
from models import FashionYOLOv8, get_fashion_config

# Create configuration
config = get_fashion_config()

# Initialize model
model = FashionYOLOv8(config)

# Run inference
predictions = model.predict("path/to/image.jpg")
```

### Training Pipeline

```python
from models import train_fashion_yolo, get_fashion_config
from torch.utils.data import DataLoader

# Create configuration
config = get_fashion_config()
config.training.epochs = 100
config.training.batch_size = 16

# Create data loaders (implement your dataset)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Train model
history = train_fashion_yolo(
    config=config,
    train_loader=train_loader,
    val_loader=val_loader
)
```

### Data Conversion

```python
from models import YOLODataConverter

# Convert DeepFashion2 to YOLO format
YOLODataConverter.deepfashion2_to_yolo(
    annotation_path="deepfashion2_annotations.json",
    output_dir="yolo_annotations",
    image_size=(640, 640)
)
```

## Configuration

The module uses a comprehensive configuration system with the following main components:

### YOLOModelConfig
- Model architecture settings
- Input/output configurations
- Inference parameters

### YOLOTrainingConfig
- Training hyperparameters
- Data augmentation settings
- Loss function parameters
- Fashion-specific settings

### YOLOEvaluationConfig
- Evaluation metrics configuration
- IoU thresholds
- Output settings

### YOLOExperimentConfig
- Experiment tracking setup
- Logging configuration
- Output directories

Example configuration:

```python
from models import YOLOConfig

# Create and customize configuration
config = YOLOConfig()
config.model.model_size = "yolov8s-seg.pt"
config.model.num_classes = 13
config.training.epochs = 100
config.training.batch_size = 16
config.training.learning_rate = 0.01

# Save configuration
config.to_yaml("my_config.yaml")

# Load configuration
config = YOLOConfig.from_yaml("my_config.yaml")
```

## Fashion Categories

The module supports 13 fashion categories from the DeepFashion2 dataset:

1. short_sleeved_shirt
2. long_sleeved_shirt
3. short_sleeved_outwear
4. long_sleeved_outwear
5. vest
6. sling
7. shorts
8. trousers
9. skirt
10. short_sleeved_dress
11. long_sleeved_dress
12. vest_dress
13. sling_dress

## Custom Loss Functions

The module includes a custom `FashionSegmentationLoss` that combines:

- **Bounding Box Loss**: IoU-based loss for object detection
- **Classification Loss**: Focal loss with fashion-specific class weights
- **Segmentation Loss**: Combination of Binary Cross-Entropy and Dice loss

```python
from models import FashionSegmentationLoss

loss_fn = FashionSegmentationLoss(
    box_loss_weight=7.5,
    cls_loss_weight=0.5,
    seg_loss_weight=1.0,
    focal_loss_gamma=1.5
)
```

## Evaluation Metrics

The module provides comprehensive evaluation metrics:

- **Detection Metrics**: AP@0.5, AP@0.5:0.95, mAP, Precision, Recall
- **Segmentation Metrics**: mIoU, Dice coefficient, per-class IoU
- **Visualization**: Confusion matrices, PR curves, class distributions

```python
# Compute metrics
metrics = model.compute_metrics(predictions, targets)
print(f"mAP@0.5: {metrics['mAP@0.5']:.3f}")
print(f"mIoU: {metrics['mIoU']:.3f}")
print(f"Dice: {metrics['Dice']:.3f}")
```

## Training Features

### Early Stopping
```python
from models import EarlyStopping

early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.001,
    mode='max'
)
```

### Learning Rate Scheduling
Cosine annealing learning rate scheduler with warmup support.

### Distributed Training
```bash
# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=4 train_script.py
```

### Experiment Tracking
```python
# TensorBoard integration
config.experiment.tensorboard_log = True

# Weights & Biases integration
config.experiment.wandb_log = True
config.experiment.wandb_project = "fashion-detection"
```

## Visualization

The module provides extensive visualization capabilities:

### Detection Visualization
```python
from models import YOLOVisualizer

visualizer = YOLOVisualizer(class_names)
vis_image = visualizer.visualize_detections(
    image, boxes, scores, class_ids, masks
)
```

### Training Plots
```python
# Plot training history
visualizer.plot_results_grid(images, predictions)

# Plot class distribution
visualizer.plot_class_distribution(class_counts)
```

## Model Checkpointing

```python
# Save checkpoint
model.save_checkpoint(
    "checkpoint.pth",
    epoch=10,
    optimizer=optimizer,
    is_best=True
)

# Load checkpoint
epoch = model.load_checkpoint("checkpoint.pth", optimizer)
```

## Fine-tuning

```python
# Fine-tune on fashion dataset
history = model.fine_tune(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    learning_rate=1e-4,
    freeze_backbone=True
)
```

## Model Export

```python
# Export to different formats
model.export_model("model.onnx", format="onnx")
model.export_model("model.torchscript", format="torchscript")
```

## Advanced Usage

### Custom Callbacks
```python
def custom_callback(epoch, metrics):
    print(f"Epoch {epoch}: {metrics}")

trainer = YOLOTrainer(
    config=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    callbacks=[custom_callback]
)
```

### Post-processing
```python
from models import YOLOPostProcessor

post_processor = YOLOPostProcessor()

# Apply NMS
filtered_boxes, filtered_scores, filtered_classes = post_processor.non_max_suppression(
    boxes, scores, class_ids, iou_threshold=0.45
)

# Filter small objects
filtered_results = post_processor.filter_small_objects(
    boxes, scores, class_ids, masks, min_area=100
)
```

## Performance Optimization

### Memory Optimization
- Gradient accumulation for large batch sizes
- Mixed precision training support
- Efficient data loading with multiple workers

### Speed Optimization
- Model compilation with TorchScript
- ONNX export for inference optimization
- TensorRT support for production deployment

## Testing

Run the example script to test the module:

```bash
python models/example_usage.py
```

This will demonstrate:
- Configuration management
- Data conversion
- Model creation and training
- Inference and evaluation
- Visualization capabilities

## Common Issues and Solutions

### CUDA Out of Memory
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

### Slow Training
- Increase number of data loading workers
- Use faster data augmentation
- Enable distributed training

### Poor Performance
- Adjust learning rate
- Tune loss function weights
- Increase training epochs
- Use better data augmentation

## Contributing

To contribute to this module:

1. Follow the existing code style
2. Add comprehensive docstrings
3. Include type hints
4. Add unit tests
5. Update documentation

## License

This module is part of the fashion detection system and follows the same license terms.

## Citation

If you use this module in your research, please cite:

```bibtex
@misc{fashion_yolo_module,
  title={YOLOv8 Fashion Detection and Segmentation Module},
  author={Fashion Detection Team},
  year={2024},
  howpublished={\\url{https://github.com/your-repo/fashion_detection}}
}
```