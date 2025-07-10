"""
Example usage of YOLOv8 Fashion Detection and Segmentation Module.

This script demonstrates how to use the comprehensive YOLOv8 segmentation module
for fashion detection tasks including training, evaluation, and inference.
"""

import os
import sys
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import from models
sys.path.append(str(Path(__file__).parent.parent))

from models import (
    YOLOConfig,
    FashionYOLOv8,
    YOLOTrainer,
    YOLODataConverter,
    YOLOVisualizer,
    get_fashion_config,
    create_trainer,
    train_fashion_yolo
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyFashionDataset(Dataset):
    """
    Dummy dataset for demonstration purposes.
    In practice, you would use your actual DeepFashion2 dataset.
    """
    
    def __init__(self, num_samples: int = 100, image_size: tuple = (640, 640)):
        self.num_samples = num_samples
        self.image_size = image_size
        self.fashion_categories = [
            "short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear",
            "long_sleeved_outwear", "vest", "sling", "shorts", "trousers",
            "skirt", "short_sleeved_dress", "long_sleeved_dress", "vest_dress", "sling_dress"
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate dummy image
        image = torch.rand(3, *self.image_size)
        
        # Generate dummy targets
        num_objects = np.random.randint(1, 5)
        targets = {
            'boxes': torch.rand(num_objects, 4) * min(self.image_size),  # Random boxes
            'classes': torch.randint(0, len(self.fashion_categories), (num_objects,)),
            'masks': torch.rand(num_objects, *self.image_size) > 0.5  # Random binary masks
        }
        
        return image, targets


def example_configuration():
    """Demonstrate configuration usage."""
    print("=== Configuration Example ===")
    
    # Get default fashion configuration
    config = get_fashion_config()
    
    # Modify configuration
    config.training.epochs = 50
    config.training.batch_size = 8
    config.training.learning_rate = 0.001
    config.model.model_size = "yolov8s-seg.pt"
    
    # Save configuration
    config.to_yaml("fashion_config.yaml")
    print("Configuration saved to fashion_config.yaml")
    
    # Load configuration
    loaded_config = YOLOConfig.from_yaml("fashion_config.yaml")
    print(f"Loaded configuration: {loaded_config.training.epochs} epochs")
    
    return config


def example_data_conversion():
    """Demonstrate data conversion utilities."""
    print("\n=== Data Conversion Example ===")
    
    # Create dummy DeepFashion2 annotation
    dummy_annotation = {
        "image_001": {
            "file_name": "image_001.jpg",
            "width": 640,
            "height": 640,
            "items": [
                {
                    "category_id": 1,
                    "category_name": "short_sleeved_shirt",
                    "bounding_box": [100, 100, 300, 400],
                    "segmentation": [100, 100, 300, 100, 300, 400, 100, 400]
                }
            ]
        }
    }
    
    # Save dummy annotation
    import json
    with open("dummy_annotation.json", "w") as f:
        json.dump(dummy_annotation, f)
    
    # Convert to YOLO format
    os.makedirs("yolo_annotations", exist_ok=True)
    YOLODataConverter.deepfashion2_to_yolo(
        "dummy_annotation.json",
        "yolo_annotations"
    )
    
    print("Converted DeepFashion2 to YOLO format")
    
    # Check converted file
    yolo_file = Path("yolo_annotations/image_001.txt")
    if yolo_file.exists():
        with open(yolo_file, 'r') as f:
            print(f"YOLO annotation: {f.read().strip()}")


def example_model_creation():
    """Demonstrate model creation and basic operations."""
    print("\n=== Model Creation Example ===")
    
    # Create configuration
    config = get_fashion_config()
    config.model.model_size = "yolov8n-seg.pt"  # Use nano model for demo
    
    # Create model
    model = FashionYOLOv8(config)
    
    # Get model info
    info = model.get_model_info()
    print(f"Model info: {info}")
    
    # Create dummy input
    dummy_input = torch.rand(1, 3, 640, 640)
    
    # Forward pass
    model.model.eval()
    with torch.no_grad():
        outputs = model.forward(dummy_input)
    
    print(f"Model outputs keys: {list(outputs.keys())}")
    
    return model


def example_training():
    """Demonstrate training pipeline."""
    print("\n=== Training Example ===")
    
    # Create configuration
    config = get_fashion_config()
    config.training.epochs = 2  # Small number for demo
    config.training.batch_size = 4
    config.model.model_size = "yolov8n-seg.pt"
    
    # Create dummy datasets
    train_dataset = DummyFashionDataset(num_samples=20)
    val_dataset = DummyFashionDataset(num_samples=10)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
    
    # Create model
    model = FashionYOLOv8(config)
    
    # Create trainer
    trainer = YOLOTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    print("Training started...")
    try:
        # Train model (this would normally take much longer)
        history = trainer.train()
        print("Training completed successfully!")
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    except Exception as e:
        print(f"Training failed: {e}")
        # This is expected with dummy data
    
    return trainer


def example_inference():
    """Demonstrate inference and visualization."""
    print("\n=== Inference Example ===")
    
    # Create configuration
    config = get_fashion_config()
    config.model.model_size = "yolov8n-seg.pt"
    
    # Create model
    model = FashionYOLOv8(config)
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Run inference
    try:
        predictions = model.predict(dummy_image, conf=0.1)  # Lower confidence for demo
        
        if predictions:
            pred = predictions[0]
            print(f"Detected {len(pred['boxes'])} objects")
            print(f"Classes: {pred['class_ids']}")
            print(f"Scores: {pred['scores']}")
            
            # Visualize results
            visualizer = YOLOVisualizer(config.training.fashion_categories)
            
            if len(pred['boxes']) > 0:
                vis_image = visualizer.visualize_detections(
                    dummy_image,
                    pred['boxes'],
                    pred['scores'],
                    pred['class_ids'],
                    pred['masks']
                )
                
                # Save visualization
                cv2.imwrite("inference_result.jpg", vis_image)
                print("Inference result saved to inference_result.jpg")
            else:
                print("No objects detected (expected with dummy data)")
        else:
            print("No predictions returned")
            
    except Exception as e:
        print(f"Inference failed: {e}")
        # This is expected with dummy data and no proper model weights


def example_evaluation():
    """Demonstrate evaluation metrics."""
    print("\n=== Evaluation Example ===")
    
    # Create configuration
    config = get_fashion_config()
    
    # Create model
    model = FashionYOLOv8(config)
    
    # Create dummy predictions and targets
    dummy_predictions = [{
        'boxes': torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]]),
        'scores': torch.tensor([0.9, 0.8]),
        'classes': torch.tensor([0, 1]),
        'masks': torch.rand(2, 640, 640) > 0.5
    }]
    
    dummy_targets = [{
        'boxes': torch.tensor([[105, 105, 205, 205], [295, 295, 395, 395]]),
        'classes': torch.tensor([0, 1]),
        'masks': torch.rand(2, 640, 640) > 0.5
    }]
    
    # Compute metrics
    metrics = model.compute_metrics(dummy_predictions, dummy_targets)
    print(f"Evaluation metrics: {metrics}")


def example_visualization():
    """Demonstrate visualization utilities."""
    print("\n=== Visualization Example ===")
    
    # Create configuration
    config = get_fashion_config()
    
    # Create visualizer
    visualizer = YOLOVisualizer(config.training.fashion_categories)
    
    # Create dummy class distribution
    class_counts = {
        "short_sleeved_shirt": 150,
        "long_sleeved_shirt": 120,
        "trousers": 200,
        "skirt": 80,
        "dress": 90
    }
    
    # Plot class distribution
    visualizer.plot_class_distribution(class_counts, "class_distribution.png")
    print("Class distribution plot saved to class_distribution.png")
    
    # Create dummy detection results
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    dummy_boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
    dummy_scores = np.array([0.9, 0.8])
    dummy_class_ids = np.array([0, 1])
    
    # Visualize detections
    vis_image = visualizer.visualize_detections(
        dummy_image,
        dummy_boxes,
        dummy_scores,
        dummy_class_ids,
        save_path="detection_visualization.jpg"
    )
    
    print("Detection visualization saved to detection_visualization.jpg")


def example_checkpointing():
    """Demonstrate model checkpointing."""
    print("\n=== Checkpointing Example ===")
    
    # Create configuration
    config = get_fashion_config()
    config.model.model_size = "yolov8n-seg.pt"
    
    # Create model
    model = FashionYOLOv8(config)
    
    # Create dummy optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    model.save_checkpoint("checkpoints/test_checkpoint.pth", 1, optimizer, is_best=True)
    print("Checkpoint saved to checkpoints/test_checkpoint.pth")
    
    # Load checkpoint
    try:
        epoch = model.load_checkpoint("checkpoints/test_checkpoint.pth", optimizer)
        print(f"Checkpoint loaded successfully, epoch: {epoch}")
    except Exception as e:
        print(f"Checkpoint loading failed: {e}")


def cleanup_demo_files():
    """Clean up files created during demo."""
    files_to_remove = [
        "fashion_config.yaml",
        "dummy_annotation.json",
        "inference_result.jpg",
        "class_distribution.png",
        "detection_visualization.jpg"
    ]
    
    dirs_to_remove = [
        "yolo_annotations",
        "checkpoints",
        "logs",
        "outputs",
        "results"
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
    
    import shutil
    for dir in dirs_to_remove:
        if os.path.exists(dir):
            shutil.rmtree(dir)


def main():
    """Main function to run all examples."""
    print("YOLOv8 Fashion Detection Module - Example Usage")
    print("=" * 50)
    
    try:
        # Run examples
        config = example_configuration()
        example_data_conversion()
        model = example_model_creation()
        trainer = example_training()
        example_inference()
        example_evaluation()
        example_visualization()
        example_checkpointing()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("Note: Some examples may show expected errors due to dummy data.")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up demo files
        cleanup_demo_files()
        print("Demo files cleaned up.")


if __name__ == "__main__":
    main()