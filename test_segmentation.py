#!/usr/bin/env python3
"""
Test script for evaluating YOLOv8 segmentation model performance.
Includes proper metrics handling for segmentation models.
"""

import os
import torch
import logging
from pathlib import Path
from ultralytics import YOLO
import yaml
import time
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_segmentation_model(
    model_path='runs/train/rtx4060_yolov8s/weights/best.pt',
    data_yaml='deepfashion2.yaml',
    test_dir='data/images/test',
    save_predictions=True,
    visualize_samples=5
):
    """Test segmentation model and display comprehensive metrics."""
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        logger.info("Available models:")
        for model in Path('runs/train').glob('**/weights/best.pt'):
            logger.info(f"  - {model}")
        return
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Validate on test set
    logger.info("Running validation on test set...")
    try:
        results = model.val(data=data_yaml, split='test', device=device)
        
        # Handle metrics based on model type
        if hasattr(results, 'seg'):
            # Segmentation metrics
            logger.info("\n=== Segmentation Metrics ===")
            logger.info(f"Mask mAP50: {results.seg.map50:.3f}")
            logger.info(f"Mask mAP50-95: {results.seg.map:.3f}")
            
            # Per-class metrics
            if hasattr(results.seg, 'maps'):
                logger.info("\nPer-class mAP50:")
                class_names = model.names
                for i, (name, map_score) in enumerate(zip(class_names.values(), results.seg.maps)):
                    logger.info(f"  {name}: {map_score:.3f}")
        
        if hasattr(results, 'box'):
            # Box metrics
            logger.info("\n=== Box Detection Metrics ===")
            logger.info(f"Box mAP50: {results.box.map50:.3f}")
            logger.info(f"Box mAP50-95: {results.box.map:.3f}")
            
            # Per-class metrics
            if hasattr(results.box, 'maps'):
                logger.info("\nPer-class Box mAP50:")
                class_names = model.names
                for i, (name, map_score) in enumerate(zip(class_names.values(), results.box.maps)):
                    logger.info(f"  {name}: {map_score:.3f}")
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        logger.info("Attempting alternative validation approach...")
        
        # Alternative approach - run inference on test images
        test_metrics = evaluate_on_test_images(model, test_dir, visualize_samples)
        return test_metrics

def evaluate_on_test_images(model, test_dir, num_visualize=5):
    """Evaluate model by running inference on test images."""
    
    test_path = Path(test_dir)
    if not test_path.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return None
    
    # Get test images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_images = []
    for ext in image_extensions:
        test_images.extend(test_path.glob(f'*{ext}'))
        test_images.extend(test_path.glob(f'*{ext.upper()}'))
    
    logger.info(f"Found {len(test_images)} test images")
    
    if len(test_images) == 0:
        logger.error("No test images found")
        return None
    
    # Run inference
    results_list = []
    inference_times = []
    
    logger.info("Running inference on test images...")
    for i, img_path in enumerate(test_images[:100]):  # Limit to 100 images for speed
        start_time = time.time()
        results = model(img_path, verbose=False)
        inference_times.append(time.time() - start_time)
        results_list.append(results[0])
        
        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i + 1}/{min(len(test_images), 100)} images")
    
    # Calculate inference statistics
    avg_inference_time = np.mean(inference_times)
    logger.info(f"\nAverage inference time: {avg_inference_time*1000:.2f} ms")
    logger.info(f"FPS: {1/avg_inference_time:.2f}")
    
    # Visualize sample predictions
    if num_visualize > 0:
        visualize_predictions(results_list[:num_visualize], test_images[:num_visualize], model.names)
    
    # Calculate basic metrics
    calculate_detection_stats(results_list, model.names)
    
    return results_list

def visualize_predictions(results_list, image_paths, class_names):
    """Visualize segmentation predictions."""
    
    output_dir = Path('test_results')
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"\nSaving visualizations to {output_dir}")
    
    for idx, (result, img_path) in enumerate(zip(results_list, image_paths)):
        # Plot original image with predictions
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Original image
        img = Image.open(img_path)
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Predictions
        axes[1].imshow(img)
        axes[1].set_title('Predictions')
        axes[1].axis('off')
        
        # Draw boxes and masks
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor='r', facecolor='none')
                axes[1].add_patch(rect)
                
                # Add label
                label = f'{class_names[cls]}: {conf:.2f}'
                axes[1].text(x1, y1-5, label, color='white', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
        
        # Draw masks if available
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            # Combine all masks
            combined_mask = np.zeros_like(masks[0])
            for i, mask in enumerate(masks):
                combined_mask = np.maximum(combined_mask, mask * (i + 1))
            
            # Overlay mask
            axes[1].imshow(combined_mask, alpha=0.5, cmap='jet')
        
        plt.tight_layout()
        save_path = output_dir / f'prediction_{idx}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {save_path}")

def calculate_detection_stats(results_list, class_names):
    """Calculate basic detection statistics."""
    
    total_detections = 0
    class_counts = {name: 0 for name in class_names.values()}
    confidence_scores = []
    
    for result in results_list:
        if result.boxes is not None:
            total_detections += len(result.boxes)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            
            for cls, conf in zip(classes, confs):
                class_counts[class_names[cls]] += 1
                confidence_scores.append(conf)
    
    logger.info(f"\n=== Detection Statistics ===")
    logger.info(f"Total detections: {total_detections}")
    logger.info(f"Average detections per image: {total_detections/len(results_list):.2f}")
    
    if confidence_scores:
        logger.info(f"Average confidence: {np.mean(confidence_scores):.3f}")
        logger.info(f"Min confidence: {np.min(confidence_scores):.3f}")
        logger.info(f"Max confidence: {np.max(confidence_scores):.3f}")
    
    logger.info("\nDetections per class:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            logger.info(f"  {class_name}: {count}")

def create_data_yaml():
    """Create data.yaml file for DeepFashion2."""
    data_config = {
        'path': '/media/kunwar-padda/Gold/SENG474/fashion_detection/data',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        
        # Class names from DeepFashion2
        'names': {
            0: 'short_sleeved_shirt',
            1: 'long_sleeved_shirt', 
            2: 'short_sleeved_outwear',
            3: 'long_sleeved_outwear',
            4: 'vest',
            5: 'sling',
            6: 'shorts',
            7: 'trousers',
            8: 'skirt',
            9: 'short_sleeved_dress',
            10: 'long_sleeved_dress',
            11: 'vest_dress',
            12: 'sling_dress'
        }
    }
    
    yaml_path = 'deepfashion2.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    return yaml_path

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test YOLOv8 segmentation model')
    parser.add_argument('--model', type=str, 
                       default='runs/train/rtx4060_yolov8s/weights/best.pt',
                       help='Path to model weights')
    parser.add_argument('--data', type=str, default='deepfashion2.yaml',
                       help='Path to data yaml file')
    parser.add_argument('--test-dir', type=str, default='data/images/test',
                       help='Path to test images directory')
    parser.add_argument('--visualize', type=int, default=5,
                       help='Number of predictions to visualize')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save prediction visualizations')
    
    args = parser.parse_args()
    
    # Create data yaml if needed
    if not os.path.exists(args.data):
        args.data = create_data_yaml()
    
    # Test model
    test_segmentation_model(
        model_path=args.model,
        data_yaml=args.data,
        test_dir=args.test_dir,
        save_predictions=not args.no_save,
        visualize_samples=args.visualize
    )