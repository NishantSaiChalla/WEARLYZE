#!/usr/bin/env python3
"""
Optimized training script for RTX 4060 8GB VRAM.
Maximizes GPU utilization while staying within memory constraints.
"""

import os
import torch
import logging
from pathlib import Path
from ultralytics import YOLO
import yaml
import time
from torch.cuda.amp import GradScaler, autocast

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_for_rtx4060():
    """Configure optimal settings for RTX 4060 8GB."""
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set memory fraction to use most of the GPU
    torch.cuda.set_per_process_memory_fraction(0.95)
    
    logger.info("Enabled RTX 4060 optimizations")

def get_optimal_batch_size(model_size='s'):
    """Get optimal batch size for RTX 4060 based on model size."""
    # Tested values for 8GB VRAM with mixed precision
    batch_sizes = {
        'n': 32,  # nano: ~2GB VRAM
        's': 24,  # small: ~3GB VRAM  
        'm': 16,  # medium: ~5GB VRAM
        'l': 12,  # large: ~6GB VRAM
        'x': 8,   # xlarge: ~7GB VRAM
    }
    return batch_sizes.get(model_size, 16)

def train_optimized(
    data_yaml='deepfashion2.yaml',
    model_size='s',  # Use small model for better speed/accuracy tradeoff
    epochs=100,
    imgsz=640,
    device=0,
    project='runs/train',
    name='rtx4060_optimized'
):
    """Train with settings optimized for RTX 4060."""
    
    # Initialize optimizations
    optimize_for_rtx4060()
    
    # Get optimal batch size
    batch_size = get_optimal_batch_size(model_size)
    logger.info(f"Using batch size: {batch_size} for YOLOv8{model_size}")
    
    # Initialize model
    model = YOLO(f'yolov8{model_size}-seg.pt')
    
    # Training arguments optimized for RTX 4060
    args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': device,
        'project': project,
        'name': name,
        
        # Performance optimizations
        'amp': True,  # Always use mixed precision
        'cache': 'ram',  # Cache images in RAM for faster loading
        'workers': 4,  # Optimal for most systems
        'close_mosaic': 10,  # Disable mosaic for last epochs
        
        # Training optimizations
        'optimizer': 'AdamW',  # Better than SGD for small batches
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Augmentation (moderate for stability)
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        
        # Validation
        'val': True,
        'save': True,
        'save_period': 10,
        'plots': False,  # Disable plots during training for speed
        'patience': 50,
        
        # Logging
        'verbose': True,
        'exist_ok': True,
    }
    
    # Log GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        logger.info(f"CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}")
    
    # Start training
    start_time = time.time()
    
    try:
        results = model.train(**args)
        
        # Log training time
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
        
        # Validate final model
        metrics = model.val()
        
        # Handle metrics based on model type (detection vs segmentation)
        if hasattr(metrics, 'seg'):
            # Segmentation model metrics
            logger.info(f"Final Mask mAP50: {metrics.seg.map50:.3f}")
            logger.info(f"Final Mask mAP50-95: {metrics.seg.map:.3f}")
        
        if hasattr(metrics, 'box'):
            # Box detection metrics
            logger.info(f"Final Box mAP50: {metrics.box.map50:.3f}")
            logger.info(f"Final Box mAP50-95: {metrics.box.map:.3f}")
        
        return results
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error(f"Out of memory with batch size {batch_size}")
            logger.info("Try reducing batch size or using a smaller model")
            # Clear cache and try again with smaller batch
            torch.cuda.empty_cache()
            if batch_size > 8:
                logger.info(f"Retrying with batch size {batch_size // 2}")
                args['batch'] = batch_size // 2
                return model.train(**args)
        raise e

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
    
    parser = argparse.ArgumentParser(description='Optimized training for RTX 4060')
    parser.add_argument('--model', type=str, default='s', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    
    args = parser.parse_args()
    
    # Create data yaml if needed
    data_yaml = create_data_yaml()
    
    # Train with optimized settings
    train_optimized(
        data_yaml=data_yaml,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        name=f'rtx4060_yolov8{args.model}'
    )