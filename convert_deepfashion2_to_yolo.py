#!/usr/bin/env python3
"""
Convert DeepFashion2 dataset format to YOLO format.
Converts JSON annotations to YOLO .txt format and organizes directory structure.
"""

import json
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DeepFashion2 to YOLO class mapping
DEEPFASHION2_TO_YOLO = {
    1: 0,   # short_sleeved_shirt
    2: 1,   # long_sleeved_shirt
    3: 2,   # short_sleeved_outwear
    4: 3,   # long_sleeved_outwear
    5: 4,   # vest
    6: 5,   # sling
    7: 6,   # shorts
    8: 7,   # trousers
    9: 8,   # skirt
    10: 9,  # short_sleeved_dress
    11: 10, # long_sleeved_dress
    12: 11, # vest_dress
    13: 12  # sling_dress
}

def convert_deepfashion2_to_yolo_segmentation(json_path, img_width, img_height):
    """Convert DeepFashion2 JSON annotation to YOLO segmentation format."""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    yolo_labels = []
    
    # Process each item in the annotation
    for item_id, item_data in data.get('item', {}).items():
        category_id = item_data.get('category_id')
        
        # Skip if category not in our mapping
        if category_id not in DEEPFASHION2_TO_YOLO:
            continue
            
        yolo_class = DEEPFASHION2_TO_YOLO[category_id]
        
        # Get segmentation data
        segmentation = item_data.get('segmentation', [])
        if not segmentation:
            continue
            
        # Convert segmentation points to YOLO format (normalized)
        seg_points = []
        for i in range(0, len(segmentation), 2):
            if i + 1 < len(segmentation):
                x = segmentation[i] / img_width
                y = segmentation[i + 1] / img_height
                seg_points.extend([x, y])
        
        if len(seg_points) >= 6:  # Need at least 3 points for a polygon
            yolo_label = f"{yolo_class} " + " ".join(map(str, seg_points))
            yolo_labels.append(yolo_label)
    
    return yolo_labels

def convert_dataset():
    """Convert the entire DeepFashion2 dataset to YOLO format."""
    
    data_root = Path('/media/kunwar-padda/Gold/SENG474/fashion_detection/data')
    
    # Create YOLO directory structure
    yolo_root = data_root / 'yolo_format'
    yolo_root.mkdir(exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        (yolo_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Process train set
    logger.info("Converting training set...")
    train_img_dir = data_root / 'images' / 'train' / 'image'
    train_anno_dir = data_root / 'images' / 'train' / 'annos'
    
    if train_img_dir.exists() and train_anno_dir.exists():
        convert_split(train_img_dir, train_anno_dir, yolo_root, 'train')
    else:
        logger.warning(f"Train directories not found: {train_img_dir}, {train_anno_dir}")
    
    # Process validation set
    logger.info("Converting validation set...")
    val_img_dir = data_root / 'images' / 'val' / 'image'  
    val_anno_dir = data_root / 'images' / 'val' / 'annos'
    
    if val_img_dir.exists() and val_anno_dir.exists():
        convert_split(val_img_dir, val_anno_dir, yolo_root, 'val')
    else:
        logger.warning(f"Val directories not found: {val_img_dir}, {val_anno_dir}")
    
    # Process test set (images only, no annotations)
    logger.info("Converting test set...")
    test_img_dir = data_root / 'images' / 'test' / 'test' / 'image'
    
    if test_img_dir.exists():
        convert_test_split(test_img_dir, yolo_root)
    else:
        logger.warning(f"Test directory not found: {test_img_dir}")
    
    # Create data.yaml
    create_data_yaml(yolo_root)
    
    logger.info(f"Conversion complete! YOLO dataset saved to: {yolo_root}")

def convert_split(img_dir, anno_dir, yolo_root, split):
    """Convert a single split (train/val)."""
    
    converted_count = 0
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(img_dir.glob(f'*{ext}'))
        image_files.extend(img_dir.glob(f'*{ext.upper()}'))
    
    logger.info(f"Found {len(image_files)} images in {split} set")
    
    for img_path in image_files:
        # Find corresponding annotation file
        anno_path = anno_dir / f"{img_path.stem}.json"
        
        if not anno_path.exists():
            logger.warning(f"No annotation found for {img_path.name}")
            continue
        
        try:
            # Get image dimensions
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            # Convert annotation
            yolo_labels = convert_deepfashion2_to_yolo_segmentation(
                anno_path, img_width, img_height
            )
            
            # Copy image to YOLO structure
            dst_img_path = yolo_root / 'images' / split / img_path.name
            shutil.copy2(img_path, dst_img_path)
            
            # Save YOLO label file
            dst_label_path = yolo_root / 'labels' / split / f"{img_path.stem}.txt"
            with open(dst_label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))
            
            converted_count += 1
            
            if converted_count % 100 == 0:
                logger.info(f"  Converted {converted_count} images...")
                
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
    
    logger.info(f"Converted {converted_count} images for {split} set")

def convert_test_split(img_dir, yolo_root):
    """Convert test split (images only, no labels)."""
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(img_dir.glob(f'*{ext}'))
        image_files.extend(img_dir.glob(f'*{ext.upper()}'))
    
    logger.info(f"Found {len(image_files)} images in test set")
    
    converted_count = 0
    for img_path in image_files:
        try:
            # Copy image to YOLO structure
            dst_img_path = yolo_root / 'images' / 'test' / img_path.name
            shutil.copy2(img_path, dst_img_path)
            
            # Create empty label file
            dst_label_path = yolo_root / 'labels' / 'test' / f"{img_path.stem}.txt"
            dst_label_path.touch()
            
            converted_count += 1
            
            if converted_count % 1000 == 0:
                logger.info(f"  Copied {converted_count} test images...")
                
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
    
    logger.info(f"Copied {converted_count} test images")

def create_data_yaml(yolo_root):
    """Create data.yaml file for YOLO training."""
    
    yaml_content = f"""# DeepFashion2 dataset in YOLO format
path: {yolo_root}
train: images/train
val: images/val
test: images/test

# Classes (13 clothing categories)
names:
  0: short_sleeved_shirt
  1: long_sleeved_shirt
  2: short_sleeved_outwear
  3: long_sleeved_outwear
  4: vest
  5: sling
  6: shorts
  7: trousers
  8: skirt
  9: short_sleeved_dress
  10: long_sleeved_dress
  11: vest_dress
  12: sling_dress
"""
    
    with open(yolo_root / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"Created data.yaml at {yolo_root / 'data.yaml'}")

if __name__ == '__main__':
    convert_dataset()