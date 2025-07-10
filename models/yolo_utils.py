"""
Utility functions for YOLO model operations and data processing.

This module provides helper functions for YOLO model operations including data conversion,
visualization, and post-processing functions for fashion detection and segmentation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from ultralytics.utils.plotting import Annotator, colors


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLODataConverter:
    """Utility class for converting between YOLO and DeepFashion2 formats."""
    
    DEEPFASHION2_CATEGORIES = {
        1: "short_sleeved_shirt",
        2: "long_sleeved_shirt", 
        3: "short_sleeved_outwear",
        4: "long_sleeved_outwear",
        5: "vest",
        6: "sling",
        7: "shorts",
        8: "trousers",
        9: "skirt",
        10: "short_sleeved_dress",
        11: "long_sleeved_dress",
        12: "vest_dress",
        13: "sling_dress"
    }
    
    @staticmethod
    def deepfashion2_to_yolo(
        annotation_path: str,
        output_dir: str,
        image_size: Tuple[int, int] = (640, 640)
    ) -> None:
        """
        Convert DeepFashion2 annotations to YOLO format.
        
        Args:
            annotation_path: Path to DeepFashion2 annotation file
            output_dir: Directory to save YOLO format annotations
            image_size: Target image size (width, height)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        
        for image_id, image_info in annotations.items():
            yolo_annotations = []
            
            # Process each item in the image
            for item in image_info.get('items', []):
                category_id = item.get('category_id', 0)
                if category_id == 0:
                    continue
                
                # Convert category to YOLO format (0-indexed)
                yolo_class = category_id - 1
                
                # Get bounding box
                bbox = item.get('bounding_box', [])
                if len(bbox) != 4:
                    continue
                
                x1, y1, x2, y2 = bbox
                img_width = image_info.get('width', image_size[0])
                img_height = image_info.get('height', image_size[1])
                
                # Convert to YOLO format (normalized center coordinates)
                center_x = (x1 + x2) / 2 / img_width
                center_y = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Get segmentation mask if available
                segmentation = item.get('segmentation', [])
                if segmentation:
                    # Convert segmentation to YOLO format
                    normalized_seg = []
                    for i in range(0, len(segmentation), 2):
                        if i + 1 < len(segmentation):
                            x = segmentation[i] / img_width
                            y = segmentation[i + 1] / img_height
                            normalized_seg.extend([x, y])
                    
                    # Format: class_id x1 y1 x2 y2 ... (segmentation points)
                    yolo_line = f"{yolo_class} " + " ".join(map(str, normalized_seg))
                else:
                    # Format: class_id center_x center_y width height
                    yolo_line = f"{yolo_class} {center_x} {center_y} {width} {height}"
                
                yolo_annotations.append(yolo_line)
            
            # Save YOLO annotation file
            image_name = image_info.get('file_name', f'{image_id}.jpg')
            annotation_file = output_path / f"{Path(image_name).stem}.txt"
            
            with open(annotation_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
    
    @staticmethod
    def yolo_to_deepfashion2(
        yolo_dir: str,
        output_path: str,
        image_dir: str,
        image_size: Tuple[int, int] = (640, 640)
    ) -> None:
        """
        Convert YOLO annotations to DeepFashion2 format.
        
        Args:
            yolo_dir: Directory containing YOLO annotation files
            output_path: Path to save DeepFashion2 format annotations
            image_dir: Directory containing images
            image_size: Image size (width, height)
        """
        yolo_path = Path(yolo_dir)
        image_path = Path(image_dir)
        
        annotations = {}
        
        for txt_file in yolo_path.glob("*.txt"):
            image_name = txt_file.stem
            image_file = None
            
            # Find corresponding image file
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                potential_image = image_path / f"{image_name}{ext}"
                if potential_image.exists():
                    image_file = potential_image
                    break
            
            if image_file is None:
                logger.warning(f"No image found for {image_name}")
                continue
            
            # Get image dimensions
            img = Image.open(image_file)
            img_width, img_height = img.size
            
            items = []
            
            with open(txt_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0]) + 1  # Convert to 1-indexed
                    
                    if len(parts) == 5:
                        # Bounding box format
                        center_x, center_y, width, height = map(float, parts[1:5])
                        
                        # Convert to absolute coordinates
                        x1 = (center_x - width / 2) * img_width
                        y1 = (center_y - height / 2) * img_height
                        x2 = (center_x + width / 2) * img_width
                        y2 = (center_y + height / 2) * img_height
                        
                        item = {
                            'category_id': class_id,
                            'category_name': YOLODataConverter.DEEPFASHION2_CATEGORIES[class_id],
                            'bounding_box': [x1, y1, x2, y2]
                        }
                    else:
                        # Segmentation format
                        coords = list(map(float, parts[1:]))
                        segmentation = []
                        
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                x = coords[i] * img_width
                                y = coords[i + 1] * img_height
                                segmentation.extend([x, y])
                        
                        # Calculate bounding box from segmentation
                        if segmentation:
                            x_coords = segmentation[::2]
                            y_coords = segmentation[1::2]
                            x1, x2 = min(x_coords), max(x_coords)
                            y1, y2 = min(y_coords), max(y_coords)
                            
                            item = {
                                'category_id': class_id,
                                'category_name': YOLODataConverter.DEEPFASHION2_CATEGORIES[class_id],
                                'bounding_box': [x1, y1, x2, y2],
                                'segmentation': segmentation
                            }
                        else:
                            continue
                    
                    items.append(item)
            
            annotations[image_name] = {
                'file_name': image_file.name,
                'width': img_width,
                'height': img_height,
                'items': items
            }
        
        # Save annotations
        with open(output_path, 'w') as f:
            json.dump(annotations, f, indent=2)


class YOLOVisualizer:
    """Utility class for visualizing YOLO detection and segmentation results."""
    
    def __init__(self, class_names: List[str]):
        """
        Initialize visualizer with class names.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.colors = colors(len(class_names))
    
    def visualize_detections(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        masks: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detection results on image.
        
        Args:
            image: Input image (H, W, C)
            boxes: Bounding boxes (N, 4) in xyxy format
            scores: Confidence scores (N,)
            class_ids: Class IDs (N,)
            masks: Segmentation masks (N, H, W) - optional
            save_path: Path to save visualization
        
        Returns:
            Annotated image
        """
        annotator = Annotator(image.copy())
        
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            color = self.colors[int(class_id)]
            label = f"{self.class_names[int(class_id)]} {score:.2f}"
            
            # Draw bounding box
            annotator.box_label(box, label, color=color)
            
            # Draw segmentation mask if available
            if masks is not None and i < len(masks):
                mask = masks[i]
                annotator.masks([mask], colors=[color], alpha=0.3)
        
        annotated_image = annotator.result()
        
        if save_path:
            cv2.imwrite(save_path, annotated_image)
        
        return annotated_image
    
    def plot_results_grid(
        self,
        images: List[np.ndarray],
        predictions: List[Dict[str, Any]],
        save_path: Optional[str] = None,
        max_images: int = 16
    ) -> None:
        """
        Plot grid of detection results.
        
        Args:
            images: List of input images
            predictions: List of prediction dictionaries
            save_path: Path to save plot
            max_images: Maximum number of images to plot
        """
        n_images = min(len(images), max_images)
        n_cols = 4
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_images):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Get prediction
            pred = predictions[i]
            boxes = pred.get('boxes', np.array([]))
            scores = pred.get('scores', np.array([]))
            class_ids = pred.get('class_ids', np.array([]))
            masks = pred.get('masks', None)
            
            # Visualize
            vis_image = self.visualize_detections(
                images[i], boxes, scores, class_ids, masks
            )
            
            ax.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Image {i+1}")
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_images, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_class_distribution(
        self,
        class_counts: Dict[str, int],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot class distribution histogram.
        
        Args:
            class_counts: Dictionary of class names and counts
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 6))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        bars = plt.bar(classes, counts, color=self.colors[:len(classes)])
        plt.xlabel('Fashion Categories')
        plt.ylabel('Count')
        plt.title('Distribution of Fashion Categories')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class YOLOPostProcessor:
    """Post-processing utilities for YOLO outputs."""
    
    @staticmethod
    def non_max_suppression(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        iou_threshold: float = 0.45,
        score_threshold: float = 0.25
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply Non-Maximum Suppression to filter overlapping detections.
        
        Args:
            boxes: Bounding boxes (N, 4) in xyxy format
            scores: Confidence scores (N,)
            class_ids: Class IDs (N,)
            iou_threshold: IoU threshold for NMS
            score_threshold: Score threshold for filtering
        
        Returns:
            Filtered boxes, scores, and class IDs
        """
        # Filter by score threshold
        valid_mask = scores > score_threshold
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        class_ids = class_ids[valid_mask]
        
        if len(boxes) == 0:
            return boxes, scores, class_ids
        
        # Apply NMS per class
        keep_indices = []
        
        for class_id in torch.unique(class_ids):
            class_mask = class_ids == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            
            # Apply NMS
            nms_indices = nms(class_boxes, class_scores, iou_threshold)
            
            # Get original indices
            original_indices = torch.where(class_mask)[0][nms_indices]
            keep_indices.extend(original_indices.tolist())
        
        keep_indices = torch.tensor(keep_indices, dtype=torch.long)
        
        return boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]
    
    @staticmethod
    def filter_small_objects(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        min_area: float = 100.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Filter out small objects based on bounding box area.
        
        Args:
            boxes: Bounding boxes (N, 4)
            scores: Confidence scores (N,)
            class_ids: Class IDs (N,)
            masks: Segmentation masks (N, H, W) - optional
            min_area: Minimum area threshold
        
        Returns:
            Filtered boxes, scores, class IDs, and masks
        """
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Filter by area
        valid_mask = areas >= min_area
        
        filtered_boxes = boxes[valid_mask]
        filtered_scores = scores[valid_mask]
        filtered_class_ids = class_ids[valid_mask]
        filtered_masks = masks[valid_mask] if masks is not None else None
        
        return filtered_boxes, filtered_scores, filtered_class_ids, filtered_masks
    
    @staticmethod
    def resize_masks(
        masks: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Resize segmentation masks to target size.
        
        Args:
            masks: Input masks (N, H, W)
            target_size: Target size (height, width)
        
        Returns:
            Resized masks
        """
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        
        # Add batch dimension if needed
        if masks.dim() == 3:
            masks = masks.unsqueeze(0)
        
        # Resize using interpolation
        resized_masks = F.interpolate(
            masks.float(),
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Threshold to get binary masks
        resized_masks = (resized_masks > 0.5).float()
        
        return resized_masks.squeeze(0)
    
    @staticmethod
    def crop_masks_to_boxes(
        masks: torch.Tensor,
        boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Crop masks to their corresponding bounding boxes.
        
        Args:
            masks: Input masks (N, H, W)
            boxes: Bounding boxes (N, 4) in xyxy format
        
        Returns:
            Cropped masks
        """
        cropped_masks = []
        
        for mask, box in zip(masks, boxes):
            x1, y1, x2, y2 = box.int()
            
            # Ensure coordinates are within mask bounds
            h, w = mask.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Crop mask
            cropped_mask = mask[y1:y2, x1:x2]
            cropped_masks.append(cropped_mask)
        
        return cropped_masks


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate IoU between two binary masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
    
    Returns:
        IoU value
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    return intersection / union if union > 0 else 0.0


def calculate_dice_score(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Dice coefficient between two binary masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
    
    Returns:
        Dice coefficient
    """
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    
    return 2 * intersection / total if total > 0 else 0.0


def create_yolo_dataset_yaml(
    train_path: str,
    val_path: str,
    test_path: str,
    class_names: List[str],
    output_path: str
) -> None:
    """
    Create YOLO dataset configuration YAML file.
    
    Args:
        train_path: Path to training images
        val_path: Path to validation images
        test_path: Path to test images
        class_names: List of class names
        output_path: Output path for YAML file
    """
    dataset_config = {
        'path': '.',
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': len(class_names),
        'names': class_names
    }
    
    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    logger.info(f"Dataset configuration saved to {output_path}")


def load_class_names(names_file: str) -> List[str]:
    """
    Load class names from file.
    
    Args:
        names_file: Path to names file
    
    Returns:
        List of class names
    """
    with open(names_file, 'r') as f:
        return [line.strip() for line in f.readlines()]


def save_class_names(class_names: List[str], names_file: str) -> None:
    """
    Save class names to file.
    
    Args:
        class_names: List of class names
        names_file: Path to names file
    """
    with open(names_file, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    logger.info(f"Class names saved to {names_file}")