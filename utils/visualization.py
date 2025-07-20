"""
Visualization utilities for fashion detection and segmentation.

This module provides functions for visualizing predictions with colored masks,
bounding boxes, and labels for multi-instance segmentation.
"""

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Optional, Tuple, Union
import colorsys


def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate n visually distinct colors.
    
    Args:
        n: Number of colors to generate
        
    Returns:
        List of RGB color tuples
    """
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 2) * 0.3  # Alternate between 0.7 and 1.0
        value = 0.8 + (i % 3) * 0.1  # Vary brightness slightly
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    
    return colors


def visualize_predictions_with_masks(
    images: torch.Tensor,
    predictions: List[Dict],
    save_path: Optional[str] = None,
    class_names: Optional[Dict[int, str]] = None,
    mask_alpha: float = 0.5,
    show_labels: bool = True,
    show_confidence: bool = True,
    max_instances: int = 10,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Visualize predictions with colored segmentation masks.
    
    Args:
        images: Batch of images (B, C, H, W)
        predictions: List of prediction dictionaries
        save_path: Path to save visualization
        class_names: Dictionary mapping class IDs to names
        mask_alpha: Transparency for masks
        show_labels: Whether to show class labels
        show_confidence: Whether to show confidence scores
        max_instances: Maximum instances to show per image
        colors: Custom colors for each class
        
    Returns:
        Visualization array
    """
    batch_size = images.shape[0]
    
    # Convert images to numpy
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
        if images.shape[1] == 3:  # CHW to HWC
            images = images.transpose(0, 2, 3, 1)
    
    # Denormalize if needed
    if images.max() <= 1.0:
        images = (images * 255).astype(np.uint8)
    
    # Generate colors if not provided
    if colors is None:
        num_classes = len(class_names) if class_names else 20
        color_list = generate_distinct_colors(num_classes)
        colors = {i: color_list[i % len(color_list)] for i in range(num_classes)}
    
    # Create figure
    fig, axes = plt.subplots(
        2, min(batch_size, 2),
        figsize=(10 * min(batch_size, 2), 20)
    )
    
    if batch_size == 1:
        axes = axes.reshape(-1, 1)
    elif batch_size == 2:
        axes = axes.reshape(2, 2)[:, :batch_size]
    
    for img_idx in range(min(batch_size, 4)):
        # Original image
        ax_orig = axes[0, img_idx % 2]
        ax_orig.imshow(images[img_idx])
        ax_orig.set_title(f'Original Image {img_idx + 1}')
        ax_orig.axis('off')
        
        # Prediction with masks
        ax_pred = axes[1, img_idx % 2]
        
        # Create overlay image
        overlay = images[img_idx].copy()
        mask_overlay = np.zeros_like(overlay)
        
        if img_idx < len(predictions):
            pred = predictions[img_idx]
            
            # Sort by confidence score
            if 'scores' in pred and len(pred['scores']) > 0:
                sorted_indices = torch.argsort(pred['scores'], descending=True)
                sorted_indices = sorted_indices[:max_instances]
            else:
                sorted_indices = range(min(len(pred.get('boxes', [])), max_instances))
            
            # Draw masks
            if 'masks' in pred:
                for idx in sorted_indices:
                    if idx < len(pred['masks']):
                        mask = pred['masks'][idx]
                        if isinstance(mask, torch.Tensor):
                            mask = mask.cpu().numpy()
                        
                        # Get class color
                        if 'labels' in pred and idx < len(pred['labels']):
                            class_id = int(pred['labels'][idx])
                            color = colors.get(class_id, (255, 255, 255))
                        else:
                            color = colors.get(idx % len(colors), (255, 255, 255))
                        
                        # Apply mask with color
                        mask_bool = mask > 0.5
                        for c in range(3):
                            mask_overlay[:, :, c][mask_bool] = color[c]
            
            # Blend with original image
            overlay = cv2.addWeighted(overlay, 1 - mask_alpha, mask_overlay, mask_alpha, 0)
            
            # Draw bounding boxes and labels
            if 'boxes' in pred:
                for idx in sorted_indices:
                    if idx < len(pred['boxes']):
                        box = pred['boxes'][idx]
                        if isinstance(box, torch.Tensor):
                            box = box.cpu().numpy()
                        
                        x1, y1, x2, y2 = box
                        
                        # Get class info
                        if 'labels' in pred and idx < len(pred['labels']):
                            class_id = int(pred['labels'][idx])
                            class_name = class_names.get(class_id, f'Class {class_id}') if class_names else f'Class {class_id}'
                            color = colors.get(class_id, (255, 255, 255))
                        else:
                            class_name = f'Instance {idx}'
                            color = colors.get(idx % len(colors), (255, 255, 255))
                        
                        # Draw box
                        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Prepare label
                        label = class_name
                        if show_confidence and 'scores' in pred and idx < len(pred['scores']):
                            conf = float(pred['scores'][idx])
                            label = f'{label} {conf:.2f}'
                        
                        # Draw label background
                        if show_labels:
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                            cv2.rectangle(
                                overlay,
                                (int(x1), int(y1) - label_size[1] - 4),
                                (int(x1) + label_size[0], int(y1)),
                                color,
                                -1
                            )
                            
                            # Draw label text
                            cv2.putText(
                                overlay,
                                label,
                                (int(x1), int(y1) - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA
                            )
        
        ax_pred.imshow(overlay)
        ax_pred.set_title(f'Predictions - Image {img_idx + 1}')
        ax_pred.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Convert figure to array
    fig.canvas.draw()
    vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    if not save_path:
        plt.close()
    
    return vis_array


def create_instance_colormap(
    instances: List[Dict],
    height: int,
    width: int,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Create a colormap showing different instances with unique colors.
    
    Args:
        instances: List of instance dictionaries with masks
        height: Image height
        width: Image width
        colors: Custom colors for each class
        
    Returns:
        Colormap array (H, W, 3)
    """
    colormap = np.zeros((height, width, 3), dtype=np.uint8)
    
    if colors is None:
        color_list = generate_distinct_colors(len(instances))
        colors = {i: color_list[i] for i in range(len(instances))}
    
    for idx, instance in enumerate(instances):
        if 'mask' in instance:
            mask = instance['mask']
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            # Get color for this instance
            if 'category_id' in instance:
                color = colors.get(instance['category_id'], colors.get(idx, (255, 255, 255)))
            else:
                color = colors.get(idx, (255, 255, 255))
            
            # Apply color to mask region
            mask_bool = mask > 0.5
            for c in range(3):
                colormap[:, :, c][mask_bool] = color[c]
    
    return colormap


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
    title: str = 'Confusion Matrix'
) -> None:
    """
    Plot confusion matrix for classification results.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the plot
        normalize: Whether to normalize the matrix
        title: Plot title
    """
    import seaborn as sns
    
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
    )
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_segmentation_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict],
    save_path: Optional[str] = None,
    metric_type: str = 'iou'
) -> None:
    """
    Visualize segmentation metrics (IoU or Dice) for each instance.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        save_path: Path to save visualization
        metric_type: Type of metric ('iou' or 'dice')
    """
    from .metrics import calculate_mask_iou, calculate_dice_score
    
    metrics = []
    categories = []
    
    for pred, gt in zip(predictions, ground_truth):
        if 'masks' in pred and 'masks' in gt:
            pred_masks = pred['masks']
            gt_masks = gt['masks']
            
            for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks, gt_masks)):
                if metric_type == 'iou':
                    score = calculate_mask_iou(pred_mask, gt_mask)
                else:
                    score = calculate_dice_score(pred_mask, gt_mask)
                
                metrics.append(score)
                
                if 'labels' in pred and i < len(pred['labels']):
                    categories.append(int(pred['labels'][i]))
                else:
                    categories.append(0)
    
    # Create box plot by category
    plt.figure(figsize=(12, 6))
    
    unique_categories = sorted(set(categories))
    data_by_category = [
        [m for m, c in zip(metrics, categories) if c == cat]
        for cat in unique_categories
    ]
    
    plt.boxplot(data_by_category, labels=unique_categories)
    plt.xlabel('Category')
    plt.ylabel(f'{metric_type.upper()} Score')
    plt.title(f'{metric_type.upper()} Distribution by Category')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_training_dashboard(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """
    Create a comprehensive training dashboard with multiple plots.
    
    Args:
        metrics_history: Dictionary of metric histories
        save_path: Path to save the dashboard
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Loss plot
    ax1 = fig.add_subplot(gs[0, :2])
    if 'train_loss' in metrics_history:
        ax1.plot(metrics_history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in metrics_history:
        ax1.plot(metrics_history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # mAP plot
    ax2 = fig.add_subplot(gs[0, 2])
    if 'val_map50' in metrics_history:
        ax2.plot(metrics_history['val_map50'], label='mAP@50', linewidth=2, color='green')
    if 'val_map50_95' in metrics_history:
        ax2.plot(metrics_history['val_map50_95'], label='mAP@50:95', linewidth=2, color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP')
    ax2.set_title('Detection Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # IoU plot
    ax3 = fig.add_subplot(gs[1, 0])
    if 'val_miou' in metrics_history:
        ax3.plot(metrics_history['val_miou'], linewidth=2, color='blue')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('mIoU')
    ax3.set_title('Mean IoU')
    ax3.grid(True, alpha=0.3)
    
    # Dice plot
    ax4 = fig.add_subplot(gs[1, 1])
    if 'val_dice' in metrics_history:
        ax4.plot(metrics_history['val_dice'], linewidth=2, color='red')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Dice Score')
    ax4.set_title('Dice Coefficient')
    ax4.grid(True, alpha=0.3)
    
    # Learning rate plot
    ax5 = fig.add_subplot(gs[1, 2])
    if 'learning_rate' in metrics_history:
        ax5.plot(metrics_history['learning_rate'], linewidth=2, color='purple')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Learning Rate')
    ax5.set_title('Learning Rate Schedule')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    
    # Summary statistics
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Calculate summary stats
    summary_text = "Training Summary\n" + "="*50 + "\n"
    
    if 'val_map50' in metrics_history and metrics_history['val_map50']:
        best_map = max(metrics_history['val_map50'])
        best_epoch = metrics_history['val_map50'].index(best_map) + 1
        summary_text += f"Best mAP@50: {best_map:.4f} (Epoch {best_epoch})\n"
    
    if 'val_miou' in metrics_history and metrics_history['val_miou']:
        best_iou = max(metrics_history['val_miou'])
        best_epoch = metrics_history['val_miou'].index(best_iou) + 1
        summary_text += f"Best mIoU: {best_iou:.4f} (Epoch {best_epoch})\n"
    
    if 'val_dice' in metrics_history and metrics_history['val_dice']:
        best_dice = max(metrics_history['val_dice'])
        best_epoch = metrics_history['val_dice'].index(best_dice) + 1
        summary_text += f"Best Dice: {best_dice:.4f} (Epoch {best_epoch})\n"
    
    if 'train_loss' in metrics_history and metrics_history['train_loss']:
        final_loss = metrics_history['train_loss'][-1]
        summary_text += f"Final Training Loss: {final_loss:.4f}\n"
    
    ax6.text(0.1, 0.5, summary_text, fontsize=14, family='monospace',
             verticalalignment='center', transform=ax6.transAxes)
    
    plt.suptitle('Training Dashboard', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()