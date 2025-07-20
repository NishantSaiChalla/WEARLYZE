"""
Evaluation metrics for fashion detection and segmentation.

This module provides comprehensive metrics for evaluating object detection
and instance segmentation performance, including mAP, IoU, and Dice scores.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import warnings


def calculate_iou(box1: Union[torch.Tensor, np.ndarray], 
                 box2: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU score
    """
    if isinstance(box1, torch.Tensor):
        box1 = box1.cpu().numpy()
    if isinstance(box2, torch.Tensor):
        box2 = box2.cpu().numpy()
    
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_mask_iou(mask1: Union[torch.Tensor, np.ndarray],
                      mask2: Union[torch.Tensor, np.ndarray],
                      threshold: float = 0.5) -> float:
    """
    Calculate IoU for segmentation masks.
    
    Args:
        mask1: First mask
        mask2: Second mask
        threshold: Threshold for binary conversion
        
    Returns:
        IoU score
    """
    if isinstance(mask1, torch.Tensor):
        mask1 = mask1.cpu().numpy()
    if isinstance(mask2, torch.Tensor):
        mask2 = mask2.cpu().numpy()
    
    # Convert to binary masks
    mask1_binary = (mask1 > threshold).astype(np.uint8)
    mask2_binary = (mask2 > threshold).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1_binary, mask2_binary).sum()
    union = np.logical_or(mask1_binary, mask2_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def calculate_dice_score(mask1: Union[torch.Tensor, np.ndarray],
                        mask2: Union[torch.Tensor, np.ndarray],
                        threshold: float = 0.5,
                        smooth: float = 1e-6) -> float:
    """
    Calculate Dice coefficient for segmentation masks.
    
    Args:
        mask1: First mask
        mask2: Second mask
        threshold: Threshold for binary conversion
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice score
    """
    if isinstance(mask1, torch.Tensor):
        mask1 = mask1.cpu().numpy()
    if isinstance(mask2, torch.Tensor):
        mask2 = mask2.cpu().numpy()
    
    # Convert to binary masks
    mask1_binary = (mask1 > threshold).astype(np.uint8)
    mask2_binary = (mask2 > threshold).astype(np.uint8)
    
    # Calculate intersection
    intersection = np.logical_and(mask1_binary, mask2_binary).sum()
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (mask1_binary.sum() + mask2_binary.sum() + smooth)
    
    return dice


def calculate_ap(
    detections: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    class_id: Optional[int] = None
) -> float:
    """
    Calculate Average Precision (AP) for a specific class.
    
    Args:
        detections: List of detection dictionaries
        ground_truths: List of ground truth dictionaries
        iou_threshold: IoU threshold for positive detection
        class_id: Specific class ID (None for all classes)
        
    Returns:
        Average Precision score
    """
    # Collect all detections and ground truths for the class
    all_detections = []
    all_ground_truths = []
    
    for img_idx, (dets, gts) in enumerate(zip(detections, ground_truths)):
        # Filter detections by class
        if 'boxes' in dets and 'scores' in dets and 'labels' in dets:
            # Handle tensor dimensions properly
            boxes = dets['boxes']
            scores = dets['scores']
            labels = dets['labels']
            
            # Convert to numpy if needed and ensure proper shape
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            
            # Handle case where tensors might be empty or have unexpected shape
            if len(boxes.shape) == 1:
                # Single box case
                if len(boxes) == 4:  # Valid box
                    boxes = boxes.reshape(1, 4)
                    scores = np.array([scores]) if np.isscalar(scores) else scores.reshape(1)
                    labels = np.array([labels]) if np.isscalar(labels) else labels.reshape(1)
                else:
                    continue
            elif len(boxes.shape) == 0 or boxes.shape[0] == 0:
                # No detections
                continue
            
            # Process each detection
            for i in range(len(boxes)):
                try:
                    box = boxes[i] if len(boxes.shape) > 1 else boxes
                    score = scores[i] if len(scores.shape) > 0 and scores.shape[0] > i else scores
                    label = labels[i] if len(labels.shape) > 0 and labels.shape[0] > i else labels
                    
                    if class_id is None or int(label) == class_id:
                        all_detections.append({
                            'image_id': img_idx,
                            'box': box,
                            'score': float(score),
                            'label': int(label)
                        })
                except (IndexError, ValueError) as e:
                    # Skip problematic detections
                    continue
        
        # Filter ground truths by class
        if 'boxes' in gts and 'labels' in gts:
            # Handle tensor dimensions properly for ground truths
            gt_boxes = gts['boxes']
            gt_labels = gts['labels']
            
            # Convert to numpy if needed
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu().numpy()
            if isinstance(gt_labels, torch.Tensor):
                gt_labels = gt_labels.cpu().numpy()
            
            # Handle different shapes
            if len(gt_boxes.shape) == 1:
                if len(gt_boxes) == 4:  # Single valid box
                    gt_boxes = gt_boxes.reshape(1, 4)
                    gt_labels = np.array([gt_labels]) if np.isscalar(gt_labels) else gt_labels.reshape(1)
                else:
                    continue
            elif len(gt_boxes.shape) == 0 or gt_boxes.shape[0] == 0:
                continue
            
            # Process each ground truth
            for i in range(len(gt_boxes)):
                try:
                    box = gt_boxes[i] if len(gt_boxes.shape) > 1 else gt_boxes
                    label = gt_labels[i] if len(gt_labels.shape) > 0 and gt_labels.shape[0] > i else gt_labels
                    
                    if class_id is None or int(label) == class_id:
                        all_ground_truths.append({
                            'image_id': img_idx,
                            'box': box,
                            'label': int(label),
                            'matched': False
                        })
                except (IndexError, ValueError) as e:
                    # Skip problematic ground truths
                    continue
    
    if not all_detections:
        return 0.0
    
    # Sort detections by confidence score
    all_detections.sort(key=lambda x: x['score'], reverse=True)
    
    # Match detections to ground truths
    tp = np.zeros(len(all_detections))
    fp = np.zeros(len(all_detections))
    
    for det_idx, detection in enumerate(all_detections):
        # Find ground truths for this image
        image_gts = [gt for gt in all_ground_truths if gt['image_id'] == detection['image_id']]
        
        if not image_gts:
            fp[det_idx] = 1
            continue
        
        # Find best matching ground truth
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(image_gts):
            if gt['matched']:
                continue
            
            iou = calculate_iou(detection['box'], gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if match is good enough
        if best_iou >= iou_threshold:
            tp[det_idx] = 1
            image_gts[best_gt_idx]['matched'] = True
        else:
            fp[det_idx] = 1
    
    # Calculate precision and recall
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    
    total_positives = len([gt for gt in all_ground_truths if not class_id or gt['label'] == class_id])
    
    if total_positives == 0:
        return 0.0
    
    precision = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-6)
    recall = cumulative_tp / total_positives
    
    # Calculate AP using 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    
    return ap


def calculate_map(
    detections: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    num_classes: int = 13
) -> Dict[str, float]:
    """
    Calculate mean Average Precision (mAP) across multiple IoU thresholds.
    
    Args:
        detections: List of detection dictionaries
        ground_truths: List of ground truth dictionaries
        iou_thresholds: List of IoU thresholds
        num_classes: Number of classes
        
    Returns:
        Dictionary of mAP metrics
    """
    results = {}
    
    # Calculate AP for each class and IoU threshold
    class_aps = defaultdict(list)
    
    for iou_threshold in iou_thresholds:
        threshold_aps = []
        
        for class_id in range(1, num_classes + 1):  # Classes 1-13 for DeepFashion2
            ap = calculate_ap(detections, ground_truths, iou_threshold, class_id)
            threshold_aps.append(ap)
            class_aps[class_id].append(ap)
        
        # Store threshold-specific mAP
        if iou_threshold == 0.5:
            results['map50'] = np.mean(threshold_aps)
        elif iou_threshold == 0.75:
            results['map75'] = np.mean(threshold_aps)
    
    # Calculate mAP@0.5:0.95 (average across all thresholds)
    all_aps = []
    for class_id in range(1, num_classes + 1):
        if class_aps[class_id]:
            all_aps.append(np.mean(class_aps[class_id]))
    
    results['map50_95'] = np.mean(all_aps) if all_aps else 0.0
    
    # Per-class AP at IoU 0.5
    results['per_class_ap'] = {}
    for class_id in range(1, num_classes + 1):
        ap_50 = calculate_ap(detections, ground_truths, 0.5, class_id)
        results['per_class_ap'][class_id] = ap_50
    
    return results


def calculate_segmentation_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate segmentation metrics including mIoU and Dice scores.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truths: List of ground truth dictionaries
        iou_threshold: IoU threshold for matching instances
        
    Returns:
        Dictionary of segmentation metrics
    """
    all_ious = []
    all_dice_scores = []
    class_ious = defaultdict(list)
    class_dice = defaultdict(list)
    
    for pred, gt in zip(predictions, ground_truths):
        if 'masks' not in pred or 'masks' not in gt:
            continue
        
        pred_masks = pred['masks']
        gt_masks = gt['masks']
        pred_labels = pred.get('labels', [])
        gt_labels = gt.get('labels', [])
        
        # Convert to numpy/tensor if needed
        if isinstance(pred_masks, torch.Tensor):
            pred_masks = pred_masks.cpu()
        if isinstance(gt_masks, torch.Tensor):
            gt_masks = gt_masks.cpu()
        
        # Skip if no masks
        if len(pred_masks) == 0 or len(gt_masks) == 0:
            continue
        
        # Ensure masks have proper dimensions
        if len(pred_masks.shape) == 2:
            pred_masks = pred_masks.unsqueeze(0)
        if len(gt_masks.shape) == 2:
            gt_masks = gt_masks.unsqueeze(0)
        
        # Match predicted masks to ground truth masks
        for i in range(len(pred_masks)):
            try:
                pred_mask = pred_masks[i]
                best_iou = 0
                best_gt_idx = -1
                
                for j in range(len(gt_masks)):
                    try:
                        gt_mask = gt_masks[j]
                        iou = calculate_mask_iou(pred_mask, gt_mask)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j
                    except Exception as e:
                        continue
                
                if best_iou >= iou_threshold and best_gt_idx != -1:
                    # Calculate metrics for matched pair
                    gt_mask = gt_masks[best_gt_idx]
                    
                    iou_score = calculate_mask_iou(pred_mask, gt_mask)
                    dice_score = calculate_dice_score(pred_mask, gt_mask)
                    
                    all_ious.append(iou_score)
                    all_dice_scores.append(dice_score)
                    
                    # Per-class metrics
                    if i < len(pred_labels):
                        try:
                            class_id = int(pred_labels[i].item() if hasattr(pred_labels[i], 'item') else pred_labels[i])
                            class_ious[class_id].append(iou_score)
                            class_dice[class_id].append(dice_score)
                        except (IndexError, ValueError):
                            pass
            except Exception as e:
                continue
    
    # Calculate overall metrics
    miou = np.mean(all_ious) if all_ious else 0.0
    mean_dice = np.mean(all_dice_scores) if all_dice_scores else 0.0
    
    # Calculate per-class metrics
    per_class_iou = {}
    per_class_dice = {}
    
    for class_id in class_ious:
        per_class_iou[class_id] = np.mean(class_ious[class_id])
        per_class_dice[class_id] = np.mean(class_dice[class_id])
    
    return {
        'miou': miou,
        'dice': mean_dice,
        'per_class_iou': per_class_iou,
        'per_class_dice': per_class_dice,
        'num_matched_instances': len(all_ious)
    }


def calculate_pixel_accuracy(
    predictions: List[Dict],
    ground_truths: List[Dict]
) -> float:
    """
    Calculate pixel-wise accuracy for segmentation.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truths: List of ground truth dictionaries
        
    Returns:
        Pixel accuracy
    """
    total_correct = 0
    total_pixels = 0
    
    for pred, gt in zip(predictions, ground_truths):
        if 'masks' not in pred or 'masks' not in gt:
            continue
        
        pred_masks = pred['masks']
        gt_masks = gt['masks']
        
        # Create combined masks
        pred_combined = torch.zeros_like(pred_masks[0]) if pred_masks else None
        gt_combined = torch.zeros_like(gt_masks[0]) if gt_masks else None
        
        if pred_combined is not None and gt_combined is not None:
            # Combine all masks (take max for overlapping regions)
            for mask in pred_masks:
                pred_combined = torch.max(pred_combined, mask)
            for mask in gt_masks:
                gt_combined = torch.max(gt_combined, mask)
            
            # Convert to binary
            pred_binary = (pred_combined > 0.5).float()
            gt_binary = (gt_combined > 0.5).float()
            
            # Calculate accuracy
            correct = (pred_binary == gt_binary).sum().item()
            total = gt_binary.numel()
            
            total_correct += correct
            total_pixels += total
    
    return total_correct / total_pixels if total_pixels > 0 else 0.0


def evaluate_model_comprehensive(
    model,
    dataloader,
    device: torch.device,
    num_classes: int = 13,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5
) -> Dict[str, Union[float, Dict]]:
    """
    Comprehensive evaluation of the model on a dataset.
    
    Args:
        model: Trained model
        dataloader: Evaluation dataloader
        device: Device to run evaluation on
        num_classes: Number of classes
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for evaluation
        
    Returns:
        Dictionary of comprehensive metrics
    """
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            
            # Get predictions
            outputs = model(images)
            
            # Process predictions
            for i in range(len(images)):
                pred = {}
                if 'boxes' in outputs:
                    # Filter by confidence
                    if 'scores' in outputs:
                        keep = outputs['scores'][i] > conf_threshold
                        pred['boxes'] = outputs['boxes'][i][keep]
                        pred['scores'] = outputs['scores'][i][keep]
                        if 'classes' in outputs:
                            pred['labels'] = outputs['classes'][i][keep]
                        if 'masks' in outputs:
                            pred['masks'] = outputs['masks'][i][keep]
                    else:
                        pred['boxes'] = outputs['boxes'][i]
                        if 'classes' in outputs:
                            pred['labels'] = outputs['classes'][i]
                        if 'masks' in outputs:
                            pred['masks'] = outputs['masks'][i]
                
                all_predictions.append(pred)
                all_ground_truths.append(targets[i])
    
    # Calculate all metrics
    results = {}
    
    # Detection metrics
    detection_metrics = calculate_map(all_predictions, all_ground_truths, num_classes=num_classes)
    results.update(detection_metrics)
    
    # Segmentation metrics
    segmentation_metrics = calculate_segmentation_metrics(all_predictions, all_ground_truths)
    results.update(segmentation_metrics)
    
    # Pixel accuracy
    pixel_acc = calculate_pixel_accuracy(all_predictions, all_ground_truths)
    results['pixel_accuracy'] = pixel_acc
    
    return results


def print_evaluation_summary(metrics: Dict[str, Union[float, Dict]]) -> None:
    """
    Print a formatted summary of evaluation metrics.
    
    Args:
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    # Detection metrics
    print("\nDETECTION METRICS:")
    print("-" * 30)
    if 'map50' in metrics:
        print(f"mAP@0.5:      {metrics['map50']:.4f}")
    if 'map75' in metrics:
        print(f"mAP@0.75:     {metrics['map75']:.4f}")
    if 'map50_95' in metrics:
        print(f"mAP@0.5:0.95: {metrics['map50_95']:.4f}")
    
    # Segmentation metrics
    print("\nSEGMENTATION METRICS:")
    print("-" * 30)
    if 'miou' in metrics:
        print(f"Mean IoU:     {metrics['miou']:.4f}")
    if 'dice' in metrics:
        print(f"Dice Score:   {metrics['dice']:.4f}")
    if 'pixel_accuracy' in metrics:
        print(f"Pixel Acc:    {metrics['pixel_accuracy']:.4f}")
    
    # Per-class metrics
    if 'per_class_ap' in metrics:
        print("\nPER-CLASS AP@0.5:")
        print("-" * 30)
        for class_id, ap in metrics['per_class_ap'].items():
            print(f"Class {class_id:2d}: {ap:.4f}")
    
    if 'per_class_iou' in metrics:
        print("\nPER-CLASS IoU:")
        print("-" * 30)
        for class_id, iou in metrics['per_class_iou'].items():
            print(f"Class {class_id:2d}: {iou:.4f}")
    
    print("\n" + "="*60)