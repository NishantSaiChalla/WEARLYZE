"""
YOLOv8 Segmentation Module for Fashion Detection.

This module provides a comprehensive wrapper around ultralytics YOLOv8 for fashion detection
and segmentation tasks, with support for custom loss functions, fine-tuning, and evaluation.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import SegmentMetrics
from ultralytics.utils.plotting import Annotator
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import select_device

from .yolo_config import YOLOConfig, YOLOModelConfig
from .yolo_utils import (
    YOLOPostProcessor, 
    YOLOVisualizer, 
    calculate_iou, 
    calculate_mask_iou, 
    calculate_dice_score
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FashionSegmentationLoss(nn.Module):
    """Custom loss function for fashion segmentation tasks."""
    
    def __init__(
        self,
        box_loss_weight: float = 7.5,
        cls_loss_weight: float = 0.5,
        seg_loss_weight: float = 1.0,
        focal_loss_gamma: float = 1.5,
        label_smoothing: float = 0.0,
        fashion_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize fashion segmentation loss.
        
        Args:
            box_loss_weight: Weight for bounding box loss
            cls_loss_weight: Weight for classification loss
            seg_loss_weight: Weight for segmentation loss
            focal_loss_gamma: Gamma parameter for focal loss
            label_smoothing: Label smoothing factor
            fashion_weights: Class-specific weights for fashion categories
        """
        super().__init__()
        self.box_loss_weight = box_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.seg_loss_weight = seg_loss_weight
        self.focal_loss_gamma = focal_loss_gamma
        self.label_smoothing = label_smoothing
        
        # Fashion-specific class weights
        self.fashion_weights = fashion_weights or {
            'short_sleeved_shirt': 1.0,
            'long_sleeved_shirt': 1.0,
            'short_sleeved_outwear': 1.2,
            'long_sleeved_outwear': 1.2,
            'vest': 1.1,
            'sling': 1.3,
            'shorts': 1.0,
            'trousers': 1.0,
            'skirt': 1.1,
            'short_sleeved_dress': 1.2,
            'long_sleeved_dress': 1.2,
            'vest_dress': 1.3,
            'sling_dress': 1.3
        }
        
        # Convert to tensor
        self.class_weights = torch.tensor(list(self.fashion_weights.values()))
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for classification.
        
        Args:
            pred: Predicted logits
            target: Ground truth labels
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.focal_loss_gamma * ce_loss
        
        return focal_loss.mean()
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss for segmentation.
        
        Args:
            pred: Predicted segmentation masks
            target: Ground truth masks
            
        Returns:
            Dice loss value
        """
        pred = torch.sigmoid(pred)
        smooth = 1e-8
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss for fashion segmentation.
        
        Args:
            predictions: Dictionary of predictions
            targets: Dictionary of ground truth targets
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Bounding box loss (IoU loss)
        if 'boxes' in predictions and 'boxes' in targets:
            box_loss = self.compute_box_loss(predictions['boxes'], targets['boxes'])
            losses['box_loss'] = box_loss * self.box_loss_weight
        
        # Classification loss (focal loss with class weights)
        if 'classes' in predictions and 'classes' in targets:
            cls_loss = self.compute_cls_loss(predictions['classes'], targets['classes'])
            losses['cls_loss'] = cls_loss * self.cls_loss_weight
        
        # Segmentation loss (combination of BCE and Dice)
        if 'masks' in predictions and 'masks' in targets:
            seg_loss = self.compute_seg_loss(predictions['masks'], targets['masks'])
            losses['seg_loss'] = seg_loss * self.seg_loss_weight
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses
    
    def compute_box_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute bounding box loss."""
        # Implementation depends on specific box format
        # This is a placeholder for IoU-based loss
        return F.smooth_l1_loss(pred_boxes, target_boxes)
    
    def compute_cls_loss(self, pred_classes: torch.Tensor, target_classes: torch.Tensor) -> torch.Tensor:
        """Compute classification loss with class weights."""
        if self.class_weights.device != pred_classes.device:
            self.class_weights = self.class_weights.to(pred_classes.device)
        
        # Apply focal loss with class weights
        return self.focal_loss(pred_classes, target_classes)
    
    def compute_seg_loss(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """Compute segmentation loss."""
        # Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(pred_masks, target_masks)
        
        # Dice loss
        dice_loss = self.dice_loss(pred_masks, target_masks)
        
        # Combine losses
        return bce_loss + dice_loss


class FashionYOLOv8(nn.Module):
    """
    Fashion-specific YOLOv8 wrapper for detection and segmentation.
    
    This class provides a comprehensive interface for fashion detection and segmentation
    using YOLOv8 with custom loss functions and evaluation metrics.
    """
    
    def __init__(
        self,
        config: YOLOConfig,
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize Fashion YOLOv8 model.
        
        Args:
            config: YOLOv8 configuration
            model_path: Path to pre-trained model
            device: Device for model (cuda/cpu)
        """
        super().__init__()
        
        self.config = config
        self.device = select_device(device)
        
        # Initialize model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            self.model = YOLO(config.model.model_size)
            logger.info(f"Initialized model with {config.model.model_size}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize custom loss function
        self.loss_fn = FashionSegmentationLoss(
            box_loss_weight=config.training.box_loss_gain,
            cls_loss_weight=config.training.cls_loss_gain,
            seg_loss_weight=config.training.seg_loss_gain,
            focal_loss_gamma=config.training.focal_loss_gamma,
            label_smoothing=config.training.label_smoothing
        )
        
        # Initialize post-processor and visualizer
        self.post_processor = YOLOPostProcessor()
        self.visualizer = YOLOVisualizer(config.training.fashion_categories)
        
        # Initialize metrics
        # Create class names dict for metrics
        class_names = {i: f"class_{i}" for i in range(config.model.num_classes)}
        self.metrics = SegmentMetrics(names=class_names)
        
        # Training state
        self.training_state = {
            'epoch': 0,
            'best_map': 0.0,
            'best_miou': 0.0,
            'train_losses': [],
            'val_losses': [],
            'val_metrics': []
        }
    
    def forward(
        self,
        x: torch.Tensor,
        augment: bool = False,
        profile: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (B, C, H, W)
            augment: Whether to use test-time augmentation
            profile: Whether to profile the model
            
        Returns:
            Dictionary of model outputs
        """
        # Use ultralytics YOLO forward pass
        results = self.model(x, augment=augment, profile=profile)
        
        # Process results into dictionary format
        outputs = {}
        
        if hasattr(results, 'boxes') and results.boxes is not None:
            outputs['boxes'] = results.boxes.xyxy
            outputs['scores'] = results.boxes.conf
            outputs['classes'] = results.boxes.cls
        
        if hasattr(results, 'masks') and results.masks is not None:
            outputs['masks'] = results.masks.data
        
        return outputs
    
    def predict(
        self,
        source: Union[str, np.ndarray, torch.Tensor],
        conf: float = 0.25,
        iou: float = 0.45,
        save: bool = False,
        save_txt: bool = False,
        save_conf: bool = False,
        show_labels: bool = True,
        show_conf: bool = True,
        max_det: int = 300,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run inference on input source.
        
        Args:
            source: Input source (image path, numpy array, or tensor)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save: Whether to save results
            save_txt: Whether to save results as text
            save_conf: Whether to save confidence scores
            show_labels: Whether to show labels
            show_conf: Whether to show confidence scores
            max_det: Maximum number of detections
            **kwargs: Additional arguments
            
        Returns:
            List of prediction dictionaries
        """
        # Run inference
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            save=save,
            save_txt=save_txt,
            save_conf=save_conf,
            show_labels=show_labels,
            show_conf=show_conf,
            max_det=max_det,
            **kwargs
        )
        
        # Process results
        predictions = []
        
        for result in results:
            pred_dict = {
                'image_path': result.path,
                'image_shape': result.orig_shape,
                'boxes': result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([]),
                'scores': result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([]),
                'class_ids': result.boxes.cls.cpu().numpy() if result.boxes is not None else np.array([]),
                'masks': result.masks.data.cpu().numpy() if result.masks is not None else None
            }
            
            predictions.append(pred_dict)
        
        return predictions
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.forward(images)
            
            # Compute loss
            loss_dict = self.loss_fn(outputs, targets)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        avg_loss = total_loss / num_batches
        
        return {
            'train_loss': avg_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
    
    def validate(
        self,
        val_loader: DataLoader,
        compute_metrics: bool = True
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            compute_metrics: Whether to compute detailed metrics
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.forward(images)
                
                # Compute loss
                loss_dict = self.loss_fn(outputs, targets)
                total_loss += loss_dict['total_loss'].item()
                
                # Collect predictions and targets for metrics
                if compute_metrics:
                    all_predictions.append(outputs)
                    all_targets.append(targets)
        
        avg_loss = total_loss / len(val_loader)
        metrics = {'val_loss': avg_loss}
        
        # Compute detailed metrics
        if compute_metrics and all_predictions:
            detailed_metrics = self.compute_metrics(all_predictions, all_targets)
            metrics.update(detailed_metrics)
        
        return metrics
    
    def compute_metrics(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: List of prediction dictionaries
            targets: List of target dictionaries
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Detection metrics (AP@0.5, AP@0.5:0.95)
        if self.config.evaluation.compute_ap:
            ap_metrics = self.compute_ap_metrics(predictions, targets)
            metrics.update(ap_metrics)
        
        # Segmentation metrics (mIoU, Dice)
        if self.config.evaluation.compute_miou or self.config.evaluation.compute_dice:
            seg_metrics = self.compute_segmentation_metrics(predictions, targets)
            metrics.update(seg_metrics)
        
        return metrics
    
    def compute_ap_metrics(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Compute Average Precision metrics."""
        # This would implement mAP calculation
        # For now, return placeholder values
        return {
            'mAP@0.5': 0.0,
            'mAP@0.5:0.95': 0.0,
            'AP_per_class': {}
        }
    
    def compute_segmentation_metrics(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Compute segmentation metrics."""
        total_iou = 0.0
        total_dice = 0.0
        num_samples = 0
        
        for pred, target in zip(predictions, targets):
            if 'masks' in pred and 'masks' in target:
                pred_masks = pred['masks'].cpu().numpy()
                target_masks = target['masks'].cpu().numpy()
                
                for pred_mask, target_mask in zip(pred_masks, target_masks):
                    iou = calculate_mask_iou(pred_mask > 0.5, target_mask > 0.5)
                    dice = calculate_dice_score(pred_mask > 0.5, target_mask > 0.5)
                    
                    total_iou += iou
                    total_dice += dice
                    num_samples += 1
        
        if num_samples > 0:
            mean_iou = total_iou / num_samples
            mean_dice = total_dice / num_samples
        else:
            mean_iou = 0.0
            mean_dice = 0.0
        
        return {
            'mIoU': mean_iou,
            'Dice': mean_dice
        }
    
    def save_checkpoint(
        self,
        checkpoint_path: str,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        is_best: bool = False
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch
            optimizer: Optimizer state
            scheduler: Scheduler state
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_state': self.training_state,
            'config': self.config.to_dict()
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = str(Path(checkpoint_path).parent / 'best_model.pth')
            torch.save(checkpoint, best_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> int:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            optimizer: Optimizer to load state
            scheduler: Scheduler to load state
            
        Returns:
            Epoch number from checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        if 'training_state' in checkpoint:
            self.training_state = checkpoint['training_state']
        
        epoch = checkpoint.get('epoch', 0)
        logger.info(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}")
        
        return epoch
    
    def fine_tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        learning_rate: float = 1e-4,
        freeze_backbone: bool = True
    ) -> Dict[str, List[float]]:
        """
        Fine-tune the model on fashion dataset.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            learning_rate: Learning rate for fine-tuning
            freeze_backbone: Whether to freeze backbone layers
            
        Returns:
            Training history
        """
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs
        )
        
        # Training loop
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_map': [],
            'val_miou': []
        }
        
        best_map = 0.0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, scheduler)
            history['train_loss'].append(train_metrics['train_loss'])
            
            # Validate
            val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_map'].append(val_metrics.get('mAP@0.5', 0.0))
            history['val_miou'].append(val_metrics.get('mIoU', 0.0))
            
            # Save best model
            current_map = val_metrics.get('mAP@0.5', 0.0)
            is_best = current_map > best_map
            
            if is_best:
                best_map = current_map
                self.training_state['best_map'] = best_map
            
            # Save checkpoint
            checkpoint_path = f"{self.config.experiment.checkpoint_dir}/epoch_{epoch+1}.pth"
            self.save_checkpoint(checkpoint_path, epoch+1, optimizer, scheduler, is_best)
            
            logger.info(f"Epoch {epoch+1} - Train Loss: {train_metrics['train_loss']:.4f}, "
                       f"Val Loss: {val_metrics['val_loss']:.4f}, "
                       f"Val mAP@0.5: {current_map:.4f}")
        
        return history
    
    def freeze_backbone(self) -> None:
        """Freeze backbone layers for fine-tuning."""
        for param in self.model.model.parameters():
            param.requires_grad = False
        
        # Unfreeze head layers
        for param in self.model.model.model[-1].parameters():
            param.requires_grad = True
        
        logger.info("Backbone layers frozen for fine-tuning")
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        
        logger.info("All layers unfrozen")
    
    def export_model(
        self,
        export_path: str,
        format: str = 'onnx',
        **kwargs
    ) -> None:
        """
        Export model to different formats.
        
        Args:
            export_path: Path to save exported model
            format: Export format (onnx, torchscript, etc.)
            **kwargs: Additional export arguments
        """
        self.model.export(format=format, **kwargs)
        logger.info(f"Model exported to {export_path} in {format} format")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        return {
            'model_size': self.config.model.model_size,
            'num_classes': self.config.model.num_classes,
            'input_size': self.config.model.input_size,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'device': str(self.device),
            'training_state': self.training_state
        }