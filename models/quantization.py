"""
Model Quantization Utilities for Fashion Detection

This module provides utilities for quantizing fashion classification models
for efficient deployment, including post-training quantization (PTQ),
quantization-aware training (QAT), and various model compression techniques.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import copy
from pathlib import Path
import numpy as np

from .classifiers import BaseFashionClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantizationConfig:
    """Configuration class for model quantization."""
    
    def __init__(
        self,
        backend: str = 'fbgemm',
        quantization_type: str = 'static',
        calibration_dataset_size: int = 1000,
        bit_width: int = 8,
        per_channel: bool = True,
        symmetric: bool = False,
        reduce_range: bool = False
    ):
        """
        Initialize quantization configuration.
        
        Args:
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
            quantization_type: Type of quantization ('static', 'dynamic', 'qat')
            calibration_dataset_size: Size of calibration dataset for static quantization
            bit_width: Number of bits for quantization (8 or 16)
            per_channel: Whether to use per-channel quantization
            symmetric: Whether to use symmetric quantization
            reduce_range: Whether to reduce quantization range
        """
        self.backend = backend
        self.quantization_type = quantization_type
        self.calibration_dataset_size = calibration_dataset_size
        self.bit_width = bit_width
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.reduce_range = reduce_range
    
    def get_qconfig(self) -> torch.quantization.QConfig:
        """Get quantization configuration object."""
        if self.backend == 'fbgemm':
            if self.quantization_type == 'static':
                return torch.quantization.get_default_qconfig('fbgemm')
            elif self.quantization_type == 'qat':
                return torch.quantization.get_default_qat_qconfig('fbgemm')
        elif self.backend == 'qnnpack':
            if self.quantization_type == 'static':
                return torch.quantization.get_default_qconfig('qnnpack')
            elif self.quantization_type == 'qat':
                return torch.quantization.get_default_qat_qconfig('qnnpack')
        
        return torch.quantization.default_qconfig


class QuantizableModel(nn.Module):
    """Wrapper to make fashion models quantizable."""
    
    def __init__(self, model: BaseFashionClassifier):
        """
        Initialize quantizable model wrapper.
        
        Args:
            model: Fashion classification model to quantize
        """
        super().__init__()
        self.model = model
        
        # Add quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization stubs."""
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self):
        """Fuse model layers for better quantization performance."""
        # This method should be overridden for specific models
        # to fuse appropriate layers (e.g., conv-bn-relu)
        pass


class PostTrainingQuantizer:
    """Post-training quantization utilities."""
    
    def __init__(self, config: QuantizationConfig):
        """
        Initialize post-training quantizer.
        
        Args:
            config: Quantization configuration
        """
        self.config = config
        
        # Set quantization backend
        torch.backends.quantized.engine = config.backend
    
    def quantize_static(
        self,
        model: BaseFashionClassifier,
        calibration_dataloader: torch.utils.data.DataLoader,
        device: str = 'cpu'
    ) -> nn.Module:
        """
        Perform static post-training quantization.
        
        Args:
            model: Model to quantize
            calibration_dataloader: DataLoader for calibration
            device: Device to use for calibration
        
        Returns:
            Quantized model
        """
        logger.info("Starting static post-training quantization...")
        
        # Move model to CPU for quantization
        model.eval()
        model.to(device)
        
        # Create quantizable model
        quantizable_model = QuantizableModel(model)
        quantizable_model.eval()
        
        # Fuse model layers if possible
        if hasattr(quantizable_model, 'fuse_model'):
            quantizable_model.fuse_model()
        
        # Set quantization configuration
        quantizable_model.qconfig = self.config.get_qconfig()
        
        # Prepare model for quantization
        prepared_model = torch.quantization.prepare(quantizable_model, inplace=False)
        
        # Calibrate model
        self._calibrate_model(prepared_model, calibration_dataloader, device)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model, inplace=False)
        
        logger.info("Static post-training quantization completed")
        return quantized_model
    
    def quantize_dynamic(
        self,
        model: BaseFashionClassifier,
        qconfig_spec: Optional[Dict[str, Any]] = None
    ) -> nn.Module:
        """
        Perform dynamic post-training quantization.
        
        Args:
            model: Model to quantize
            qconfig_spec: Quantization configuration specification
        
        Returns:
            Quantized model
        """
        logger.info("Starting dynamic post-training quantization...")
        
        model.eval()
        model.to('cpu')
        
        # Default qconfig_spec for dynamic quantization
        if qconfig_spec is None:
            qconfig_spec = {
                nn.Linear: torch.quantization.default_dynamic_qconfig,
                nn.LSTM: torch.quantization.default_dynamic_qconfig,
                nn.GRU: torch.quantization.default_dynamic_qconfig
            }
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec,
            dtype=torch.qint8
        )
        
        logger.info("Dynamic post-training quantization completed")
        return quantized_model
    
    def _calibrate_model(
        self,
        model: nn.Module,
        calibration_dataloader: torch.utils.data.DataLoader,
        device: str = 'cpu'
    ):
        """Calibrate model for static quantization."""
        logger.info("Calibrating model for static quantization...")
        
        model.eval()
        model.to(device)
        
        with torch.no_grad():
            for i, (data, _) in enumerate(calibration_dataloader):
                if i >= self.config.calibration_dataset_size:
                    break
                
                data = data.to(device)
                _ = model(data)
        
        logger.info("Model calibration completed")


class QuantizationAwareTraining:
    """Quantization-aware training utilities."""
    
    def __init__(self, config: QuantizationConfig):
        """
        Initialize quantization-aware training.
        
        Args:
            config: Quantization configuration
        """
        self.config = config
        
        # Set quantization backend
        torch.backends.quantized.engine = config.backend
    
    def prepare_model_for_qat(
        self,
        model: BaseFashionClassifier,
        example_input: torch.Tensor
    ) -> nn.Module:
        """
        Prepare model for quantization-aware training.
        
        Args:
            model: Model to prepare for QAT
            example_input: Example input for tracing
        
        Returns:
            Model prepared for QAT
        """
        logger.info("Preparing model for quantization-aware training...")
        
        model.train()
        
        # Create quantizable model
        quantizable_model = QuantizableModel(model)
        
        # Fuse model layers if possible
        if hasattr(quantizable_model, 'fuse_model'):
            quantizable_model.fuse_model()
        
        # Set quantization configuration
        quantizable_model.qconfig = self.config.get_qconfig()
        
        # Prepare model for QAT
        prepared_model = torch.quantization.prepare_qat(quantizable_model, inplace=False)
        
        logger.info("Model prepared for quantization-aware training")
        return prepared_model
    
    def convert_qat_model(self, qat_model: nn.Module) -> nn.Module:
        """
        Convert QAT model to quantized model.
        
        Args:
            qat_model: Model trained with QAT
        
        Returns:
            Quantized model
        """
        logger.info("Converting QAT model to quantized model...")
        
        qat_model.eval()
        quantized_model = torch.quantization.convert(qat_model, inplace=False)
        
        logger.info("QAT model conversion completed")
        return quantized_model


class ModelPruner:
    """Model pruning utilities for compression."""
    
    def __init__(self, structured: bool = False):
        """
        Initialize model pruner.
        
        Args:
            structured: Whether to use structured pruning
        """
        self.structured = structured
    
    def prune_model(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.5,
        importance_scores: Optional[Dict[str, torch.Tensor]] = None
    ) -> nn.Module:
        """
        Prune model weights.
        
        Args:
            model: Model to prune
            pruning_ratio: Fraction of weights to prune
            importance_scores: Optional importance scores for pruning
        
        Returns:
            Pruned model
        """
        logger.info(f"Pruning model with ratio: {pruning_ratio}")
        
        if self.structured:
            return self._structured_prune(model, pruning_ratio, importance_scores)
        else:
            return self._unstructured_prune(model, pruning_ratio, importance_scores)
    
    def _unstructured_prune(
        self,
        model: nn.Module,
        pruning_ratio: float,
        importance_scores: Optional[Dict[str, torch.Tensor]] = None
    ) -> nn.Module:
        """Perform unstructured pruning."""
        import torch.nn.utils.prune as prune
        
        parameters_to_prune = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply magnitude-based pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )
        
        # Remove pruning reparameterization
        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')
        
        return model
    
    def _structured_prune(
        self,
        model: nn.Module,
        pruning_ratio: float,
        importance_scores: Optional[Dict[str, torch.Tensor]] = None
    ) -> nn.Module:
        """Perform structured pruning."""
        import torch.nn.utils.prune as prune
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Prune entire output channels
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=pruning_ratio,
                    n=2,
                    dim=0  # Prune output channels
                )
                prune.remove(module, 'weight')
            elif isinstance(module, nn.Linear):
                # Prune entire output neurons
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=pruning_ratio,
                    n=2,
                    dim=0  # Prune output neurons
                )
                prune.remove(module, 'weight')
        
        return model


class ModelCompressor:
    """Comprehensive model compression utilities."""
    
    def __init__(self):
        """Initialize model compressor."""
        self.pruner = ModelPruner()
        self.quantization_configs = {
            'int8': QuantizationConfig(bit_width=8),
            'int16': QuantizationConfig(bit_width=16),
            'dynamic': QuantizationConfig(quantization_type='dynamic')
        }
    
    def compress_model(
        self,
        model: BaseFashionClassifier,
        compression_config: Dict[str, Any],
        calibration_dataloader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Compress model using multiple techniques.
        
        Args:
            model: Model to compress
            compression_config: Configuration for compression
            calibration_dataloader: DataLoader for calibration
        
        Returns:
            Dictionary containing compressed models and metrics
        """
        logger.info("Starting model compression...")
        
        results = {}
        original_size = self._get_model_size(model)
        
        # Apply pruning if requested
        if compression_config.get('pruning', {}).get('enabled', False):
            pruning_config = compression_config['pruning']
            pruned_model = self.pruner.prune_model(
                copy.deepcopy(model),
                pruning_ratio=pruning_config.get('ratio', 0.5),
                importance_scores=pruning_config.get('importance_scores')
            )
            
            results['pruned'] = {
                'model': pruned_model,
                'size': self._get_model_size(pruned_model),
                'compression_ratio': original_size / self._get_model_size(pruned_model)
            }
        
        # Apply quantization if requested
        if compression_config.get('quantization', {}).get('enabled', False):
            quant_config = compression_config['quantization']
            quant_type = quant_config.get('type', 'dynamic')
            
            if quant_type == 'static' and calibration_dataloader is not None:
                quantizer = PostTrainingQuantizer(self.quantization_configs['int8'])
                quantized_model = quantizer.quantize_static(
                    copy.deepcopy(model),
                    calibration_dataloader
                )
            else:
                quantizer = PostTrainingQuantizer(self.quantization_configs['dynamic'])
                quantized_model = quantizer.quantize_dynamic(copy.deepcopy(model))
            
            results['quantized'] = {
                'model': quantized_model,
                'size': self._get_model_size(quantized_model),
                'compression_ratio': original_size / self._get_model_size(quantized_model)
            }
        
        # Apply both pruning and quantization if requested
        if (compression_config.get('pruning', {}).get('enabled', False) and 
            compression_config.get('quantization', {}).get('enabled', False)):
            
            # First prune, then quantize
            pruned_model = self.pruner.prune_model(
                copy.deepcopy(model),
                pruning_ratio=compression_config['pruning'].get('ratio', 0.5)
            )
            
            if compression_config['quantization'].get('type', 'dynamic') == 'static':
                quantizer = PostTrainingQuantizer(self.quantization_configs['int8'])
                if calibration_dataloader is not None:
                    combined_model = quantizer.quantize_static(pruned_model, calibration_dataloader)
                else:
                    combined_model = quantizer.quantize_dynamic(pruned_model)
            else:
                quantizer = PostTrainingQuantizer(self.quantization_configs['dynamic'])
                combined_model = quantizer.quantize_dynamic(pruned_model)
            
            results['pruned_quantized'] = {
                'model': combined_model,
                'size': self._get_model_size(combined_model),
                'compression_ratio': original_size / self._get_model_size(combined_model)
            }
        
        logger.info("Model compression completed")
        return results
    
    def _get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes."""
        total_size = 0
        for param in model.parameters():
            total_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            total_size += buffer.nelement() * buffer.element_size()
        
        return total_size
    
    def benchmark_compressed_models(
        self,
        compressed_models: Dict[str, Any],
        test_dataloader: torch.utils.data.DataLoader,
        device: str = 'cpu'
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark compressed models.
        
        Args:
            compressed_models: Dictionary of compressed models
            test_dataloader: DataLoader for testing
            device: Device to use for benchmarking
        
        Returns:
            Dictionary of benchmark results
        """
        logger.info("Benchmarking compressed models...")
        
        results = {}
        
        for model_name, model_info in compressed_models.items():
            model = model_info['model']
            model.eval()
            model.to(device)
            
            # Measure inference time
            inference_times = []
            correct_predictions = 0
            total_predictions = 0
            
            with torch.no_grad():
                for data, target in test_dataloader:
                    data, target = data.to(device), target.to(device)
                    
                    # Measure inference time
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    
                    start_time.record()
                    output = model(data)
                    end_time.record()
                    
                    torch.cuda.synchronize()
                    inference_time = start_time.elapsed_time(end_time)
                    inference_times.append(inference_time)
                    
                    # Calculate accuracy
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    pred = output.argmax(dim=1)
                    correct_predictions += (pred == target).sum().item()
                    total_predictions += target.size(0)
            
            accuracy = correct_predictions / total_predictions
            avg_inference_time = np.mean(inference_times)
            
            results[model_name] = {
                'accuracy': accuracy,
                'avg_inference_time_ms': avg_inference_time,
                'model_size_mb': model_info['size'] / (1024 * 1024),
                'compression_ratio': model_info.get('compression_ratio', 1.0)
            }
        
        logger.info("Benchmarking completed")
        return results


def save_compressed_model(
    model: nn.Module,
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save compressed model with metadata.
    
    Args:
        model: Compressed model to save
        save_path: Path to save the model
        metadata: Optional metadata to save with the model
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'metadata': metadata or {}
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Compressed model saved to {save_path}")


def load_compressed_model(
    load_path: str,
    model_class: type,
    device: str = 'cpu'
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load compressed model.
    
    Args:
        load_path: Path to load the model from
        model_class: Model class to instantiate
        device: Device to load the model on
    
    Returns:
        Tuple of loaded model and metadata
    """
    checkpoint = torch.load(load_path, map_location=device)
    
    # This is a simplified version - you might need to adjust based on your model structure
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    metadata = checkpoint.get('metadata', {})
    
    logger.info(f"Compressed model loaded from {load_path}")
    return model, metadata


# Export all classes and functions
__all__ = [
    'QuantizationConfig',
    'QuantizableModel',
    'PostTrainingQuantizer',
    'QuantizationAwareTraining',
    'ModelPruner',
    'ModelCompressor',
    'save_compressed_model',
    'load_compressed_model'
]