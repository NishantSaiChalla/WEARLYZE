#!/bin/bash

# Optimized training script for RTX 4060 8GB

echo "🚀 Starting optimized training for RTX 4060 8GB"

# Set CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_BENCHMARK=1
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 4060 architecture

# Monitor GPU usage
echo "📊 GPU Status before training:"
nvidia-smi

# Clear any existing CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Run training with small model (best speed/accuracy tradeoff for RTX 4060)
echo "🏃 Starting training with YOLOv8s (recommended for RTX 4060)..."
python train_rtx4060_optimized.py \
    --model s \
    --epochs 100 \
    --imgsz 640

# For faster training but lower accuracy, use nano model:
# python train_rtx4060_optimized.py --model n --epochs 100

# For higher accuracy but slower training, use medium model:
# python train_rtx4060_optimized.py --model m --epochs 100

echo "✅ Training completed!"
echo "📊 Final GPU Status:"
nvidia-smi