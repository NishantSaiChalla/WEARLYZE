#!/bin/bash

# Example training script for Fashion Detection System

# Set environment variables for S3 (if using S3)
# export AWS_ACCESS_KEY_ID="your_access_key"
# export AWS_SECRET_ACCESS_KEY="your_secret_key"
# export AWS_DEFAULT_REGION="us-east-1"

# Training with S3 data
# echo "Training YOLOv8 model with S3 data..."
# python train.py \
#     --model yolo \
#     --dataset deepfashion2 \
#     --data-root s3://your-bucket/deepfashion2 \
#     --use-s3 \
#     --s3-bucket your-bucket \
#     --s3-region us-east-1 \
#     --epochs 100 \
#     --batch-size 32 \
#     --lr 0.001 \
#     --workers 8 \
#     --pretrained \
#     --mixed-precision \
#     --balanced-sampling \
#     --experiment-name yolo_fashion_s3 \
#     --wandb-project fashion-detection \
#     --output-dir outputs/yolo_s3

Training with local data
echo "Training YOLOv8 model with local data..."
python train.py \
    --model yolo \
    --dataset deepfashion2 \
    --data-root /1000\ images/ \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --workers 8 \
    --pretrained \
    --mixed-precision \
    --experiment-name yolo_fashion_local \
    --output-dir outputs/yolo_local

# Distributed training (multi-GPU)
# echo "Training with distributed data parallel..."
# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     train.py \
#     --model yolo \
#     --dataset deepfashion2 \
#     --data-root s3://your-bucket/deepfashion2 \
#     --use-s3 \
#     --s3-bucket your-bucket \
#     --distributed \
#     --epochs 100 \
#     --batch-size 128 \
#     --experiment-name yolo_fashion_distributed

# Resume training from checkpoint
# echo "Resuming training from checkpoint..."
# python train.py \
#     --model yolo \
#     --dataset deepfashion2 \
#     --data-root s3://your-bucket/deepfashion2 \
#     --use-s3 \
#     --s3-bucket your-bucket \
#     --checkpoint outputs/yolo_s3/checkpoints/best.pth \
#     --epochs 50 \
#     --experiment-name yolo_fashion_resumed