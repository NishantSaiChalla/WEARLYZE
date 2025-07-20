#!/bin/bash

#======================================================================
# DeepFashion2 Training Script
# 
# This script provides an easy way to train YOLOv8 segmentation model
# on the DeepFashion2 dataset with optimal configurations.
#======================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Default values
CONFIG_FILE="configs/deepfashion2_config.yaml"
RESUME_CHECKPOINT=""
MODEL_SIZE="yolov8m-seg.pt"
BATCH_SIZE=16
EPOCHS=100
LEARNING_RATE=0.001
USE_WANDB=false
GPU_ID=0

# Help function
show_help() {
    cat << EOF
DeepFashion2 Training Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -c, --config FILE       Configuration file (default: configs/deepfashion2_config.yaml)
    -r, --resume FILE       Resume from checkpoint
    -m, --model SIZE        Model size (yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt)
    -b, --batch-size NUM    Batch size (default: 16)
    -e, --epochs NUM        Number of epochs (default: 100)
    -l, --lr NUM            Learning rate (default: 0.001)
    -w, --wandb             Enable Weights & Biases logging
    -g, --gpu ID            GPU ID to use (default: 0)
    -h, --help              Show this help message

EXAMPLES:
    # Basic training with default settings
    $0

    # Training with custom batch size and learning rate
    $0 --batch-size 32 --lr 0.0005

    # Resume training from checkpoint
    $0 --resume outputs/deepfashion2/checkpoints/best_model.pth

    # Training with Weights & Biases logging
    $0 --wandb

    # Training with larger model
    $0 --model yolov8l-seg.pt --batch-size 8

REQUIREMENTS:
    - DeepFashion2 dataset at /media/kunwar-padda/Gold/DeepFashion2/
    - CUDA-capable GPU recommended
    - Python dependencies: torch, ultralytics, opencv-python, etc.

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_SIZE="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -l|--lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -w|--wandb)
            USE_WANDB=true
            shift
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Print header
print_header "========================================"
print_header "   DeepFashion2 Training Pipeline"
print_header "========================================"

# Check if running in correct directory
if [[ ! -f "train_deepfashion2.py" ]]; then
    print_error "Please run this script from the project root directory"
    print_error "Expected to find: train_deepfashion2.py"
    exit 1
fi

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    print_error "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Check if dataset exists
DATASET_PATH="/media/kunwar-padda/Gold/DeepFashion2/deepfashion2_original_images"
if [[ ! -d "$DATASET_PATH" ]]; then
    print_error "DeepFashion2 dataset not found at: $DATASET_PATH"
    print_error "Please ensure the dataset is downloaded and extracted correctly"
    exit 1
fi

# Check for CUDA availability
print_status "Checking CUDA availability..."
if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep -q "True"; then
    CUDA_DEVICES=$(python3 -c "import torch; print('CUDA devices:', torch.cuda.device_count())")
    print_status "$CUDA_DEVICES"
    export CUDA_VISIBLE_DEVICES=$GPU_ID
else
    print_warning "CUDA not available. Training will use CPU (very slow)"
fi

# Create output directories
print_status "Creating output directories..."
mkdir -p outputs/deepfashion2/{checkpoints,logs,visualizations,results}

# Update config file with command line arguments
print_status "Updating configuration..."
TEMP_CONFIG=$(mktemp)
cp "$CONFIG_FILE" "$TEMP_CONFIG"

# Use Python to update config values
python3 << EOF
import yaml

with open('$TEMP_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

# Update values from command line
config['model']['model_size'] = '$MODEL_SIZE'
config['training']['batch_size'] = $BATCH_SIZE
config['training']['epochs'] = $EPOCHS
config['training']['learning_rate'] = $LEARNING_RATE
config['experiment']['use_wandb'] = $USE_WANDB

with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(config, f)
EOF

# Print training configuration
print_header "Training Configuration:"
echo "  Config file:    $CONFIG_FILE"
echo "  Model size:     $MODEL_SIZE"
echo "  Batch size:     $BATCH_SIZE"
echo "  Epochs:         $EPOCHS"
echo "  Learning rate:  $LEARNING_RATE"
echo "  Use W&B:        $USE_WANDB"
echo "  GPU ID:         $GPU_ID"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    echo "  Resume from:    $RESUME_CHECKPOINT"
fi

# Estimate training time
print_status "Estimating training time..."
ESTIMATED_TIME=$(python3 << EOF
import torch
batch_size = $BATCH_SIZE
epochs = $EPOCHS

# Rough estimates based on model size and hardware
time_per_epoch = {
    'yolov8n-seg.pt': 5,   # minutes
    'yolov8s-seg.pt': 8,
    'yolov8m-seg.pt': 12,
    'yolov8l-seg.pt': 18,
    'yolov8x-seg.pt': 25
}.get('$MODEL_SIZE', 12)

# Adjust for batch size (smaller batch = more time)
time_factor = 16.0 / batch_size
total_minutes = epochs * time_per_epoch * time_factor

hours = int(total_minutes // 60)
minutes = int(total_minutes % 60)

if hours > 0:
    print(f"Estimated training time: {hours}h {minutes}m")
else:
    print(f"Estimated training time: {minutes}m")
EOF
)
echo "  $ESTIMATED_TIME"

# Ask for confirmation
echo ""
read -p "Continue with training? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_status "Training cancelled"
    rm -f "$TEMP_CONFIG"
    exit 0
fi

# Install dependencies if needed
print_status "Checking dependencies..."
python3 -c "import ultralytics, torch, cv2, matplotlib, seaborn, wandb" 2>/dev/null || {
    print_warning "Some dependencies missing. Installing..."
    pip install ultralytics torch torchvision opencv-python matplotlib seaborn wandb pyyaml tqdm
}

# Set up wandb if enabled
if [[ "$USE_WANDB" == "true" ]]; then
    print_status "Setting up Weights & Biases..."
    if ! python3 -c "import wandb; wandb.login()" 2>/dev/null; then
        print_warning "W&B login failed. Please run 'wandb login' manually"
        print_warning "Continuing without W&B logging..."
        USE_WANDB=false
    fi
fi

# Start training
print_header "Starting Training..."
echo "  Output directory: outputs/deepfashion2/"
echo "  Logs will be saved to: outputs/deepfashion2/logs/"
echo ""

# Prepare training command
TRAIN_CMD="python3 train_deepfashion2.py --config $TEMP_CONFIG"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME_CHECKPOINT"
fi

# Create log file
LOG_FILE="outputs/deepfashion2/logs/training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

# Run training with logging
print_status "Training command: $TRAIN_CMD"
print_status "Logs: $LOG_FILE"
echo ""

if $TRAIN_CMD 2>&1 | tee "$LOG_FILE"; then
    print_header "Training completed successfully!"
    print_status "Results saved to: outputs/deepfashion2/"
    print_status "Best model: outputs/deepfashion2/checkpoints/best_model.pth"
    
    # Show final results
    if [[ -f "outputs/deepfashion2/results/metrics_history.json" ]]; then
        print_status "Final metrics:"
        python3 << EOF
import json
with open('outputs/deepfashion2/results/metrics_history.json', 'r') as f:
    metrics = json.load(f)

if 'val_map50' in metrics and metrics['val_map50']:
    print(f"  Best mAP@50: {max(metrics['val_map50']):.4f}")
if 'val_miou' in metrics and metrics['val_miou']:
    print(f"  Best mIoU:   {max(metrics['val_miou']):.4f}")
if 'val_dice' in metrics and metrics['val_dice']:
    print(f"  Best Dice:   {max(metrics['val_dice']):.4f}")
EOF
    fi
else
    print_error "Training failed! Check logs: $LOG_FILE"
    exit 1
fi

# Cleanup
rm -f "$TEMP_CONFIG"

print_header "========================================="
print_header "   Training Pipeline Complete!"
print_header "========================================="