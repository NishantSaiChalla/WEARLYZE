# Fashion Detection System

A comprehensive fashion detection system that combines state-of-the-art computer vision models (YOLOv8) with vision-language models (CLIP) for accurate fashion item detection, classification, and similarity search.

## Features

- **Object Detection**: YOLOv8-based detection of fashion items in images
- **Multi-modal Understanding**: CLIP integration for text-based fashion search
- **Similarity Search**: FAISS-powered vector similarity search for finding similar fashion items
- **Scalable Architecture**: Designed for production deployment with cloud storage support
- **Experiment Tracking**: Integrated with Weights & Biases for model training monitoring
- **API Support**: FastAPI-based inference server for real-time predictions

## Project Structure

```
fashion_detection/
├── config/              # Configuration files
│   ├── __init__.py
│   └── default_config.yaml
├── data/               # Data loading and preprocessing
│   └── __init__.py
├── models/             # Model architectures
│   └── __init__.py
├── training/           # Training scripts
│   └── __init__.py
├── evaluation/         # Evaluation metrics
│   └── __init__.py
├── utils/              # Utility functions
│   └── __init__.py
├── inference/          # Inference pipeline
│   └── __init__.py
├── tests/              # Unit tests
│   └── __init__.py
├── requirements.txt    # Project dependencies
├── setup.py           # Package installation
└── README.md          # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM recommended
- 50GB+ free disk space for models and data

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fashion-detection.git
cd fashion-detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### GPU Installation

For GPU support with CUDA:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install with GPU extras
pip install -e ".[gpu]"
```

## Quick Start

### 1. Configuration

Edit `config/default_config.yaml` to set up your environment:

```yaml
model:
  yolo_model: "yolov8x.pt"
  clip_model: "ViT-B/32"
  
data:
  batch_size: 32
  num_workers: 4
  
training:
  epochs: 100
  learning_rate: 0.001
```

### 2. Training

Train a new model:

```bash
fashion-train --config config/default_config.yaml
```

### 3. Inference

Run inference on an image:

```bash
fashion-detect --image path/to/image.jpg --output results/
```

### 4. API Server

Start the inference API server:

```bash
uvicorn inference.api:app --reload
```

## Dataset Structure

The system expects datasets in the following format:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Models

### YOLOv8
- Used for object detection and bounding box prediction
- Supports various model sizes: n, s, m, l, x
- Custom trained on fashion-specific datasets

### CLIP
- Used for vision-language understanding
- Enables text-based fashion search
- Provides rich feature embeddings for similarity search

### FAISS
- Efficient similarity search and clustering
- Supports both CPU and GPU implementations
- Scales to millions of fashion items

## Development

### Running Tests

```bash
pytest tests/ -v --cov=fashion_detection
```

### Code Formatting

```bash
# Format code with black
black fashion_detection/

# Check code style
flake8 fashion_detection/

# Type checking
mypy fashion_detection/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## API Documentation

Once the API server is running, visit:
- Interactive API docs: http://localhost:8000/docs
- API specification: http://localhost:8000/openapi.json

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- CLIP by OpenAI
- FAISS by Facebook Research
- PyTorch community

## Contact

For questions and support:
- Email: your.email@example.com
- Issues: https://github.com/yourusername/fashion-detection/issues