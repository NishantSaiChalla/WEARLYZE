# Fashion Segmentation UI

Interactive web interface for fashion item segmentation using YOLOv8.

## Features

- Upload fashion images to get segmented clothing items
- Visual overlay with colored masks for each clothing type
- Individual segmented pieces extracted as separate images
- Confidence scores and class labels
- Support for 13 clothing categories

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have a trained YOLOv8 segmentation model in the parent directory under `runs/train/`

## Usage

Run the interface:
```bash
python fashion_segmentation_ui.py
```

The interface will be available at `http://localhost:7860`

## Supported Clothing Categories

- Short-sleeved shirt
- Long-sleeved shirt  
- Short-sleeved outwear
- Long-sleeved outwear
- Vest
- Sling
- Shorts
- Trousers
- Skirt
- Short-sleeved dress
- Long-sleeved dress
- Vest dress
- Sling dress

## Color Legend

Each clothing type is assigned a unique color in the segmentation overlay for easy identification.