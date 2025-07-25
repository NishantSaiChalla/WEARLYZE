#!/usr/bin/env python3
"""
Fashion Segmentation UI using Gradio.
Upload an image and get segmented clothing items with labels.
"""

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import cv2
from pathlib import Path
import logging
import colorsys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

# Define distinct colors for each class
CLASS_COLORS = {
    'short_sleeved_shirt': '#FF6B6B',
    'long_sleeved_shirt': '#4ECDC4',
    'short_sleeved_outwear': '#45B7D1',
    'long_sleeved_outwear': '#96CEB4',
    'vest': '#FECA57',
    'sling': '#DDA0DD',
    'shorts': '#98D8C8',
    'trousers': '#F7DC6F',
    'skirt': '#F8B500',
    'short_sleeved_dress': '#E056FD',
    'long_sleeved_dress': '#B83B5E',
    'vest_dress': '#6A89CC',
    'sling_dress': '#82589F'
}

def load_model(model_path=None):
    """Load the YOLO segmentation model."""
    global model
    
    if model_path is None:
        # Try to find the best model
        possible_paths = [
            '../runs/best.pt',
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                model_path = path
                break
        
        if model_path is None:
            # Look for any model in runs directory
            models = list(Path('../runs/train').glob('**/weights/best.pt'))
            if models:
                model_path = str(models[0])
            else:
                raise FileNotFoundError("No trained model found. Please train a model first.")
    
    logger.info(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    logger.info("Model loaded successfully!")
    
    return model

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_mask_overlay(image, masks, classes, confidences, alpha=0.5):
    """Create colored overlay for segmentation masks."""
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    overlay = np.zeros_like(img_array)
    img_height, img_width = img_array.shape[:2]
    
    # Get class names
    class_names = list(model.names.values())
    
    # Apply each mask with its class color
    for i, (mask, cls_idx, conf) in enumerate(zip(masks, classes, confidences)):
        if conf < 0.5:  # Skip low confidence detections
            continue
            
        # Get class name and color
        class_name = class_names[int(cls_idx)]
        color = hex_to_rgb(CLASS_COLORS.get(class_name, '#FFFFFF'))
        
        # Resize mask to match image dimensions
        mask_height, mask_width = mask.shape
        if mask_height != img_height or mask_width != img_width:
            mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        
        # Apply mask
        mask_binary = mask > 0.5
        overlay[mask_binary] = color
    
    # Blend with original image
    result = cv2.addWeighted(img_array, 1-alpha, overlay, alpha, 0)
    
    return result

def create_individual_segments(image, masks, classes, confidences, boxes):
    """Extract individual clothing items as separate images."""
    
    segments = []
    img_array = np.array(image)
    img_height, img_width = img_array.shape[:2]
    class_names = list(model.names.values())
    
    for i, (mask, cls_idx, conf, box) in enumerate(zip(masks, classes, confidences, boxes)):
        if conf < 0.5:  # Skip low confidence detections
            continue
        
        # Get class name
        class_name = class_names[int(cls_idx)]
        
        # Resize mask to match image dimensions
        mask_height, mask_width = mask.shape
        if mask_height != img_height or mask_width != img_width:
            mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        
        # Create masked image
        mask_binary = mask > 0.5
        masked_img = img_array.copy()
        masked_img[~mask_binary] = 255  # White background
        
        # Crop to bounding box
        x1, y1, x2, y2 = map(int, box)
        cropped = masked_img[y1:y2, x1:x2]
        
        # Create label
        label = f"{class_name} ({conf:.2f})"
        
        segments.append((cropped, label))
    
    return segments

def segment_fashion(image):
    """Main segmentation function for Gradio interface."""
    
    if model is None:
        return None, None, "Model not loaded. Please check if a trained model exists."
    
    try:
        # Run inference
        results = model(image, conf=0.25)[0]
        
        if results.masks is None:
            return image, None, "No clothing items detected in the image."
        
        # Extract predictions
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        
        # Create overlay image
        overlay_img = create_mask_overlay(image, masks, classes, confidences)
        overlay_pil = Image.fromarray(overlay_img.astype(np.uint8))
        
        # Add bounding boxes and labels
        draw = ImageDraw.Draw(overlay_pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        class_names = list(model.names.values())
        detected_items = []
        
        for i, (box, cls_idx, conf) in enumerate(zip(boxes, classes, confidences)):
            if conf < 0.5:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[int(cls_idx)]
            color = CLASS_COLORS.get(class_name, '#FFFFFF')
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1), label, fill='white', font=font)
            
            detected_items.append(f"- {class_name} (confidence: {conf:.2f})")
        
        # Create individual segments
        segments = create_individual_segments(image, masks, classes, confidences, boxes)
        
        # Create segment gallery
        if segments:
            segment_images = []
            for seg_img, label in segments[:6]:  # Limit to 6 segments
                seg_pil = Image.fromarray(seg_img.astype(np.uint8))
                # Add label to image
                draw = ImageDraw.Draw(seg_pil)
                draw.text((10, 10), label, fill='black', font=font)
                segment_images.append(seg_pil)
        else:
            segment_images = None
        
        # Create summary
        summary = f"Detected {len(detected_items)} clothing items:\n" + "\n".join(detected_items)
        
        return overlay_pil, segment_images, summary
        
    except Exception as e:
        logger.error(f"Error during segmentation: {e}")
        return None, None, f"Error: {str(e)}"

def create_ui():
    """Create Gradio interface."""
    
    # Load model on startup
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    # Create interface
    with gr.Blocks(title="Fashion Segmentation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 👗 Fashion Segmentation with YOLOv8
        
        Upload an image to segment and identify different clothing items.
        The model will detect and segment various fashion items including shirts, dresses, pants, skirts, and more.
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Fashion Image",
                    type="pil",
                    height=400
                )
                
                segment_btn = gr.Button("🎯 Segment Clothing", variant="primary", size="lg")
                
                examples = gr.Examples(
                    examples=[
                        "examples/fashion1.jpg",
                        "examples/fashion2.jpg",
                        "examples/fashion3.jpg"
                    ] if Path("examples").exists() else [],
                    inputs=input_image,
                    label="Example Images"
                )
            
            with gr.Column():
                output_image = gr.Image(
                    label="Segmented Result",
                    type="pil",
                    height=400
                )
                
                summary = gr.Textbox(
                    label="Detection Summary",
                    lines=5,
                    max_lines=10
                )
        
        with gr.Row():
            gr.Markdown("### 🖼️ Individual Clothing Segments")
        
        with gr.Row():
            segment_gallery = gr.Gallery(
                label="Extracted Clothing Items",
                show_label=True,
                elem_id="gallery",
                columns=3,
                rows=2,
                height="auto"
            )
        
        # Legend
        with gr.Row():
            gr.Markdown("""
            ### 🎨 Color Legend
            """)
            
            legend_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"
            for class_name, color in CLASS_COLORS.items():
                legend_html += f"""
                <div style='display: flex; align-items: center; gap: 5px;'>
                    <div style='width: 20px; height: 20px; background-color: {color};'></div>
                    <span>{class_name.replace('_', ' ').title()}</span>
                </div>
                """
            legend_html += "</div>"
            
            gr.HTML(legend_html)
        
        # Connect events
        segment_btn.click(
            fn=segment_fashion,
            inputs=input_image,
            outputs=[output_image, segment_gallery, summary]
        )
        
        input_image.change(
            fn=lambda: (None, None, ""),
            outputs=[output_image, segment_gallery, summary]
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the UI
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Create a public link
        debug=True
    )
