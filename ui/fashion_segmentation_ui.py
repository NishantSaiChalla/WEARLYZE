#!/usr/bin/env python3
"""
Fashion Segmentation UI using Gradio + CLIP-FAISS Search
"""

import os
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
import cv2
from pathlib import Path
import logging
import faiss
from transformers import CLIPProcessor, CLIPModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
yolo_model = None
clip_model = None
clip_processor = None
faiss_index = None
faiss_file_paths = None

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
    global yolo_model
    if model_path is None:
        possible_paths = ['../runs/best.pt']
        for path in possible_paths:
            if Path(path).exists():
                model_path = path
                break
        if model_path is None:
            models = list(Path('../runs/train').glob('**/weights/best.pt'))
            if models:
                model_path = str(models[0])
            else:
                raise FileNotFoundError("No trained model found. Please train a model first.")
    logger.info(f"Loading YOLO model from: {model_path}")
    yolo_model = YOLO(model_path)
    logger.info("YOLO model loaded successfully!")
    return yolo_model

def load_clip_model(model_path="./local_clip_model", index_path="clothing_index.faiss", file_paths_path="file_paths.npy"):
    global clip_model, clip_processor, faiss_index, faiss_file_paths
    clip_model = CLIPModel.from_pretrained(model_path)
    clip_processor = CLIPProcessor.from_pretrained(model_path)
    faiss_index = faiss.read_index(index_path)
    faiss_file_paths = np.load(file_paths_path, allow_pickle=True)
    logger.info("CLIP model and FAISS index loaded.")

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_contrasting_color(bg_color):
    r, g, b = bg_color
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return (0, 0, 0) if brightness > 128 else (255, 255, 255)

def create_mask_overlay(image, masks, classes, confidences, alpha=0.5):
    img_array = np.array(image)
    overlay = np.zeros_like(img_array)
    img_height, img_width = img_array.shape[:2]
    class_names = list(yolo_model.names.values())
    for i, (mask, cls_idx, conf) in enumerate(zip(masks, classes, confidences)):
        if conf < 0.5:
            continue
        class_name = class_names[int(cls_idx)]
        color = hex_to_rgb(CLASS_COLORS.get(class_name, '#FFFFFF'))
        mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        mask_binary = mask > 0.5
        overlay[mask_binary] = color
    result = cv2.addWeighted(img_array, 1-alpha, overlay, alpha, 0)
    return result

def query_similar_images(masked_image, top_k=3):
    global clip_model, clip_processor, faiss_index, faiss_file_paths
    if clip_model is None or faiss_index is None:
        return []
    inputs = clip_processor(images=masked_image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
        embedding = embedding / (embedding.norm(dim=-1, keepdim=True) + 1e-8)
    distances, indices = faiss_index.search(embedding.cpu().numpy(), top_k)
    return [(faiss_file_paths[idx], dist) for idx, dist in zip(indices[0], distances[0])]

def segment_fashion(image):
    if yolo_model is None:
        return None, "Model not loaded.", None
    try:
        results = yolo_model(image, conf=0.25)[0]
        if results.masks is None:
            return image, "No clothing items detected.", None
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        overlay_img = create_mask_overlay(image, masks, classes, confidences)
        overlay_pil = Image.fromarray(overlay_img.astype(np.uint8))
        draw = ImageDraw.Draw(overlay_pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        class_names = list(yolo_model.names.values())
        detected_items = []
        for i, (box, cls_idx, conf) in enumerate(zip(boxes, classes, confidences)):
            if conf < 0.5:
                continue
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[int(cls_idx)]
            color = CLASS_COLORS.get(class_name, '#FFFFFF')
            color_rgb = hex_to_rgb(color)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label = f"{class_name}: {conf:.2f}"
            bbox = draw.textbbox((x1, y1), label, font=font)
            padding = 4
            bbox_padded = (bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding)
            draw.rectangle(bbox_padded, fill=color)
            text_color = get_contrasting_color(color_rgb)
            draw.text((x1, y1), label, fill=text_color, font=font)
            detected_items.append(f"- {class_name} (confidence: {conf:.2f})")
        summary = f"Detected {len(detected_items)} clothing items:\n" + "\n".join(detected_items)
        similar_image_paths = []
        if len(boxes):
            top_box = boxes[0]
            x1, y1, x2, y2 = map(int, top_box)
            top_segment = np.array(image)[y1:y2, x1:x2]
            top_segment_pil = Image.fromarray(top_segment.astype(np.uint8))
            similar_image_paths = query_similar_images(top_segment_pil)
        similar_images = []
        for path, dist in similar_image_paths:
            try:
                img = Image.open(path).convert("RGB")
                label = f"Distance: {dist:.2f}"
                similar_images.append((img, label))
            except Exception as e:
                logger.warning(f"Failed to load similar image {path}: {e}")
        return overlay_pil, summary, similar_images
    except Exception as e:
        logger.error(f"Error during segmentation: {e}")
        return None, f"Error: {str(e)}", None

def create_ui():
    try:
        load_model()
        load_clip_model()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
    with gr.Blocks(title="Fashion Segmentation", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 👗 Fashion Segmentation with YOLOv8 + CLIP
        Upload an image to segment and retrieve visually similar clothing items.
        """)
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Fashion Image", type="pil", height=400)
                segment_btn = gr.Button("🎯 Segment Clothing", variant="primary", size="lg")
                examples = gr.Examples(
                    examples=["examples/fashion1.jpg", "examples/fashion2.jpg"] if Path("examples").exists() else [],
                    inputs=input_image,
                    label="Example Images"
                )
            with gr.Column():
                output_image = gr.Image(label="Segmented Result", type="pil", height=400)
                summary = gr.Textbox(label="Detection Summary", lines=5, max_lines=10)
        with gr.Row():
            gr.Markdown("### 🔍 Top 3 Similar Items from Dataset (via CLIP + FAISS)")
        with gr.Row():
            similar_gallery = gr.Gallery(label="Top Matches", columns=3, rows=1, height="auto")
        segment_btn.click(fn=segment_fashion, inputs=input_image, outputs=[output_image, summary, similar_gallery])
        input_image.change(fn=lambda: (None, "", None), outputs=[output_image, summary, similar_gallery])
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, debug=True)
