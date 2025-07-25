import os
import numpy as np
import faiss
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

def create_faiss_index(image_dir, model_path, index_path="clothing_index.faiss", file_paths_path="file_paths.npy"):
    """
    Create a FAISS index for images in a directory.

    Args:
        image_dir (str): Path to the directory containing images.
        model_path (str): Path to the pretrained CLIP model.
        index_path (str): Path to save the FAISS index.
        file_paths_path (str): Path to save the file paths of indexed images.
    """
    # Restrict threads globally
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # Load CLIP model and processor
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    print("Model loaded successfully!")

    # FAISS setup
    dimension = 512  # Embedding size for CLIP
    index = faiss.IndexFlatL2(dimension)
    file_paths = []

    # Process each image in the directory
    for file_name in os.listdir(image_dir):
        if file_name.endswith((".jpg", ".png",".jpeg")):
            image_path = os.path.join(image_dir, file_name)
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")

                with torch.no_grad():
                    embedding = model.get_image_features(**inputs)
                    embedding = embedding / (embedding.norm(dim=-1, keepdim=True) + 1e-8)  # Normalize

                # Add embedding to FAISS index
                index.add(embedding.cpu().numpy())
                file_paths.append(image_path)
                print(f"Indexed: {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    # Save the index and file paths
    faiss.write_index(index, index_path)
    np.save(file_paths_path, np.array(file_paths))
    print(f"FAISS index and file paths saved ({len(file_paths)} images).")


if __name__ == "__main__":
    create_faiss_index(image_dir="./assets", model_path="./local_clip_model")