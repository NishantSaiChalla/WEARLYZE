from transformers import CLIPProcessor, CLIPModel

# Model ID from Hugging Face
model_id = "openai/clip-vit-base-patch32"

# Local directory where the model will be saved
save_path = "./local_clip_model"

# Download and save model + processor
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

model.save_pretrained(save_path)
processor.save_pretrained(save_path)

print(f"Model and processor saved to {save_path}")