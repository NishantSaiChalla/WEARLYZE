#!/usr/bin/env python3
"""
Test script for evaluating the trained fashion detection model.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class TestFashionModel:
    def __init__(self, model_path, num_classes=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Define transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Class names (you can customize these based on your actual classes)
        self.class_names = [f'Class_{i}' for i in range(num_classes)]
    
    def _load_model(self, model_path):
        """Load the trained model."""
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        return model
    
    def predict_single_image(self, image_path):
        """Predict class for a single image."""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()
    
    def test_on_folder(self, test_folder, sample_size=None):
        """Test model on all images in a folder."""
        test_path = Path(test_folder)
        image_files = sorted([f for f in test_path.glob('*.jpg') if f.is_file()])
        
        if sample_size and sample_size < len(image_files):
            import random
            image_files = random.sample(image_files, sample_size)
        
        predictions = []
        confidences = []
        true_labels = []
        
        print(f"\nTesting on {len(image_files)} images...")
        
        for idx, img_path in enumerate(tqdm(image_files)):
            pred_class, conf, _ = self.predict_single_image(img_path)
            predictions.append(pred_class)
            confidences.append(conf)
            # For testing, we'll use the image index modulo 10 as true label
            # In real scenario, you'd load actual labels
            true_labels.append(idx % self.num_classes)
        
        return predictions, confidences, true_labels, image_files
    
    def visualize_predictions(self, image_files, predictions, confidences, num_samples=9):
        """Visualize sample predictions."""
        num_samples = min(num_samples, len(image_files))
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        indices = np.random.choice(len(image_files), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            img = Image.open(image_files[idx])
            axes[i].imshow(img)
            axes[i].set_title(f'Predicted: Class_{predictions[idx]}\nConfidence: {confidences[idx]:.2f}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('prediction_samples.png')
        print("Sample predictions saved to 'prediction_samples.png'")
    
    def generate_report(self, predictions, true_labels, confidences):
        """Generate classification report and confusion matrix."""
        # Classification report
        report = classification_report(true_labels, predictions, 
                                     target_names=self.class_names,
                                     output_dict=True)
        
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, 
                                  target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved to 'confusion_matrix.png'")
        
        # Calculate overall metrics
        accuracy = (np.array(predictions) == np.array(true_labels)).mean()
        avg_confidence = np.mean(confidences)
        
        return {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'report': report
        }

def main():
    parser = argparse.ArgumentParser(description='Test fashion detection model')
    parser.add_argument('--model-path', type=str, default='best_model.pth',
                        help='Path to the trained model')
    parser.add_argument('--test-folder', type=str, default='1000 images',
                        help='Folder containing test images')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of classes')
    parser.add_argument('--sample-size', type=int, default=100,
                        help='Number of images to test (None for all)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize sample predictions')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = TestFashionModel(args.model_path, args.num_classes)
    
    # Test on folder
    predictions, confidences, true_labels, image_files = tester.test_on_folder(
        args.test_folder, args.sample_size)
    
    # Generate report
    metrics = tester.generate_report(predictions, true_labels, confidences)
    
    print(f"\n{'='*50}")
    print(f"Overall Test Results:")
    print(f"{'='*50}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Average Confidence: {metrics['avg_confidence']:.2%}")
    print(f"Total Images Tested: {len(predictions)}")
    
    # Visualize if requested
    if args.visualize:
        tester.visualize_predictions(image_files, predictions, confidences)
    
    # Test single image example
    print(f"\n{'='*50}")
    print("Single Image Test Example:")
    print(f"{'='*50}")
    
    sample_image = image_files[0]
    pred_class, conf, probs = tester.predict_single_image(sample_image)
    
    print(f"Image: {sample_image.name}")
    print(f"Predicted Class: Class_{pred_class}")
    print(f"Confidence: {conf:.2%}")
    print("\nTop 3 Predictions:")
    top3_indices = np.argsort(probs)[-3:][::-1]
    for idx in top3_indices:
        print(f"  Class_{idx}: {probs[idx]:.2%}")

if __name__ == '__main__':
    main()