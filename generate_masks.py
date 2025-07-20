#!/usr/bin/env python3
"""
Generate masked images from landmarks for fashion detection training.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse

def parse_landmarks_file(landmarks_path):
    """Parse the landmarks file and return a dictionary of image data."""
    landmarks_data = {}
    
    with open(landmarks_path, 'r') as f:
        header = f.readline().strip()
        
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
                
            image_name = parts[0].replace('img/', '')
            clothes_type = int(parts[1])
            variation_type = int(parts[2])
            
            # Parse landmarks (8 landmarks with visibility and x,y coordinates)
            landmarks = []
            for i in range(8):
                base_idx = 3 + i * 3
                if base_idx + 2 < len(parts):
                    visibility = int(parts[base_idx])
                    x = int(parts[base_idx + 1])
                    y = int(parts[base_idx + 2])
                    landmarks.append((visibility, x, y))
                else:
                    landmarks.append((0, 0, 0))
            
            landmarks_data[image_name] = {
                'clothes_type': clothes_type,
                'variation_type': variation_type,
                'landmarks': landmarks
            }
    
    return landmarks_data

def create_mask_from_landmarks(image_shape, landmarks, mask_type='polygon', clothes_type=None):
    """Create a mask from landmarks.
    
    For DeepFashion dataset, the 8 landmarks typically represent:
    1. Left collar
    2. Right collar  
    3. Left sleeve
    4. Right sleeve
    5. Left hem
    6. Right hem
    7. Left waistline (not always visible)
    8. Right waistline (not always visible)
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # Get visible landmarks
    visible_points = []
    landmark_indices = []
    for i, (visibility, x, y) in enumerate(landmarks):
        if visibility == 1 and x > 0 and y > 0:
            visible_points.append([x, y])
            landmark_indices.append(i)
    
    if len(visible_points) < 3:
        # Not enough points to create a polygon
        return mask
    
    visible_points = np.array(visible_points, dtype=np.int32)
    
    if mask_type == 'polygon':
        # For fashion items, try to connect landmarks in a meaningful order
        # This creates a better polygon than random point order
        if len(landmark_indices) >= 4:
            # Try to order points to create a proper clothing outline
            ordered_points = order_fashion_landmarks(visible_points, landmark_indices)
            cv2.fillPoly(mask, [ordered_points], 255)
        else:
            cv2.fillPoly(mask, [visible_points], 255)
    elif mask_type == 'convex_hull':
        # Create convex hull mask - good default for clothing items
        hull = cv2.convexHull(visible_points)
        cv2.fillPoly(mask, [hull], 255)
    elif mask_type == 'ellipse' and len(visible_points) >= 5:
        # Fit ellipse to points
        ellipse = cv2.fitEllipse(visible_points)
        cv2.ellipse(mask, ellipse, 255, -1)
    elif mask_type == 'tight_polygon':
        # Create a tighter polygon by connecting points more intelligently
        if len(visible_points) >= 4:
            # Sort points to create a proper contour
            center = np.mean(visible_points, axis=0)
            angles = np.arctan2(visible_points[:, 1] - center[1], 
                              visible_points[:, 0] - center[0])
            sorted_indices = np.argsort(angles)
            sorted_points = visible_points[sorted_indices]
            cv2.fillPoly(mask, [sorted_points], 255)
        else:
            cv2.fillPoly(mask, [visible_points], 255)
    
    return mask

def order_fashion_landmarks(points, indices):
    """Order fashion landmarks to create a proper clothing outline."""
    # This is a simple heuristic - can be improved based on specific landmark meanings
    # Sort by y-coordinate first (top to bottom), then by x-coordinate
    sorted_indices = np.lexsort((points[:, 0], points[:, 1]))
    return points[sorted_indices]

def process_images(input_dir, output_dir, landmarks_data, mask_type='convex_hull'):
    """Process all images and generate masks."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of images
    image_files = sorted(input_path.glob('*.jpg'))
    
    processed = 0
    skipped = 0
    
    for img_path in tqdm(image_files, desc="Generating masks"):
        img_name = img_path.name
        
        if img_name not in landmarks_data:
            print(f"Warning: No landmarks found for {img_name}")
            skipped += 1
            continue
        
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error: Could not read {img_path}")
            skipped += 1
            continue
        
        # Get landmarks
        data = landmarks_data[img_name]
        landmarks = data['landmarks']
        clothes_type = data['clothes_type']
        
        # Create mask
        mask = create_mask_from_landmarks(image.shape, landmarks, mask_type, clothes_type)
        
        # Apply mask to image
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Create background (optional - make it black or blur the original)
        background = np.zeros_like(image)
        
        # Combine masked foreground with background
        inv_mask = cv2.bitwise_not(mask)
        background_masked = cv2.bitwise_and(background, background, mask=inv_mask)
        result = cv2.add(masked_image, background_masked)
        
        # Save result
        output_file = output_path / img_name
        cv2.imwrite(str(output_file), result)
        
        # Also save the mask itself
        mask_file = output_path / f"mask_{img_name}"
        cv2.imwrite(str(mask_file), mask)
        
        processed += 1
    
    print(f"\nProcessing complete!")
    print(f"Processed: {processed} images")
    print(f"Skipped: {skipped} images")

def main():
    parser = argparse.ArgumentParser(description='Generate masked images from landmarks')
    parser.add_argument('--input-dir', type=str, default='1000 images',
                        help='Input directory containing images')
    parser.add_argument('--output-dir', type=str, default='masked_images',
                        help='Output directory for masked images')
    parser.add_argument('--landmarks-file', type=str, default='1000 images/crop landmarks.txt',
                        help='Path to landmarks file')
    parser.add_argument('--mask-type', type=str, default='convex_hull',
                        choices=['polygon', 'convex_hull', 'ellipse', 'tight_polygon'],
                        help='Type of mask to generate')
    
    args = parser.parse_args()
    
    # Parse landmarks
    print(f"Parsing landmarks from {args.landmarks_file}...")
    landmarks_data = parse_landmarks_file(args.landmarks_file)
    print(f"Found landmarks for {len(landmarks_data)} images")
    
    # Process images
    print(f"\nGenerating masked images...")
    process_images(args.input_dir, args.output_dir, landmarks_data, args.mask_type)

if __name__ == '__main__':
    main()