#!/usr/bin/env python3
"""
Split crop disease dataset into training and validation sets
"""

import os
import shutil
import random
from pathlib import Path

def split_dataset(train_ratio=0.8):
    """Split images and labels into train/val sets"""
    
    # Create validation directories if they don't exist
    val_images = Path('images/val')
    val_labels = Path('labels/val')
    val_images.mkdir(exist_ok=True)
    val_labels.mkdir(exist_ok=True)
    
    # Get paths
    train_images = Path('images/train')
    train_labels = Path('labels/train')
    
    # Get list of all images
    image_files = list(train_images.glob('*.jpg'))
    
    # Shuffle for random split
    random.seed(42)  # for reproducible results
    random.shuffle(image_files)
    
    # Calculate split point
    split_point = int(len(image_files) * train_ratio)
    
    # Split into train and validation
    train_files = image_files[:split_point]
    val_files = image_files[split_point:]
    
    print(f"🌱 Crop Disease Dataset Split")
    print(f"Total images: {len(image_files)}")
    print(f"Training: {len(train_files)}")
    print(f"Validation: {len(val_files)}")
    
    # Move validation files
    for img_file in val_files:
        # Move image
        val_img_path = val_images / img_file.name
        shutil.move(str(img_file), str(val_img_path))
        
        # Move corresponding label
        label_file = train_labels / (img_file.stem + '.txt')
        val_label_path = val_labels / (img_file.stem + '.txt')
        if label_file.exists():
            shutil.move(str(label_file), str(val_label_path))
    
    print("✅ Data split completed!")
    print(f"Training images: {len(list(train_images.glob('*.jpg')))}")
    print(f"Validation images: {len(list(val_images.glob('*.jpg')))}")

if __name__ == "__main__":
    split_dataset() 