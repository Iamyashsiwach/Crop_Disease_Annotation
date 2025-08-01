#!/usr/bin/env python3
"""
Data Splitting Script for Crop Disease Segmentation Dataset
===========================================================

This script splits your annotated crop disease dataset into training and validation sets.
Based on the structure from crop_Insect_Annotation repository but adapted for segmentation.

Usage:
    python split_data.py

Author: Generated for Crop Disease Annotation Project
"""

import os
import random
import shutil
from pathlib import Path
import glob
from typing import List, Tuple

def get_image_label_pairs(images_dir: str, labels_dir: str) -> List[Tuple[str, str]]:
    """
    Get corresponding image and label file pairs.
    
    Args:
        images_dir (str): Directory containing images
        labels_dir (str): Directory containing label files
        
    Returns:
        List[Tuple[str, str]]: List of (image_path, label_path) pairs
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    pairs = []
    
    for ext in image_extensions:
        for image_path in glob.glob(os.path.join(images_dir, ext)):
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(labels_dir, f"{image_name}.txt")
            
            if os.path.exists(label_path):
                pairs.append((image_path, label_path))
            else:
                print(f"⚠️ Warning: No label file found for {image_path}")
    
    return pairs

def create_directory_structure(base_dir: str):
    """
    Create the YOLO dataset directory structure.
    
    Args:
        base_dir (str): Base directory for the dataset
    """
    dirs_to_create = [
        os.path.join(base_dir, 'images', 'train'),
        os.path.join(base_dir, 'images', 'val'),
        os.path.join(base_dir, 'labels', 'train'),
        os.path.join(base_dir, 'labels', 'val')
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def split_dataset(source_images_dir: str, source_labels_dir: str, 
                 output_dir: str = "crop_disease_seg", 
                 train_ratio: float = 0.8, seed: int = 42):
    """
    Split dataset into training and validation sets.
    
    Args:
        source_images_dir (str): Source directory containing all images
        source_labels_dir (str): Source directory containing all labels
        output_dir (str): Output directory for split dataset
        train_ratio (float): Ratio of data to use for training (0.0-1.0)
        seed (int): Random seed for reproducible splits
    """
    print("🚀 Starting dataset splitting...")
    print(f"📁 Source images: {source_images_dir}")
    print(f"📁 Source labels: {source_labels_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Train ratio: {train_ratio:.2f}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get all image-label pairs
    pairs = get_image_label_pairs(source_images_dir, source_labels_dir)
    
    if not pairs:
        print("❌ No valid image-label pairs found!")
        return
    
    print(f"📊 Found {len(pairs)} image-label pairs")
    
    # Shuffle the pairs
    random.shuffle(pairs)
    
    # Calculate split point
    train_count = int(len(pairs) * train_ratio)
    val_count = len(pairs) - train_count
    
    print(f"📊 Training samples: {train_count}")
    print(f"📊 Validation samples: {val_count}")
    
    # Create directory structure
    create_directory_structure(output_dir)
    
    # Split the data
    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:]
    
    # Copy training data
    print("\n📥 Copying training data...")
    for i, (img_path, label_path) in enumerate(train_pairs):
        img_name = os.path.basename(img_path)
        label_name = os.path.basename(label_path)
        
        # Copy image
        dst_img = os.path.join(output_dir, 'images', 'train', img_name)
        shutil.copy2(img_path, dst_img)
        
        # Copy label
        dst_label = os.path.join(output_dir, 'labels', 'train', label_name)
        shutil.copy2(label_path, dst_label)
        
        if (i + 1) % 50 == 0 or (i + 1) == len(train_pairs):
            print(f"   Copied {i + 1}/{len(train_pairs)} training samples")
    
    # Copy validation data
    print("\n📥 Copying validation data...")
    for i, (img_path, label_path) in enumerate(val_pairs):
        img_name = os.path.basename(img_path)
        label_name = os.path.basename(label_path)
        
        # Copy image
        dst_img = os.path.join(output_dir, 'images', 'val', img_name)
        shutil.copy2(img_path, dst_img)
        
        # Copy label
        dst_label = os.path.join(output_dir, 'labels', 'val', label_name)
        shutil.copy2(label_path, dst_label)
        
        if (i + 1) % 20 == 0 or (i + 1) == len(val_pairs):
            print(f"   Copied {i + 1}/{len(val_pairs)} validation samples")
    
    print(f"\n✅ Dataset splitting completed!")
    print(f"📁 Output structure:")
    print(f"   {output_dir}/")
    print(f"   ├── images/")
    print(f"   │   ├── train/ ({train_count} images)")
    print(f"   │   └── val/ ({val_count} images)")
    print(f"   └── labels/")
    print(f"       ├── train/ ({train_count} labels)")
    print(f"       └── val/ ({val_count} labels)")

def verify_split(dataset_dir: str):
    """
    Verify the split dataset structure and counts.
    
    Args:
        dataset_dir (str): Directory containing the split dataset
    """
    print(f"\n🔍 Verifying dataset: {dataset_dir}")
    
    splits = ['train', 'val']
    total_images = 0
    total_labels = 0
    
    for split in splits:
        img_dir = os.path.join(dataset_dir, 'images', split)
        label_dir = os.path.join(dataset_dir, 'labels', split)
        
        if os.path.exists(img_dir) and os.path.exists(label_dir):
            img_count = len(glob.glob(os.path.join(img_dir, '*')))
            label_count = len(glob.glob(os.path.join(label_dir, '*.txt')))
            
            print(f"   {split.upper()}: {img_count} images, {label_count} labels")
            
            if img_count != label_count:
                print(f"   ⚠️ Warning: Mismatch in {split} set!")
            
            total_images += img_count
            total_labels += label_count
        else:
            print(f"   ❌ Missing {split} directories")
    
    print(f"📊 Total: {total_images} images, {total_labels} labels")
    
    if total_images == total_labels:
        print("✅ Dataset verification passed!")
    else:
        print("❌ Dataset verification failed!")

def main():
    """
    Main function for dataset splitting.
    """
    print("🚀 Crop Disease Dataset Splitter")
    print("=" * 50)
    
    # Configuration - modify these paths according to your setup
    SOURCE_IMAGES_DIR = "images/train"  # Current images directory
    SOURCE_LABELS_DIR = "labels/train"  # Current labels directory
    OUTPUT_DIR = "crop_disease_seg"     # Output directory name
    TRAIN_RATIO = 0.8                   # 80% for training, 20% for validation
    RANDOM_SEED = 42                    # For reproducible splits
    
    # Check if source directories exist
    if not os.path.exists(SOURCE_IMAGES_DIR):
        print(f"❌ Source images directory not found: {SOURCE_IMAGES_DIR}")
        print("Please update SOURCE_IMAGES_DIR in the script to point to your images.")
        return
    
    if not os.path.exists(SOURCE_LABELS_DIR):
        print(f"❌ Source labels directory not found: {SOURCE_LABELS_DIR}")
        print("Please update SOURCE_LABELS_DIR in the script to point to your labels.")
        return
    
    # Perform the split
    split_dataset(
        source_images_dir=SOURCE_IMAGES_DIR,
        source_labels_dir=SOURCE_LABELS_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        seed=RANDOM_SEED
    )
    
    # Verify the split
    verify_split(OUTPUT_DIR)
    
    print(f"\n🎉 Dataset ready for training!")
    print(f"💡 Next steps:")
    print(f"   1. Update data.yaml with correct paths")
    print(f"   2. Run: python train_yolo_segmentation.py")

if __name__ == "__main__":
    main() 