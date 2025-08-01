#!/usr/bin/env python3
"""
Crop Disease Segmentation Training Script using YOLOv8
======================================================

This script trains a YOLOv8 segmentation model on a crop disease dataset
annotated with polygon segmentation and exported in YOLO format.

Dataset Structure Expected:
crop_disease_seg/
├── images/
│   ├── train/         # Training images
│   └── val/           # Validation images
├── labels/
│   ├── train/         # YOLO segmentation labels (.txt)
│   └── val/
├── data.yaml          # YOLO dataset config file

Author: Generated for Crop Disease Annotation Project
"""

import os
import sys
import glob
import random
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

def check_dataset_structure(dataset_path):
    """
    Verify that the dataset has the correct structure.
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        bool: True if structure is correct, False otherwise
    """
    required_dirs = [
        'images/train',
        'images/val', 
        'labels/train',
        'labels/val'
    ]
    
    print("🔍 Checking dataset structure...")
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_path, dir_path)
        if not os.path.exists(full_path):
            print(f"❌ Missing directory: {full_path}")
            return False
        else:
            print(f"✅ Found: {dir_path}")
    
    # Check for data.yaml
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(data_yaml_path):
        print(f"❌ Missing: data.yaml")
        return False
    else:
        print(f"✅ Found: data.yaml")
    
    return True

def get_sample_image(dataset_path, split='val'):
    """
    Get a random sample image from the specified split.
    
    Args:
        dataset_path (str): Path to the dataset
        split (str): 'train' or 'val'
        
    Returns:
        str: Path to a sample image, or None if no images found
    """
    image_dir = os.path.join(dataset_path, 'images', split)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(image_dir, ext)))
        all_images.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if all_images:
        return random.choice(all_images)
    return None

def main():
    """
    Main training function for YOLOv8 segmentation model.
    """
    print("🚀 Starting Crop Disease Segmentation Training")
    print("=" * 60)
    
    # Configuration
    DATASET_PATH = "crop_disease_seg"  # Adjust this path as needed
    MODEL_NAME = "yolov8s-seg.pt"      # YOLOv8 small segmentation model
    EPOCHS = 200
    IMAGE_SIZE = 640
    PROJECT_NAME = "runs/segment"
    EXPERIMENT_NAME = "crop_disease_seg"
    
    # Step 1: Check if dataset exists and has correct structure
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset directory '{DATASET_PATH}' not found!")
        print("Please ensure your dataset follows this structure:")
        print("crop_disease_seg/")
        print("├── images/")
        print("│   ├── train/")
        print("│   └── val/")
        print("├── labels/")
        print("│   ├── train/")
        print("│   └── val/")
        print("└── data.yaml")
        sys.exit(1)
    
    if not check_dataset_structure(DATASET_PATH):
        print("❌ Dataset structure is incorrect!")
        sys.exit(1)
    
    print("\n📊 Dataset structure verified successfully!")
    
    # Step 2: Load YOLOv8 segmentation model
    print(f"\n🔧 Loading YOLOv8 segmentation model: {MODEL_NAME}")
    try:
        model = YOLO(MODEL_NAME)
        print("✅ Model loaded successfully!")
        
        # Print model info
        print(f"📋 Model architecture: {model.model}")
        print(f"📋 Model task: {model.task}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you have ultralytics installed: pip install ultralytics")
        sys.exit(1)
    
    # Step 3: Configure training parameters
    data_yaml_path = os.path.join(DATASET_PATH, "data.yaml")
    print(f"\n⚙️ Training Configuration:")
    print(f"   Dataset config: {data_yaml_path}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Image size: {IMAGE_SIZE}")
    print(f"   Results directory: {PROJECT_NAME}/{EXPERIMENT_NAME}")
    
    # Step 4: Start training
    print(f"\n🏋️ Starting training...")
    print("This may take a while depending on your hardware and dataset size.")
    
    try:
        # Train the model
        results = model.train(
            data=data_yaml_path,           # Path to dataset YAML
            epochs=EPOCHS,                 # Number of training epochs
            imgsz=IMAGE_SIZE,             # Training image size
            project=PROJECT_NAME,         # Project directory
            name=EXPERIMENT_NAME,         # Experiment name
            save=True,                    # Save checkpoints
            plots=True,                   # Save training plots
            verbose=True,                 # Verbose output
            device='auto',                # Automatically select device (GPU if available)
        )
        
        print("✅ Training completed successfully!")
        print(f"📁 Results saved to: {results.save_dir}")
        
        # Print training summary
        if hasattr(results, 'results_dict'):
            print("\n📈 Training Summary:")
            for key, value in results.results_dict.items():
                print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Step 5: Load the best trained model
    print(f"\n🔄 Loading best trained model...")
    best_model_path = os.path.join(PROJECT_NAME, EXPERIMENT_NAME, "weights", "best.pt")
    
    if os.path.exists(best_model_path):
        trained_model = YOLO(best_model_path)
        print(f"✅ Best model loaded from: {best_model_path}")
    else:
        print("⚠️ Best model not found, using last trained weights")
        trained_model = model
    
    # Step 6: Run validation on the trained model
    print(f"\n📊 Validating trained model...")
    try:
        val_results = trained_model.val(
            data=data_yaml_path,
            imgsz=IMAGE_SIZE,
            split='val'
        )
        
        print("✅ Validation completed!")
        print(f"\n📈 Validation Metrics:")
        
        # Print key metrics
        if hasattr(val_results, 'results_dict'):
            metrics = val_results.results_dict
            print(f"   mAP50: {metrics.get('metrics/mAP50(M)', 'N/A'):.4f}")
            print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(M)', 'N/A'):.4f}")
            print(f"   Precision: {metrics.get('metrics/precision(M)', 'N/A'):.4f}")
            print(f"   Recall: {metrics.get('metrics/recall(M)', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"⚠️ Validation failed: {e}")
    
    # Step 7: Run prediction on a sample validation image
    print(f"\n🎯 Running prediction on sample validation image...")
    
    sample_image = get_sample_image(DATASET_PATH, 'val')
    if sample_image:
        print(f"   Selected sample: {sample_image}")
        
        try:
            # Run prediction
            prediction_results = trained_model.predict(
                source=sample_image,
                imgsz=IMAGE_SIZE,
                conf=0.25,                    # Confidence threshold
                save=True,                    # Save results
                project=PROJECT_NAME,        # Save directory
                name=f"{EXPERIMENT_NAME}_predictions"
            )
            
            print("✅ Prediction completed!")
            
            # Print prediction details
            if prediction_results:
                result = prediction_results[0]
                if result.masks is not None:
                    num_detections = len(result.masks.data)
                    print(f"   Detected {num_detections} objects with segmentation masks")
                    
                    # Print confidence scores
                    if result.boxes is not None and result.boxes.conf is not None:
                        confidences = result.boxes.conf.cpu().numpy()
                        print(f"   Confidence scores: {confidences}")
                else:
                    print("   No objects detected in the sample image")
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
    else:
        print("⚠️ No validation images found for prediction")
    
    # Step 8: Summary and next steps
    print(f"\n🎉 Training and evaluation completed!")
    print("=" * 60)
    print("📁 Generated files:")
    print(f"   Training results: {PROJECT_NAME}/{EXPERIMENT_NAME}/")
    print(f"   Best model: {PROJECT_NAME}/{EXPERIMENT_NAME}/weights/best.pt")
    print(f"   Last model: {PROJECT_NAME}/{EXPERIMENT_NAME}/weights/last.pt")
    print(f"   Training plots: {PROJECT_NAME}/{EXPERIMENT_NAME}/")
    
    print(f"\n🚀 Next steps:")
    print("   1. Review training plots and metrics")
    print("   2. Test the model on new images")
    print("   3. Deploy the model for inference")
    print("   4. Fine-tune hyperparameters if needed")
    
    print(f"\n💡 To use the trained model:")
    print(f"   from ultralytics import YOLO")
    print(f"   model = YOLO('{best_model_path}')")
    print(f"   results = model.predict('your_image.jpg')")

if __name__ == "__main__":
    main() 