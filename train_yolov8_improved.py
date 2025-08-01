#!/usr/bin/env python3
"""
Improved YOLOv8 Segmentation Training Script for Small Crop Disease Dataset
Optimized for datasets with limited samples and segmentation tasks
"""

from ultralytics import YOLO
import os

def train_improved_model():
    """Train YOLOv8 segmentation with optimized settings for small datasets"""
    print("🚀 Starting Improved YOLOv8 Segmentation Training for Small Dataset...")
    
    # Use nano segmentation model for small datasets - faster and less prone to overfitting
    model = YOLO('yolov8n-seg.pt')  # nano segmentation model
    
    # Optimized training parameters for small datasets
    results = model.train(
        data='data.yaml',
        epochs=200,           # More epochs since we have patience
        imgsz=640,
        batch=4,             # Smaller batch size for small dataset
        lr0=0.001,           # Lower initial learning rate
        patience=50,         # More patience before early stopping
        save=True,
        plots=True,
        name='crop_disease_improved',
        
        # Data augmentation settings (moderate for small datasets)
        hsv_h=0.015,         # Hue augmentation
        hsv_s=0.7,           # Saturation augmentation  
        hsv_v=0.4,           # Value augmentation
        degrees=10,          # Rotation augmentation
        translate=0.1,       # Translation augmentation
        scale=0.2,           # Scale augmentation
        shear=2,             # Shear augmentation
        perspective=0.0,     # Perspective augmentation
        flipud=0.0,          # Vertical flip probability
        fliplr=0.5,          # Horizontal flip probability
        mosaic=0.5,          # Mosaic augmentation probability
        mixup=0.1,           # Mixup augmentation probability
        
        # Segmentation specific settings
        overlap_mask=True,   # Allow overlapping masks
        mask_ratio=4,        # Mask downsampling ratio
        
        # Optimization settings
        optimizer='AdamW',   # Often works better for small datasets
        cos_lr=True,         # Cosine learning rate scheduler
        warmup_epochs=5,     # Warmup epochs
        
        # Validation settings
        val=True,            # Enable validation
        save_period=10,      # Save checkpoint every 10 epochs
    )
    
    print("✅ Improved segmentation training completed!")
    return results

def train_with_nano_model():
    """Alternative training with even smaller nano segmentation model"""
    print("🚀 Starting YOLOv8 Nano Segmentation Training...")
    
    model = YOLO('yolov8n-seg.pt')
    
    results = model.train(
        data='data.yaml',
        epochs=300,
        imgsz=416,           # Smaller image size for faster training
        batch=8,
        lr0=0.002,
        patience=100,        # Very patient
        name='crop_disease_nano',
        save=True,
        plots=True,
        
        # Conservative augmentation
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5,
        translate=0.05,
        scale=0.1,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.3,
        
        # Segmentation settings
        overlap_mask=True,
        mask_ratio=4,
        
        optimizer='SGD',
        momentum=0.9,
        weight_decay=0.0005,
    )
    
    print("✅ Nano segmentation model training completed!")
    return results

def train_without_early_stopping():
    """Train without early stopping to see full learning curve"""
    print("🚀 Training segmentation without early stopping...")
    
    model = YOLO('yolov8n-seg.pt')
    
    results = model.train(
        data='data.yaml',
        epochs=100,          # Fixed epochs, no early stopping
        imgsz=640,
        batch=6,
        lr0=0.001,
        patience=0,          # Disable early stopping
        name='crop_disease_no_early_stop',
        save=True,
        plots=True,
        
        # Moderate augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.3,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.7,
        mixup=0.2,
        
        # Segmentation settings
        overlap_mask=True,
        mask_ratio=4,
        
        optimizer='AdamW',
        cos_lr=True,
    )
    
    print("✅ Training without early stopping completed!")
    return results

if __name__ == "__main__":
    print("Improved YOLOv8 Crop Disease Segmentation Training")
    print("=" * 55)
    
    # Check if data.yaml exists
    if not os.path.exists('data.yaml'):
        print("❌ Error: data.yaml not found!")
        exit(1)
    
    # Check if validation data exists
    if not os.path.exists('images/val'):
        print("❌ Error: No validation data found!")
        print("💡 Run 'python3 split_data.py' first to create train/val split")
        exit(1)
    
    print("Select training strategy:")
    print("1. Improved training (recommended for small datasets)")
    print("2. Nano model training (fastest, good for very small datasets)")
    print("3. Training without early stopping (to see full learning curve)")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    try:
        if choice == "1":
            train_improved_model()
        elif choice == "2":
            train_with_nano_model()
        elif choice == "3":
            train_without_early_stopping()
        else:
            print("❌ Invalid choice!")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("💡 Tips for troubleshooting:")
        print("  - Try reducing batch size if you get memory errors")
        print("  - Ensure you have enough disk space")
        print("  - Check that all image and label files are valid")
        print("  - Make sure your labels are in YOLO segmentation format (polygon coordinates)") 