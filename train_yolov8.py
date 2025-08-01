#!/usr/bin/env python3
"""
Basic YOLOv8 Segmentation Training Script for Crop Disease Detection
Simple training setup for testing and quick experiments
"""

from ultralytics import YOLO
import os

def train_basic_model():
    """Basic YOLOv8 segmentation training"""
    print("🚀 Starting Basic YOLOv8 Segmentation Training...")
    
    # Use small segmentation model for basic training
    model = YOLO('yolov8s-seg.pt')  # small segmentation model
    
    # Basic training parameters
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        name='crop_disease_basic',
        save=True,
        plots=True,
        val=True,
    )
    
    print("✅ Basic training completed!")
    return results

if __name__ == "__main__":
    print("Basic YOLOv8 Crop Disease Segmentation Training")
    print("=" * 50)
    
    # Check if data.yaml exists
    if not os.path.exists('data.yaml'):
        print("❌ Error: data.yaml not found!")
        exit(1)
    
    # Check if validation data exists
    if not os.path.exists('images/val'):
        print("❌ Error: No validation data found!")
        print("💡 Run 'python3 split_data.py' first to create train/val split")
        exit(1)
    
    try:
        train_basic_model()
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("💡 Try reducing batch size if you get memory errors") 