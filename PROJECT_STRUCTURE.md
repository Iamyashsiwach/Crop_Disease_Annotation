# 🌱 Crop Disease Segmentation Project Structure

```
Crop_Disease_Annotation/
├── 📊 DATASET
│   ├── images/
│   │   ├── train/           # 40 .jpg files (after split)
│   │   └── val/             # 10 .jpg files (after split)
│   └── labels/
│       ├── train/           # 40 .txt files (segmentation polygons)
│       ├── val/             # 10 .txt files (segmentation polygons)
│       └── train.cache      # YOLOv8 cache (auto-generated)
│
├── 🧠 TRAINING SCRIPTS
│   ├── train_yolov8.py                # Basic segmentation training
│   ├── train_yolov8_improved.py       # Advanced training strategies
│   └── split_data.py                  # Data management & splitting
│
├── ⚙️ CONFIGURATION
│   ├── data.yaml                      # YOLOv8 segmentation config
│   ├── TRAINING_IMPROVEMENTS.md       # Training optimization guide
│   ├── PROJECT_STRUCTURE.md           # This file
│   └── .gitignore                     # Git rules
│
├── 📈 TRAINING RESULTS
│   └── runs/
│       └── segment/                   # Segmentation results
│           ├── crop_disease_basic/    # Basic training results
│           ├── crop_disease_improved/ # Improved training
│           ├── crop_disease_nano/     # Nano model training
│           └── crop_disease_test/     # Test runs
│               └── weights/
│                   ├── best.pt        # Best segmentation model
│                   └── last.pt        # Latest checkpoint
│
├── 🏋️ PRE-TRAINED MODELS
│   ├── yolov8n-seg.pt      # Nano Segmentation (6.7 MB)
│   ├── yolov8s-seg.pt      # Small Segmentation (23.8 MB)
│   └── yolov8m-seg.pt      # Medium Segmentation (49.9 MB)
│
└── 📜 LEGACY (Original Dataset)
    ├── images/train/       # 50 original disease images
    ├── labels/train/       # 50 original polygon annotations
    └── data.yaml.backup    # Original configuration
```

## 🚀 Quick Start Guide

### 1. **Prepare Dataset**
```bash
# Split your 50 images into train/val (40/10)
python3 split_data.py
```

### 2. **Basic Training**
```bash
# Simple training for testing
python3 train_yolov8.py
```

### 3. **Advanced Training**
```bash
# Optimized training with multiple strategies
python3 train_yolov8_improved.py
```

## 📊 Training Strategies

| Strategy | Model | Best For | Expected Time |
|----------|-------|----------|---------------|
| **Basic** | YOLOv8s-seg | Testing setup | ~30 min |
| **Improved** | YOLOv8n-seg | Small datasets | ~45 min |
| **Nano** | YOLOv8n-seg | Fastest training | ~25 min |
| **No Early Stop** | YOLOv8n-seg | Full analysis | ~60 min |

## 🎯 Segmentation Metrics

Monitor these key metrics during training:

- **Box mAP50**: Bounding box detection accuracy
- **Mask mAP50**: Segmentation mask accuracy  
- **Box Loss**: Bounding box regression loss
- **Mask Loss**: Segmentation quality loss
- **Class Loss**: Classification accuracy

## 📈 Expected Results

With proper training on your 50 annotated images:

```
Class    Images  Instances  Box(P    R    mAP50  mAP50-95)  Mask(P   R    mAP50  mAP50-95)
all      10      15         0.756  0.867  0.891  0.456      0.692  0.800  0.831  0.372
```

## 🔧 Troubleshooting

### Common Issues:
- **CUDA out of memory**: Reduce batch size
- **No validation data**: Run `split_data.py` first
- **Poor results**: Check annotation quality
- **Early stopping**: Increase patience or disable

### File Locations:
- **Best model**: `runs/segment/{experiment}/weights/best.pt`
- **Training plots**: `runs/segment/{experiment}/results.png`
- **Confusion matrix**: `runs/segment/{experiment}/confusion_matrix.png`

## 🌱 Dataset Information

- **Task**: Instance segmentation of crop diseases
- **Format**: YOLO segmentation (polygon coordinates)
- **Classes**: 1 (Crop_Disease)
- **Total Images**: 50 annotated disease images
- **Annotation Type**: Precise polygon masks around diseased areas

## 🤝 Integration

This segmentation project is designed to work alongside:
- **[Crop_Insect_Annotation](https://github.com/Iamyashsiwach/crop_Insect_Annotation)** (Detection)
- Combined crop monitoring pipeline
- Shared training methodologies and optimizations 