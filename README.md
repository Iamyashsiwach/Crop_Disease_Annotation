# 🌱 Crop Disease Annotation - Segmentation

**YOLOv8 Instance Segmentation for Crop Disease Detection**

![YOLO](https://img.shields.io/badge/YOLO-v8-blue?style=for-the-badge)
![Task](https://img.shields.io/badge/Task-Segmentation-green?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-50_Images-orange?style=for-the-badge)

> Precise polygon segmentation of crop diseases using YOLOv8 segmentation models. Optimized for small datasets with advanced training strategies.

## 🚀 Quick Start

```bash
# 1. Split your dataset (50 images → 40 train + 10 val)
python3 split_data.py

# 2. Run optimized training
python3 train_yolov8_improved.py

# Choose strategy 1 (Improved training) for best results
```

## 📊 Dataset

- **50 annotated crop disease images**
- **Polygon segmentation masks** (CVAT exported)
- **YOLO format** label files
- **Single class**: Crop_Disease

## 🎯 Training Results

| Model | mAP50(M) | mAP50-95(M) | Training Time |
|-------|----------|-------------|---------------|
| YOLOv8n-seg | ~0.83 | ~0.37 | ~45 min |
| YOLOv8s-seg | ~0.89 | ~0.46 | ~60 min |

## 📁 Project Structure

See **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** for detailed organization.

```
📊 DATASET → 🧠 TRAINING SCRIPTS → ⚙️ CONFIGURATION → 📈 RESULTS
```

## 🔧 Training Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Improved** | Nano model, optimized augmentation | Small datasets ⭐ |
| **Basic** | Simple setup for testing | Quick experiments |
| **Nano** | Fastest training, conservative settings | Speed optimization |
| **No Early Stop** | Full learning curve analysis | Research & debugging |

## 📈 Key Features

- ✅ **Optimized for small datasets** (50 images)
- ✅ **Proper train/validation split** (80/20)
- ✅ **Multiple training strategies**
- ✅ **Segmentation-specific optimizations**
- ✅ **Comprehensive documentation**
- ✅ **Compatible with [Crop_Insect_Annotation](https://github.com/Iamyashsiwach/crop_Insect_Annotation)**

## 🛠️ Requirements

```bash
pip install ultralytics opencv-python Pillow
```

## 📖 Documentation

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed project organization
- **[TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md)** - Training optimizations & troubleshooting

## 🤝 Related Projects

- **[Crop_Insect_Annotation](https://github.com/Iamyashsiwach/crop_Insect_Annotation)** - Crop insect detection (companion project)

## 📊 Training Monitoring

Monitor key segmentation metrics:
- **Mask mAP50**: Primary segmentation accuracy metric
- **Box mAP50**: Detection accuracy
- **Mask Loss**: Segmentation quality loss
- **Training/Validation curves**: Overfitting detection

## 🎯 Usage Example

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/segment/crop_disease_improved/weights/best.pt')

# Run segmentation
results = model.predict('disease_image.jpg')

# Access segmentation masks
masks = results[0].masks.data  # Polygon masks
boxes = results[0].boxes.xyxy  # Bounding boxes
```

## 🌱 Dataset Notes

- **Annotation Tool**: CVAT (Computer Vision Annotation Tool)
- **Export Format**: YOLO segmentation
- **Coordinate System**: Normalized polygon coordinates (0-1)
- **Quality**: Manually verified polygon precision

---

**🎉 Ready to train precise crop disease segmentation models!** 