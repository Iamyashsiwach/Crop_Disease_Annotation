# Crop Disease Segmentation Training Guide

This guide will help you train a YOLOv8 segmentation model on your crop disease dataset.

## 📋 Prerequisites

1. **Python 3.8+** installed on your system
2. **GPU support** (recommended for faster training)
3. **Crop disease dataset** annotated with polygon segmentation in YOLO format

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support (CUDA), install PyTorch with CUDA:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Prepare Your Dataset

Organize your dataset in the following structure:
```
crop_disease_seg/
├── images/
│   ├── train/         # Training images (.jpg, .png, etc.)
│   └── val/           # Validation images
├── labels/
│   ├── train/         # YOLO segmentation labels (.txt)
│   └── val/           # YOLO validation labels
└── data.yaml          # Dataset configuration
```

### 3. Configure data.yaml

Use the provided `sample_data.yaml` as a template. Update the class names to match your dataset:

```yaml
path: .
train: images/train
val: images/val

names:
  0: healthy_leaf
  1: disease_type_1
  2: disease_type_2
  # Add your disease classes here

nc: 3  # Number of classes
```

### 4. Run Training

```bash
python train_yolo_segmentation.py
```

## ⚙️ Configuration Options

You can modify these parameters in the script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATASET_PATH` | `"crop_disease_seg"` | Path to your dataset |
| `MODEL_NAME` | `"yolov8s-seg.pt"` | YOLOv8 model variant |
| `EPOCHS` | `200` | Number of training epochs |
| `IMAGE_SIZE` | `640` | Training image size |
| `PROJECT_NAME` | `"runs/segment"` | Results directory |
| `EXPERIMENT_NAME` | `"crop_disease_seg"` | Experiment name |

### Available YOLOv8 Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolov8n-seg.pt` | Nano | Fastest | Lower |
| `yolov8s-seg.pt` | Small | Fast | Good |
| `yolov8m-seg.pt` | Medium | Medium | Better |
| `yolov8l-seg.pt` | Large | Slower | High |
| `yolov8x-seg.pt` | Extra Large | Slowest | Highest |

## 📊 Understanding the Output

After training, you'll find these files in `runs/segment/crop_disease_seg/`:

- **`weights/best.pt`** - Best model weights (use this for inference)
- **`weights/last.pt`** - Last epoch weights
- **`results.png`** - Training metrics plots
- **`confusion_matrix.png`** - Confusion matrix
- **`val_batch*.jpg`** - Validation predictions visualization

## 🎯 Using the Trained Model

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/segment/crop_disease_seg/weights/best.pt')

# Run prediction
results = model.predict('path/to/your/image.jpg')

# Process results
for result in results:
    # Get segmentation masks
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        
    # Get bounding boxes
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce image size: `IMAGE_SIZE = 416`
   - Use smaller model: `yolov8n-seg.pt`
   - Reduce batch size in the script

2. **No validation images found**
   - Check your dataset structure
   - Ensure validation images exist in `crop_disease_seg/images/val/`

3. **Training loss not decreasing**
   - Check label format (should be normalized coordinates)
   - Verify class indices in labels match `data.yaml`
   - Consider data augmentation

### Performance Tips

1. **Use GPU**: Ensure CUDA is properly installed
2. **Mixed precision**: Add `amp=True` to training parameters
3. **Batch size**: Increase if you have enough GPU memory
4. **Learning rate**: Adjust `lr0` parameter for your dataset

## 📈 Monitoring Training

The script automatically saves training plots. Monitor these metrics:

- **Loss curves** - Should generally decrease over time
- **mAP (mean Average Precision)** - Higher is better
- **Precision/Recall** - Balance depends on your use case

## 🎛️ Advanced Configuration

For advanced users, you can modify the training call in the script:

```python
results = model.train(
    data=data_yaml_path,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=16,              # Batch size
    lr0=0.01,              # Initial learning rate
    momentum=0.937,        # SGD momentum
    weight_decay=0.0005,   # Weight decay
    warmup_epochs=3,       # Warmup epochs
    amp=True,              # Automatic Mixed Precision
    # Add more parameters as needed
)
```

## 📞 Support

If you encounter issues:
1. Check the [Ultralytics documentation](https://docs.ultralytics.com/)
2. Verify your dataset format matches YOLO requirements
3. Ensure all dependencies are correctly installed

Happy training! 🚀 