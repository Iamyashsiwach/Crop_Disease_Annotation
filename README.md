# Crop Disease Segmentation with YOLOv8

A comprehensive solution for training YOLOv8 segmentation models to detect and segment crop diseases. This project is enhanced with advanced training features and is designed to complement the [crop_Insect_Annotation](https://github.com/Iamyashsiwach/crop_Insect_Annotation) detection system.

## 🌱 Features

- **YOLOv8 Segmentation**: State-of-the-art instance segmentation for crop diseases
- **Polygon Annotations**: Support for CVAT polygon segmentation annotations
- **Advanced Training Pipeline**: Enhanced training with hyperparameter optimization
- **Model Comparison**: Compare multiple YOLOv8 model architectures
- **Comprehensive Monitoring**: Detailed logging and visualization
- **Data Management**: Intelligent dataset splitting and validation
- **Production Ready**: GPU optimization and deployment-ready models

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
- [Advanced Training](#advanced-training)
- [Model Evaluation](#model-evaluation)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/Iamyashsiwach/Crop_Disease_Annotation.git
cd Crop_Disease_Annotation

# Install required packages
pip install -r requirements.txt

# For GPU support (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📁 Dataset Preparation

### 1. Organize Your Dataset

Your annotated dataset should follow this structure:

```
your_dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt    # Polygon segmentation labels
    ├── image2.txt
    └── ...
```

### 2. Split Dataset

Use the provided script to split your data into training and validation sets:

```bash
# Edit split_data.py to point to your dataset directories
python split_data.py
```

This creates the proper YOLO structure:

```
crop_disease_seg/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

### 3. Configure Classes

Update `crop_disease_seg/data.yaml` with your specific disease classes:

```yaml
names:
  0: healthy_leaf
  1: bacterial_blight
  2: leaf_spot
  3: rust
  # Add your disease types here

nc: 4  # Number of classes
```

## 🎯 Quick Start

### Basic Training

```bash
# Train with default settings
python train_yolo_segmentation.py
```

### Advanced Training

```bash
# Train with enhanced features
python train_yolov8_improved.py
```

### Custom Configuration

```bash
# Train with custom configuration
python train_yolov8_improved.py --config my_config.yaml
```

## 🔧 Advanced Training

### Hyperparameter Optimization

```bash
# Automatic hyperparameter tuning
python train_yolov8_improved.py --tune
```

### Model Comparison

```bash
# Compare multiple model architectures
python train_yolov8_improved.py --compare yolov8n-seg.pt yolov8s-seg.pt yolov8m-seg.pt
```

### Resume Training

```bash
# Resume from checkpoint
python train_yolov8_improved.py --config training_config.yaml
# Set resume: true in config file
```

## 📊 Model Evaluation

### Evaluate Trained Model

```bash
# Evaluate without training
python train_yolov8_improved.py --eval-only runs/segment/crop_disease_improved/weights/best.pt
```

### Performance Metrics

The training automatically provides:
- **mAP50**: Mean Average Precision at IoU 0.5
- **mAP50-95**: Mean Average Precision from IoU 0.5 to 0.95
- **Precision**: Classification precision
- **Recall**: Detection recall
- **Segmentation Quality**: Mask accuracy metrics

## 💻 Usage Examples

### Basic Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/segment/crop_disease_improved/weights/best.pt')

# Run prediction
results = model.predict('path/to/image.jpg', save=True)

# Process results
for result in results:
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        print(f"Detected {len(masks)} diseased areas")
```

### Batch Processing

```python
import glob
from ultralytics import YOLO

model = YOLO('path/to/best.pt')

# Process multiple images
image_paths = glob.glob('test_images/*.jpg')
results = model.predict(image_paths, save=True, conf=0.25)

for i, result in enumerate(results):
    print(f"Image {i+1}: {len(result.masks.data) if result.masks else 0} detections")
```

### Real-time Processing

```python
import cv2
from ultralytics import YOLO

model = YOLO('path/to/best.pt')

# Process webcam feed
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        results = model.predict(frame, conf=0.25)
        annotated_frame = results[0].plot()
        cv2.imshow('Crop Disease Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

## ⚙️ Configuration

### Training Configuration (`training_config.yaml`)

```yaml
dataset:
  path: crop_disease_seg
  data_yaml: crop_disease_seg/data.yaml

model:
  name: yolov8s-seg.pt
  pretrained: true

training:
  epochs: 200
  batch_size: 16
  image_size: 640
  patience: 50
  device: auto

hyperparameters:
  lr0: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3
  # ... more parameters
```

### Available Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `yolov8n-seg.pt` | 6.7MB | Fastest | Good | Mobile/Edge |
| `yolov8s-seg.pt` | 23.8MB | Fast | Better | Balanced |
| `yolov8m-seg.pt` | 49.9MB | Medium | High | Accuracy Focus |
| `yolov8l-seg.pt` | 83.7MB | Slow | Higher | Research |
| `yolov8x-seg.pt` | 130.5MB | Slowest | Highest | Maximum Accuracy |

## 📈 Monitoring Training

### Real-time Monitoring

Training progress is automatically logged and visualized:

- **Console Output**: Real-time metrics display
- **Log Files**: Detailed logs in `training.log`
- **TensorBoard**: Visual training monitoring
- **Plots**: Automatic generation of training curves

### Key Metrics to Watch

1. **Training Loss**: Should decrease steadily
2. **Validation mAP**: Should increase over time
3. **Learning Rate**: Monitor scheduling effectiveness
4. **GPU Utilization**: Ensure efficient resource usage

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size or image size
# In training_config.yaml:
training:
  batch_size: 8    # Reduce from 16
  image_size: 416  # Reduce from 640
```

#### No Objects Detected
- Check label format (normalized coordinates)
- Verify class indices match `data.yaml`
- Lower confidence threshold during inference

#### Poor Performance
- Increase training epochs
- Use data augmentation
- Try larger model architecture
- Check dataset quality and annotations

### Debug Mode

```bash
# Enable verbose logging
python train_yolov8_improved.py --config training_config.yaml
# Check training.log for detailed information
```

## 🔗 Integration with Crop Insect Detection

This segmentation system works seamlessly with the [crop_Insect_Annotation](https://github.com/Iamyashsiwach/crop_Insect_Annotation) detection system:

### Combined Pipeline

```python
# Load both models
disease_model = YOLO('crop_disease_seg/weights/best.pt')
insect_model = YOLO('crop_insect_det/weights/best.pt')

# Process image with both models
disease_results = disease_model.predict(image)
insect_results = insect_model.predict(image)

# Combine results for comprehensive analysis
print(f"Diseases detected: {len(disease_results[0].masks.data) if disease_results[0].masks else 0}")
print(f"Insects detected: {len(insect_results[0].boxes) if insect_results[0].boxes else 0}")
```

## 📊 Performance Benchmarks

### Training Performance
- **RTX 3080**: ~2 hours for 200 epochs (640px, batch 16)
- **RTX 4090**: ~1.2 hours for 200 epochs (640px, batch 32)
- **CPU Only**: ~24 hours for 200 epochs (not recommended)

### Model Performance (Example)
- **mAP50**: 0.85-0.92 (depending on dataset quality)
- **mAP50-95**: 0.65-0.75
- **Inference Speed**: 15-30 FPS (RTX 3080, 640px)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the amazing YOLO implementation
- [CVAT](https://github.com/opencv/cvat) for annotation tools
- The agricultural computer vision community for inspiration and feedback

## 📞 Support

For questions, issues, or contributions:
- Open an [issue](https://github.com/Iamyashsiwach/Crop_Disease_Annotation/issues)
- Check the [troubleshooting guide](#troubleshooting)
- Review the [TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md) for advanced topics

---

**Happy Training! 🌱🤖** 