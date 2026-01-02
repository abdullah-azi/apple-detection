# 🍎 Apple Detection Using Object Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive computer vision project focused on building a basic object detection system to automatically identify and localize apples in images. This project emphasizes learning the complete computer vision pipeline—from dataset preparation to training, evaluation, and inference.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Results](#results)
- [Technologies](#technologies)
- [Learning Objectives](#learning-objectives)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a single-class object detection system that:

- **Detects** whether apples are present in images
- **Localizes** apples by drawing bounding boxes around them
- **Assigns** confidence scores to each detected apple

> **Note**: This is a learning-focused project designed to understand object detection fundamentals rather than achieve production-level performance.

## ✨ Features

- 🔍 Single-class object detection (apples)
- 📦 Support for multiple annotation formats (YOLO, Pascal VOC, COCO)
- 🎨 Data augmentation pipeline
- 📊 Comprehensive evaluation metrics (IoU, Precision, Recall, mAP)
- 🖼️ Visualization tools for predictions
- 💾 Model checkpointing and resume training
- 🚀 Easy-to-use inference pipeline

## 📁 Project Structure

```
apple-detection/
├── data/
│   ├── images/          # Training/validation/test images
│   ├── annotations/     # Bounding box annotations
│   └── splits/          # Train/val/test split files
├── src/                 # Source code
│   ├── dataset.py       # Dataset class and data loading
│   ├── model.py         # Model architecture
│   ├── train.py         # Training script
│   ├── evaluate.py      # Evaluation metrics
│   ├── inference.py     # Inference on new images
│   └── utils.py         # Helper functions
├── configs/             # Configuration files
├── checkpoints/          # Saved model weights
├── results/             # Output visualizations
├── notebooks/            # Jupyter notebooks
├── Documents/            # Project documentation
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- pip or conda package manager

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd apple-detection
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

1. **Prepare your dataset:**
   - Place images in `data/images/`
   - Place annotations in `data/annotations/`
   - Create train/val/test splits in `data/splits/`

2. **Configure training parameters:**
   - Edit `configs/config.yaml` with your settings

3. **Train the model:**
   ```bash
   python src/train.py --config configs/config.yaml
   ```

4. **Run inference:**
   ```bash
   python src/inference.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pth
   ```

## 📊 Dataset

The dataset should contain:

- **Images**: Various images of apples in different environments (trees, baskets, markets, etc.)
- **Annotations**: Bounding box coordinates and class labels
- **Formats Supported**:
  - YOLO format (`.txt`)
  - Pascal VOC format (`.xml`)
  - COCO format (`.json`)

### Dataset Split

- **Training set**: For model learning
- **Validation set**: For hyperparameter tuning
- **Test set**: For final evaluation

## 🏋️ Training

The training process includes:

- Image preprocessing (resize, normalize)
- Data augmentation (flip, scale, brightness)
- Loss computation (localization + classification)
- Gradient descent optimization
- Validation monitoring
- Checkpoint saving

### Training Command

```bash
python src/train.py --config configs/config.yaml
```

### Key Hyperparameters

- Learning rate
- Batch size
- Number of epochs
- Data augmentation settings
- Model architecture selection

## 📈 Evaluation

Model performance is evaluated using:

- **IoU (Intersection over Union)**: Measures bounding box accuracy
- **Precision**: Ratio of correct detections to total detections
- **Recall**: Ratio of detected apples to total apples
- **mAP (mean Average Precision)**: Overall detection quality metric

### Evaluation Command

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth --data data/test
```

## 🔮 Inference

Run detection on new images:

```bash
python src/inference.py \
    --image path/to/image.jpg \
    --checkpoint checkpoints/best_model.pth \
    --output results/detection.jpg \
    --confidence 0.5
```

The output will include:
- Bounding boxes drawn on the image
- Confidence scores for each detection
- Saved visualization in the results folder

## 📊 Results

Visual results and evaluation metrics are saved in the `results/` directory:

- Detection visualizations
- Evaluation reports
- Training curves
- Comparison plots

## 🛠️ Technologies

- **Python**: Core programming language
- **PyTorch / TensorFlow**: Deep learning framework
- **OpenCV**: Image processing
- **NumPy**: Numerical computations
- **Matplotlib / Seaborn**: Visualization
- **Pillow**: Image handling
- **Albumentations**: Data augmentation (optional)

## 🎓 Learning Objectives

By completing this project, you will understand:

- ✅ How object detection differs from image classification
- ✅ Working with labeled datasets containing bounding boxes
- ✅ Training detection models from scratch or using transfer learning
- ✅ Evaluating detection performance using standard metrics
- ✅ Running inference on new, unseen images
- ✅ The complete computer vision pipeline

### Transferable Skills

Once you can detect apples, you can apply the same principles to:
- 🚗 Car detection
- 👤 Face detection
- 🏥 Medical anomaly detection
- And many more applications!

## 📝 Scope & Limitations

- **Single-class detection**: Only apples (not multiple fruit types)
- **Performance**: Depends on dataset quality and size
- **Optimization**: Not optimized for real-time or production deployment
- **Purpose**: Educational and learning-focused

> **Remember**: This is a learning project, not a startup pitch. The goal is understanding, not perfection.

## 🤝 Contributing

This is a learning project, but suggestions and improvements are welcome! Feel free to:

- Report issues
- Suggest enhancements
- Share your results
- Improve documentation

## 📚 Additional Resources

- [Project Overview](Documents/Project_Overview.md) - Detailed project documentation
- [Object Detection Tutorials](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [YOLO Paper](https://arxiv.org/abs/1506.02640)
- [COCO Dataset](https://cocodataset.org/)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Pre-trained models and architectures from the computer vision community
- Dataset creators and contributors
- Open-source tools and libraries

---

**Happy Learning! 🍎🔍**

> *"Why are bounding boxes always slightly off-center? It's not personal. It's math."*

