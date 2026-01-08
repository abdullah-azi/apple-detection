# Apple Detection with YOLO

A YOLO (You Only Look Once) object detection model trained to detect apples in images. This project is designed to run on Google Colab and can be easily saved to GitHub.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
  - [Google Colab Setup](#google-colab-setup)
  - [GitHub Repository Setup](#github-repository-setup)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a YOLO-based object detection model specifically trained to detect apples in images. The model is built using YOLOv8 and trained on a comprehensive fruit detection dataset. While the dataset includes multiple fruit classes (Apple, Banana, Grape, Orange, Pineapple, Watermelon), this project focuses on apple detection.

### Key Features

- **YOLOv8 Architecture**: Utilizes the latest YOLO architecture for fast and accurate object detection
- **Google Colab Compatible**: Fully configured to run on Google Colab with GPU support
- **Comprehensive Dataset**: Trained on 8,479 annotated images
- **Easy GitHub Integration**: Simple setup to save and version control your trained models

## 📦 Dataset

The dataset is located in the `fruit-detection-dataset/` directory and contains:

- **Total Images**: 8,479 images
- **Format**: YOLOv8 format (images + corresponding annotation files)
- **Classes**: 6 fruit classes (Apple, Banana, Grape, Orange, Pineapple, Watermelon)
- **Splits**:
  - Training: ~7,108 images
  - Validation: ~914 images
  - Test: ~457 images

### Dataset Structure

```
fruit-detection-dataset/
└── Fruits-detection/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    ├── data.yaml
    └── yolov8s.pt (pretrained model)
```

### Dataset Preprocessing

The dataset has been preprocessed with:
- Auto-orientation of pixel data (EXIF-orientation stripping)
- Resize to 640x640 (Stretch)
- 50% probability of horizontal flip augmentation

## 📁 Project Structure

```
apple-detection/
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
├── fruit-detection-dataset/           # Dataset directory
│   └── Fruits-detection/              # Main dataset folder
│       ├── train/                     # Training images and labels
│       ├── valid/                     # Validation images and labels
│       ├── test/                      # Test images and labels
│       ├── data.yaml                  # Dataset configuration
│       └── yolov8s.pt                 # Pretrained YOLOv8 model
├── checkpoints/                       # Saved model checkpoints
│   └── .gitkeep
├── results/                           # Training results and outputs
│   └── .gitkeep
├── documents/                         # Documentation
│   └── Repo Cred/
│       └── set_repo_credentials.md    # GitHub setup guide
└── notebooks/                         # Jupyter notebooks (if any)
```

## 🚀 Setup Instructions

### Google Colab Setup

1. **Open Google Colab**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Create a new notebook

2. **Mount Google Drive** (if dataset is stored there)
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Install Ultralytics YOLO**
   ```python
   !pip install ultralytics
   ```

4. **Upload Dataset**
   - Option 1: Upload the `fruit-detection-dataset` folder to your Google Drive
   - Option 2: Clone this repository in Colab:
   ```python
   !git clone https://github.com/your-username/apple-detection.git
   ```

5. **Update data.yaml for Colab**
   - Update the paths in `data.yaml` to match your Colab environment:
   ```yaml
   names:
     - Apple
     - Banana
     - Grape
     - Orange
     - Pineapple
     - Watermelon
   nc: 6
   train: /content/drive/MyDrive/apple-detection/fruit-detection-dataset/Fruits-detection/train/images
   val: /content/drive/MyDrive/apple-detection/fruit-detection-dataset/Fruits-detection/valid/images
   test: /content/drive/MyDrive/apple-detection/fruit-detection-dataset/Fruits-detection/test/images
   ```

6. **Verify GPU Availability**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
   ```

### GitHub Repository Setup

To save your trained models and code to GitHub, follow the instructions in `documents/Repo Cred/set_repo_credentials.md`.

#### Quick Setup Steps:

1. **Navigate to your repository**
   ```powershell
   cd D:\Drive\colab-projects\apple-detection
   ```

2. **Set the correct remote** (choose based on your account):
   
   **Personal repo (`abdullah-azi`):**
   ```powershell
   git remote set-url origin git@github-personal:abdullah-azi/apple-detection.git
   ```
   
   **Work repo (`SerenitysSlave`):**
   ```powershell
   git remote set-url origin git@github-second:SerenitysSlave/apple-detection.git
   ```

3. **Set local Git identity**:
   
   **Personal repo:**
   ```powershell
   git config user.name "abdullah-azi"
   git config user.email "syed.abdullahazi@gmail.com"
   ```
   
   **Work repo:**
   ```powershell
   git config user.name "SerenitysSlave"
   git config user.email "i221186@nu.edu.pk"
   ```

4. **Verify remote configuration:**
   ```powershell
   git remote -v
   ```

## 🏋️ Training the Model

### Basic Training

In Google Colab, use the following code to train your model:

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8s.pt')  # or yolov8n.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Train the model
results = model.train(
    data='/content/path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='apple_detection',
    project='/content/drive/MyDrive/apple-detection/results'
)
```

### Advanced Training Configuration

```python
from ultralytics import YOLO

# Initialize model
model = YOLO('yolov8s.pt')

# Train with custom parameters
results = model.train(
    data='/content/path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU device
    workers=8,
    patience=50,  # Early stopping patience
    save=True,
    save_period=10,  # Save checkpoint every 10 epochs
    val=True,  # Validate during training
    plots=True,  # Generate training plots
    name='apple_detection_v1',
    project='/content/drive/MyDrive/apple-detection/results',
    exist_ok=True,  # Overwrite existing project
    pretrained=True,
    optimizer='SGD',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0
)
```

### Training Tips

- **Start with a smaller model** (yolov8n.pt) for faster iteration
- **Use appropriate batch size** based on your GPU memory (16-32 for Colab)
- **Monitor training** using TensorBoard or the generated plots
- **Save checkpoints regularly** to avoid losing progress
- **Use early stopping** to prevent overfitting

## 📊 Model Evaluation

After training, evaluate your model:

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('/content/drive/MyDrive/apple-detection/results/apple_detection/weights/best.pt')

# Validate on test set
metrics = model.val(
    data='/content/path/to/data.yaml',
    split='test',
    imgsz=640,
    conf=0.25,
    iou=0.45
)

# Print metrics
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.mp}")
print(f"Recall: {metrics.box.mr}")
```

## 💻 Usage

### Inference on Images

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('/path/to/best.pt')

# Run inference
results = model('/path/to/image.jpg')

# Display results
results[0].show()

# Save results
results[0].save('/path/to/output.jpg')
```

### Inference on Videos

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('/path/to/best.pt')

# Run inference on video
results = model('/path/to/video.mp4')

# Save output video
results[0].save('/path/to/output.mp4')
```

### Filter for Apple Detection Only

If you want to detect only apples (class 0), you can filter the results:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('/path/to/best.pt')

# Run inference with class filter (Apple is class 0)
results = model('/path/to/image.jpg', classes=[0])

# Display results
results[0].show()
```

### Export Model

Export your model to different formats:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('/path/to/best.pt')

# Export to ONNX
model.export(format='onnx')

# Export to TensorRT
model.export(format='engine')

# Export to CoreML
model.export(format='coreml')
```

## 📈 Results

Training results will be saved in the `results/` directory, including:

- **Training curves**: Loss plots, mAP curves
- **Validation metrics**: Precision, Recall, mAP50, mAP50-95
- **Confusion matrix**: Class-wise performance
- **Sample predictions**: Validation set predictions
- **Model weights**: `best.pt` (best validation) and `last.pt` (last epoch)

### Expected Performance

With proper training, you should expect:
- **mAP50**: > 0.85 for apple detection
- **Precision**: > 0.80
- **Recall**: > 0.75

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use a smaller model (yolov8n.pt instead of yolov8s.pt)
   - Reduce image size

2. **Dataset Path Errors**
   - Verify paths in `data.yaml` are correct
   - Ensure paths use forward slashes (`/`) in Colab
   - Check that images and labels are in correct directories

3. **GitHub Push Issues**
   - Verify SSH keys are set up correctly
   - Check remote URL matches your account
   - Ensure Git identity is configured correctly

4. **Model Not Saving**
   - Check disk space in Colab
   - Verify save path is writable
   - Ensure `save=True` in training parameters

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the YOLO implementation
- The fruit detection dataset creators
- Google Colab for providing free GPU resources

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Training! 🍎**
