# Configuration Specification: Apple Detection System

## 📋 Document Information

- **Project**: Apple Detection Using Object Detection
- **Version**: 1.0
- **Date**: January 2026
- **Status**: Draft
- **Author**: Project Team

---

## 📑 Table of Contents

1. [Introduction](#1-introduction)
2. [Configuration File Format](#2-configuration-file-format)
3. [Data Configuration](#3-data-configuration)
4. [Model Configuration](#4-model-configuration)
5. [Training Configuration](#5-training-configuration)
6. [Evaluation Configuration](#6-evaluation-configuration)
7. [Inference Configuration](#7-inference-configuration)
8. [Preprocessing Configuration](#8-preprocessing-configuration)
9. [Augmentation Configuration](#9-augmentation-configuration)
10. [Path Configuration](#10-path-configuration)
11. [Hardware Configuration](#11-hardware-configuration)
12. [Logging Configuration](#12-logging-configuration)
13. [Default Configurations](#13-default-configurations)
14. [Configuration Validation](#14-configuration-validation)
15. [Environment Variables](#15-environment-variables)

---

## 1. Introduction

### 1.1 Purpose
This document specifies all configurable parameters, hyperparameters, and settings for the Apple Detection System. It defines the structure of configuration files and provides default values for all parameters.

### 1.2 Scope
This specification covers:
- Configuration file formats (YAML/JSON)
- All configurable parameters
- Default values and ranges
- Configuration validation rules
- Environment-specific settings

### 1.3 Configuration Philosophy
- **Centralized**: All settings in configuration files
- **Overrideable**: Command-line arguments can override config
- **Validated**: All parameters validated on load
- **Documented**: Clear descriptions for each parameter

---

## 2. Configuration File Format

### 2.1 Supported Formats

#### YAML Format (Recommended)
- **Extension**: `.yaml` or `.yml`
- **Advantages**: Human-readable, supports comments
- **Location**: `configs/config.yaml`

#### JSON Format (Alternative)
- **Extension**: `.json`
- **Advantages**: Machine-readable, widely supported
- **Location**: `configs/config.json`

### 2.2 Configuration File Structure

#### Hierarchical Structure
```yaml
# Top-level sections
data:
  # Data-related settings
  
model:
  # Model architecture settings
  
training:
  # Training hyperparameters
  
evaluation:
  # Evaluation settings
  
inference:
  # Inference settings
  
paths:
  # File and directory paths
  
hardware:
  # Hardware settings
```

### 2.3 Configuration Loading

#### Loading Priority
1. Default configuration (hardcoded)
2. Configuration file (YAML/JSON)
3. Command-line arguments (highest priority)
4. Environment variables (if applicable)

---

## 3. Data Configuration

### 3.1 Dataset Paths

#### Configuration Parameters
```yaml
data:
  # Root data directory
  root_dir: "data"
  
  # Image directories
  images:
    train: "data/images/train"
    val: "data/images/val"
    test: "data/images/test"
  
  # Annotation directories
  annotations:
    train: "data/annotations/train"
    val: "data/annotations/val"
    test: "data/annotations/test"
  
  # Split files
  splits:
    train: "data/splits/train.txt"
    val: "data/splits/val.txt"
    test: "data/splits/test.txt"
```

#### Default Values
- `root_dir`: `"data"`
- All subdirectories relative to `root_dir`

### 3.2 Annotation Format

#### Configuration Parameters
```yaml
data:
  annotation_format: "yolo"  # Options: "yolo", "voc", "coco"
  class_names: ["apple"]
  class_ids: [0]
  num_classes: 1
```

#### Valid Values
- `annotation_format`: `"yolo"`, `"voc"`, `"coco"`
- `class_names`: List of class names
- `class_ids`: List of corresponding class IDs
- `num_classes`: Number of classes (1 for single-class)

#### Default Values
- `annotation_format`: `"yolo"`
- `class_names`: `["apple"]`
- `class_ids`: `[0]`
- `num_classes`: `1`

### 3.3 Dataset Statistics

#### Configuration Parameters
```yaml
data:
  # Dataset statistics (auto-calculated or manual)
  stats:
    train_size: null  # Auto-calculated if null
    val_size: null
    test_size: null
    total_images: null
    total_annotations: null
```

---

## 4. Model Configuration

### 4.1 Model Architecture

#### Configuration Parameters
```yaml
model:
  # Model type
  architecture: "yolo"  # Options: "yolo", "ssd", "faster_rcnn"
  version: "v5"  # For YOLO: "v5", "v8"
  size: "s"  # Options: "n" (nano), "s" (small), "m" (medium), "l" (large), "x" (xlarge)
  
  # Backbone
  backbone: "resnet50"  # Options: "resnet50", "mobilenet", "efficientnet"
  pretrained: true  # Use pre-trained weights
  pretrained_weights: "imagenet"  # Source of pre-trained weights
```

#### Valid Values
- `architecture`: `"yolo"`, `"ssd"`, `"faster_rcnn"`
- `version`: Model version (architecture-specific)
- `size`: `"n"`, `"s"`, `"m"`, `"l"`, `"x"`
- `backbone`: `"resnet50"`, `"mobilenet"`, `"efficientnet"`
- `pretrained`: `true` or `false`
- `pretrained_weights`: `"imagenet"` or path to weights file

#### Default Values
- `architecture`: `"yolo"`
- `version`: `"v5"`
- `size`: `"s"`
- `backbone`: `"resnet50"`
- `pretrained`: `true`
- `pretrained_weights`: `"imagenet"`

### 4.2 Model Input/Output

#### Configuration Parameters
```yaml
model:
  # Input configuration
  input_size: [640, 640]  # [width, height]
  input_channels: 3  # RGB
  normalize: true
  normalization_mean: [0.485, 0.456, 0.406]  # ImageNet stats
  normalization_std: [0.229, 0.224, 0.225]  # ImageNet stats
  
  # Output configuration
  num_classes: 1
  num_anchors: 3  # For anchor-based models
  confidence_threshold: 0.25  # Default confidence threshold
  nms_threshold: 0.45  # Non-Maximum Suppression threshold
```

#### Default Values
- `input_size`: `[640, 640]`
- `input_channels`: `3`
- `normalize`: `true`
- `normalization_mean`: `[0.485, 0.456, 0.406]` (ImageNet)
- `normalization_std`: `[0.229, 0.224, 0.225]` (ImageNet)
- `num_classes`: `1`
- `confidence_threshold`: `0.25`
- `nms_threshold`: `0.45`

### 4.3 Model Initialization

#### Configuration Parameters
```yaml
model:
  # Weight initialization
  init_weights: true  # Initialize weights if not pretrained
  init_method: "kaiming"  # Options: "kaiming", "xavier", "normal"
  
  # Model-specific parameters
  anchor_sizes: null  # Auto-generated if null
  aspect_ratios: null  # Auto-generated if null
```

---

## 5. Training Configuration

### 5.1 Training Hyperparameters

#### Configuration Parameters
```yaml
training:
  # Basic training settings
  num_epochs: 100
  batch_size: 16
  num_workers: 4  # DataLoader workers
  pin_memory: true  # For GPU training
  
  # Learning rate
  learning_rate: 0.001
  lr_scheduler: "cosine"  # Options: "step", "cosine", "plateau", "none"
  lr_warmup: true
  lr_warmup_epochs: 3
  lr_warmup_factor: 0.1
  
  # Optimizer
  optimizer: "adam"  # Options: "sgd", "adam", "adamw"
  momentum: 0.937  # For SGD
  weight_decay: 0.0005
  
  # Gradient settings
  gradient_clip: 10.0  # Clip gradients (0 to disable)
  accumulate_grad_batches: 1  # Gradient accumulation
```

#### Valid Ranges
- `num_epochs`: `1` to `1000`
- `batch_size`: `1` to `128` (depends on GPU memory)
- `learning_rate`: `0.00001` to `0.1`
- `weight_decay`: `0.0` to `0.01`
- `gradient_clip`: `0.0` (disabled) to `100.0`

#### Default Values
- `num_epochs`: `100`
- `batch_size`: `16`
- `num_workers`: `4`
- `learning_rate`: `0.001`
- `lr_scheduler`: `"cosine"`
- `optimizer`: `"adam"`
- `weight_decay`: `0.0005`
- `gradient_clip`: `10.0`

### 5.2 Loss Function Configuration

#### Configuration Parameters
```yaml
training:
  loss:
    # Loss type
    type: "combined"  # Options: "combined", "focal", "giou"
    
    # Loss weights
    localization_weight: 1.0
    classification_weight: 1.0
    confidence_weight: 1.0
    
    # Focal loss parameters (if used)
    focal_alpha: 0.25
    focal_gamma: 2.0
    
    # IoU loss type
    iou_type: "giou"  # Options: "iou", "giou", "diou", "ciou"
```

#### Default Values
- `type`: `"combined"`
- `localization_weight`: `1.0`
- `classification_weight`: `1.0`
- `confidence_weight`: `1.0`
- `iou_type`: `"giou"`

### 5.3 Training Monitoring

#### Configuration Parameters
```yaml
training:
  # Validation
  val_frequency: 1  # Validate every N epochs
  val_batch_size: 32
  
  # Checkpointing
  save_checkpoint: true
  checkpoint_frequency: 5  # Save every N epochs
  save_best_only: true  # Save only best model
  monitor_metric: "val_loss"  # Metric to monitor
  monitor_mode: "min"  # "min" or "max"
  
  # Early stopping
  early_stopping: true
  early_stopping_patience: 10  # Stop if no improvement for N epochs
  early_stopping_min_delta: 0.001
  
  # Resume training
  resume_from_checkpoint: null  # Path to checkpoint or null
  resume_optimizer: true
  resume_scheduler: true
```

#### Default Values
- `val_frequency`: `1`
- `save_checkpoint`: `true`
- `checkpoint_frequency`: `5`
- `save_best_only`: `true`
- `monitor_metric`: `"val_loss"`
- `early_stopping`: `true`
- `early_stopping_patience`: `10`

### 5.4 Training Augmentation

#### Configuration Parameters
```yaml
training:
  augmentation:
    enabled: true
    # See Augmentation Configuration section
```

---

## 6. Evaluation Configuration

### 6.1 Evaluation Metrics

#### Configuration Parameters
```yaml
evaluation:
  # Metrics to compute
  metrics:
    - "iou"
    - "precision"
    - "recall"
    - "map"
  
  # IoU threshold
  iou_threshold: 0.5  # Standard IoU threshold
  iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # For mAP
  
  # Confidence threshold
  confidence_threshold: 0.25
  
  # Matching strategy
  matching_strategy: "best"  # Options: "best", "greedy"
  max_detections_per_image: 100
```

#### Default Values
- `iou_threshold`: `0.5`
- `confidence_threshold`: `0.25`
- `matching_strategy`: `"best"`
- `max_detections_per_image`: `100`

### 6.2 Evaluation Output

#### Configuration Parameters
```yaml
evaluation:
  # Output settings
  save_predictions: true
  save_visualizations: true
  visualization_format: "png"  # Options: "png", "jpg"
  
  # Report settings
  generate_report: true
  report_format: "markdown"  # Options: "markdown", "json", "html"
  report_path: "results/evaluation_report.md"
```

#### Default Values
- `save_predictions`: `true`
- `save_visualizations`: `true`
- `visualization_format`: `"png"`
- `generate_report`: `true`
- `report_format`: `"markdown"`

---

## 7. Inference Configuration

### 7.1 Inference Settings

#### Configuration Parameters
```yaml
inference:
  # Model loading
  checkpoint_path: "checkpoints/best_model.pth"
  device: "auto"  # Options: "auto", "cpu", "cuda", "cuda:0"
  
  # Detection settings
  confidence_threshold: 0.25
  nms_threshold: 0.45
  max_detections: 100
  
  # Input/Output
  input_size: [640, 640]
  output_format: "json"  # Options: "json", "image", "both"
  save_visualizations: true
```

#### Default Values
- `device`: `"auto"` (auto-detects GPU if available)
- `confidence_threshold`: `0.25`
- `nms_threshold`: `0.45`
- `max_detections`: `100`
- `output_format`: `"json"`

### 7.2 Batch Inference

#### Configuration Parameters
```yaml
inference:
  batch:
    enabled: false
    batch_size: 8
    input_directory: null
    output_directory: "results/inference"
```

---

## 8. Preprocessing Configuration

### 8.1 Image Preprocessing

#### Configuration Parameters
```yaml
preprocessing:
  # Resize settings
  resize:
    enabled: true
    target_size: [640, 640]
    maintain_aspect_ratio: false  # If true, pad instead of stretch
    padding_mode: "constant"  # Options: "constant", "edge", "reflect"
    padding_value: 114  # Gray padding value (0-255)
  
  # Normalization
  normalize: true
  normalization_method: "imagenet"  # Options: "imagenet", "custom", "none"
  custom_mean: [0.5, 0.5, 0.5]
  custom_std: [0.5, 0.5, 0.5]
  
  # Color space
  color_space: "rgb"  # Options: "rgb", "bgr", "grayscale"
  convert_to_rgb: true  # Convert BGR to RGB if needed
```

#### Default Values
- `target_size`: `[640, 640]`
- `maintain_aspect_ratio`: `false`
- `normalize`: `true`
- `normalization_method`: `"imagenet"`
- `color_space`: `"rgb"`

### 8.2 Annotation Preprocessing

#### Configuration Parameters
```yaml
preprocessing:
  annotation:
    # Coordinate transformation
    convert_format: true  # Convert to model's required format
    validate_coordinates: true
    clip_coordinates: true  # Clip to image bounds
    
    # Filtering
    filter_small_boxes: true
    min_box_size: [10, 10]  # Minimum [width, height] in pixels
    filter_large_boxes: false
    max_box_size: null  # None or [width, height]
```

#### Default Values
- `convert_format`: `true`
- `validate_coordinates`: `true`
- `clip_coordinates`: `true`
- `filter_small_boxes`: `true`
- `min_box_size`: `[10, 10]`

---

## 9. Augmentation Configuration

### 9.1 Geometric Augmentations

#### Configuration Parameters
```yaml
augmentation:
  # Enable/disable
  enabled: true
  probability: 0.5  # Probability of applying augmentation
  
  # Horizontal flip
  horizontal_flip:
    enabled: true
    probability: 0.5
  
  # Rotation
  rotation:
    enabled: true
    probability: 0.5
    max_angle: 15  # Degrees (-max_angle to +max_angle)
  
  # Scaling
  scale:
    enabled: true
    probability: 0.5
    min_scale: 0.8
    max_scale: 1.2
  
  # Translation
  translation:
    enabled: false
    probability: 0.3
    max_shift: 0.1  # Fraction of image size
```

#### Default Values
- `enabled`: `true`
- `horizontal_flip.enabled`: `true`
- `rotation.enabled`: `true`
- `rotation.max_angle`: `15`
- `scale.enabled`: `true`
- `scale.min_scale`: `0.8`
- `scale.max_scale`: `1.2`

### 9.2 Color Augmentations

#### Configuration Parameters
```yaml
augmentation:
  # Brightness
  brightness:
    enabled: true
    probability: 0.5
    factor_range: [0.8, 1.2]  # Multiply brightness by factor
  
  # Contrast
  contrast:
    enabled: true
    probability: 0.5
    factor_range: [0.8, 1.2]
  
  # Saturation
  saturation:
    enabled: true
    probability: 0.5
    factor_range: [0.8, 1.2]
  
  # Hue
  hue:
    enabled: false
    probability: 0.3
    max_shift: 0.1  # Fraction of hue range
```

#### Default Values
- `brightness.enabled`: `true`
- `contrast.enabled`: `true`
- `saturation.enabled`: `true`
- `hue.enabled`: `false`

### 9.3 Advanced Augmentations

#### Configuration Parameters
```yaml
augmentation:
  # Mosaic (combine 4 images)
  mosaic:
    enabled: false
    probability: 0.5
  
  # Mixup (blend 2 images)
  mixup:
    enabled: false
    probability: 0.3
    alpha: 0.2
  
  # Cutout (random rectangular cutouts)
  cutout:
    enabled: false
    probability: 0.3
    num_holes: 1
    max_hole_size: 0.1  # Fraction of image
```

---

## 10. Path Configuration

### 10.1 Directory Paths

#### Configuration Parameters
```yaml
paths:
  # Project root (auto-detected or set)
  project_root: null  # Auto-detected if null
  
  # Data paths
  data_root: "data"
  
  # Model paths
  checkpoints_dir: "checkpoints"
  models_dir: "models"
  
  # Output paths
  results_dir: "results"
  logs_dir: "logs"
  visualizations_dir: "results/visualizations"
  
  # Config paths
  configs_dir: "configs"
```

#### Default Values
- All paths relative to project root
- `data_root`: `"data"`
- `checkpoints_dir`: `"checkpoints"`
- `results_dir`: `"results"`
- `logs_dir`: `"logs"`

### 10.2 File Paths

#### Configuration Parameters
```yaml
paths:
  # Checkpoint files
  best_checkpoint: "checkpoints/best_model.pth"
  latest_checkpoint: "checkpoints/latest_model.pth"
  
  # Log files
  train_log: "logs/training.log"
  eval_log: "logs/evaluation.log"
  
  # Result files
  eval_report: "results/evaluation_report.md"
  predictions_file: "results/predictions.json"
```

---

## 11. Hardware Configuration

### 11.1 Device Settings

#### Configuration Parameters
```yaml
hardware:
  # Device selection
  device: "auto"  # Options: "auto", "cpu", "cuda", "cuda:0", "cuda:1"
  cuda_device: 0  # GPU index if using CUDA
  
  # Mixed precision training
  mixed_precision: true  # Use FP16 for faster training
  amp_level: "O1"  # Options: "O0", "O1", "O2", "O3"
  
  # Memory optimization
  pin_memory: true  # For DataLoader
  non_blocking: true  # For data transfer
```

#### Default Values
- `device`: `"auto"` (auto-detects GPU)
- `mixed_precision`: `true`
- `pin_memory`: `true`

### 11.2 Performance Settings

#### Configuration Parameters
```yaml
hardware:
  # DataLoader settings
  num_workers: 4  # Number of data loading workers
  prefetch_factor: 2  # Number of batches to prefetch
  
  # GPU settings
  allow_tf32: true  # Use TensorFloat-32 on Ampere GPUs
  benchmark_mode: true  # cuDNN benchmark mode
```

#### Default Values
- `num_workers`: `4`
- `prefetch_factor`: `2`
- `allow_tf32`: `true`

---

## 12. Logging Configuration

### 12.1 Logging Settings

#### Configuration Parameters
```yaml
logging:
  # Logging level
  level: "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
  
  # Log destinations
  console: true
  file: true
  log_file: "logs/training.log"
  
  # Log format
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
  
  # Logging frequency
  log_every_n_steps: 10  # Log every N training steps
  log_every_n_epochs: 1  # Log every N epochs
```

#### Default Values
- `level`: `"INFO"`
- `console`: `true`
- `file`: `true`
- `log_every_n_steps`: `10`

### 12.2 TensorBoard/Visualization

#### Configuration Parameters
```yaml
logging:
  # TensorBoard (if using)
  tensorboard:
    enabled: false
    log_dir: "logs/tensorboard"
    update_frequency: 100  # Update every N steps
  
  # WandB (if using)
  wandb:
    enabled: false
    project: "apple-detection"
    entity: null
    run_name: null
```

---

## 13. Default Configurations

### 13.1 Complete Default Configuration (YAML)

```yaml
# Apple Detection System - Default Configuration

# Data Configuration
data:
  root_dir: "data"
  images:
    train: "data/images/train"
    val: "data/images/val"
    test: "data/images/test"
  annotations:
    train: "data/annotations/train"
    val: "data/annotations/val"
    test: "data/annotations/test"
  splits:
    train: "data/splits/train.txt"
    val: "data/splits/val.txt"
    test: "data/splits/test.txt"
  annotation_format: "yolo"
  class_names: ["apple"]
  class_ids: [0]
  num_classes: 1

# Model Configuration
model:
  architecture: "yolo"
  version: "v5"
  size: "s"
  backbone: "resnet50"
  pretrained: true
  pretrained_weights: "imagenet"
  input_size: [640, 640]
  input_channels: 3
  normalize: true
  normalization_mean: [0.485, 0.456, 0.406]
  normalization_std: [0.229, 0.224, 0.225]
  num_classes: 1
  confidence_threshold: 0.25
  nms_threshold: 0.45

# Training Configuration
training:
  num_epochs: 100
  batch_size: 16
  num_workers: 4
  pin_memory: true
  learning_rate: 0.001
  lr_scheduler: "cosine"
  lr_warmup: true
  lr_warmup_epochs: 3
  optimizer: "adam"
  weight_decay: 0.0005
  gradient_clip: 10.0
  val_frequency: 1
  save_checkpoint: true
  checkpoint_frequency: 5
  save_best_only: true
  monitor_metric: "val_loss"
  early_stopping: true
  early_stopping_patience: 10

# Evaluation Configuration
evaluation:
  metrics: ["iou", "precision", "recall", "map"]
  iou_threshold: 0.5
  confidence_threshold: 0.25
  save_visualizations: true

# Inference Configuration
inference:
  checkpoint_path: "checkpoints/best_model.pth"
  device: "auto"
  confidence_threshold: 0.25
  nms_threshold: 0.45

# Preprocessing Configuration
preprocessing:
  resize:
    enabled: true
    target_size: [640, 640]
    maintain_aspect_ratio: false
  normalize: true
  normalization_method: "imagenet"

# Augmentation Configuration
augmentation:
  enabled: true
  horizontal_flip:
    enabled: true
    probability: 0.5
  rotation:
    enabled: true
    probability: 0.5
    max_angle: 15
  scale:
    enabled: true
    probability: 0.5
    min_scale: 0.8
    max_scale: 1.2
  brightness:
    enabled: true
    probability: 0.5
  contrast:
    enabled: true
    probability: 0.5
  saturation:
    enabled: true
    probability: 0.5

# Path Configuration
paths:
  data_root: "data"
  checkpoints_dir: "checkpoints"
  results_dir: "results"
  logs_dir: "logs"

# Hardware Configuration
hardware:
  device: "auto"
  mixed_precision: true
  num_workers: 4

# Logging Configuration
logging:
  level: "INFO"
  console: true
  file: true
  log_file: "logs/training.log"
```

### 13.2 Minimal Configuration

For quick testing, minimal configuration:
```yaml
data:
  root_dir: "data"
  annotation_format: "yolo"

model:
  architecture: "yolo"
  size: "s"
  pretrained: true

training:
  num_epochs: 10
  batch_size: 8
  learning_rate: 0.001
```

---

## 14. Configuration Validation

### 14.1 Validation Rules

#### Required Parameters
- Data paths must exist or be creatable
- Model architecture must be valid
- Training hyperparameters must be in valid ranges
- Paths must be valid (not contain invalid characters)

#### Range Validation
- `batch_size`: Must be positive integer
- `learning_rate`: Must be between 0.00001 and 0.1
- `num_epochs`: Must be positive integer
- `confidence_threshold`: Must be between 0.0 and 1.0
- `iou_threshold`: Must be between 0.0 and 1.0

### 14.2 Validation Functions

#### Validation Checklist
- [ ] All required paths are specified
- [ ] All numeric values are in valid ranges
- [ ] All string values are valid options
- [ ] File paths are accessible
- [ ] Model architecture is supported
- [ ] Device is available (if specified)

---

## 15. Environment Variables

### 15.1 Supported Environment Variables

#### Configuration Override
```bash
# Data paths
export APPLE_DETECTION_DATA_ROOT="data"
export APPLE_DETECTION_CHECKPOINT_DIR="checkpoints"

# Hardware
export CUDA_VISIBLE_DEVICES=0
export APPLE_DETECTION_DEVICE="cuda"

# Logging
export APPLE_DETECTION_LOG_LEVEL="DEBUG"
```

### 15.2 Environment Variable Priority

Priority order (highest to lowest):
1. Command-line arguments
2. Configuration file
3. Environment variables
4. Default values

---

## 16. Configuration Examples

### 16.1 Training Configuration

#### Example: Fast Training (Small Model)
```yaml
model:
  architecture: "yolo"
  size: "n"  # Nano - smallest model

training:
  num_epochs: 50
  batch_size: 32
  learning_rate: 0.01
  lr_scheduler: "step"
```

#### Example: High Accuracy (Large Model)
```yaml
model:
  architecture: "yolo"
  size: "l"  # Large model

training:
  num_epochs: 200
  batch_size: 8
  learning_rate: 0.0001
  lr_scheduler: "cosine"
```

### 16.2 Inference Configuration

#### Example: High Confidence Detection
```yaml
inference:
  confidence_threshold: 0.5  # Higher threshold
  nms_threshold: 0.4
  max_detections: 50
```

#### Example: Sensitive Detection (Lower Threshold)
```yaml
inference:
  confidence_threshold: 0.1  # Lower threshold
  nms_threshold: 0.5
  max_detections: 200
```

---

## 17. Configuration Management

### 17.1 Configuration Files Organization

#### Recommended Structure
```
configs/
├── config.yaml          # Default configuration
├── config_train.yaml    # Training-specific
├── config_eval.yaml     # Evaluation-specific
├── config_inference.yaml # Inference-specific
└── configs/
    ├── small_model.yaml
    ├── large_model.yaml
    └── fast_training.yaml
```

### 17.2 Configuration Versioning

#### Version Control
- Track configuration files in Git
- Tag configurations with model versions
- Document configuration changes in changelog

---

## 18. References

### 18.1 Related Documents
- [Requirements Specification](Requirements.md)
- [Data Specification](Data_Specification.md)
- [System Architecture](Architecture.md) (to be created)

### 18.2 External Resources
- [YAML Specification](https://yaml.org/spec/)
- [JSON Specification](https://www.json.org/)
- [PyTorch Configuration Best Practices](https://pytorch.org/docs/stable/notes/serialization.html)

---

**Document End**

*This configuration specification serves as the definitive guide for all configurable parameters in the Apple Detection System. All configuration files should adhere to this specification.*

