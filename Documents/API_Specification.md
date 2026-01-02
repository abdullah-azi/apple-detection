# API/Interface Specification: Apple Detection System

## 📋 Document Information

- **Project**: Apple Detection Using Object Detection
- **Version**: 1.0
- **Date**: January 2026
- **Status**: Draft
- **Author**: Project Team

---

## 📑 Table of Contents

1. [Introduction](#1-introduction)
2. [API Design Principles](#2-api-design-principles)
3. [Dataset API](#3-dataset-api)
4. [Model API](#4-model-api)
5. [Training API](#5-training-api)
6. [Evaluation API](#6-evaluation-api)
7. [Inference API](#7-inference-api)
8. [Utility APIs](#8-utility-apis)
9. [Configuration API](#9-configuration-api)
10. [Data Transformation API](#10-data-transformation-api)
11. [Visualization API](#11-visualization-api)
12. [Error Handling API](#12-error-handling-api)
13. [Type Definitions](#13-type-definitions)
14. [Usage Examples](#14-usage-examples)

---

## 1. Introduction

### 1.1 Purpose
This document specifies all public APIs, interfaces, function signatures, and data structures for the Apple Detection System. It defines the contracts between modules and provides clear specifications for implementation.

### 1.2 Scope
This specification covers:
- All public function signatures
- Class interfaces and methods
- Input/output formats
- Data structures and types
- Error handling conventions
- Usage examples

### 1.3 API Philosophy
- **Clear and Consistent**: Uniform naming and structure
- **Type Hints**: All functions include type hints
- **Documentation**: Comprehensive docstrings
- **Error Handling**: Explicit error handling
- **Extensibility**: Easy to extend and modify

---

## 2. API Design Principles

### 2.1 Naming Conventions

#### Functions
- **snake_case**: `load_dataset()`, `train_model()`
- **Descriptive**: Clear purpose from name
- **Verb-based**: Action-oriented names

#### Classes
- **PascalCase**: `AppleDataset`, `DetectionModel`
- **Noun-based**: Represent entities

#### Constants
- **UPPER_SNAKE_CASE**: `DEFAULT_BATCH_SIZE`, `MAX_IMAGE_SIZE`

### 2.2 Parameter Ordering

#### Standard Order
1. Required positional parameters
2. Optional parameters with defaults
3. Keyword-only parameters
4. Variable arguments (*args, **kwargs)

### 2.3 Return Values

#### Return Types
- **Single Value**: Return directly
- **Multiple Values**: Return tuple
- **Complex Data**: Return dictionary or dataclass
- **None**: For operations without return value

### 2.4 Error Handling

#### Exception Types
- **ValueError**: Invalid parameter values
- **FileNotFoundError**: Missing files
- **RuntimeError**: Runtime errors
- **Custom Exceptions**: Domain-specific errors

---

## 3. Dataset API

### 3.1 Dataset Class

#### Class Definition
```python
class AppleDataset(Dataset):
    """
    Dataset class for apple detection.
    
    Args:
        images_dir: Path to images directory
        annotations_dir: Path to annotations directory
        annotation_format: Format of annotations ("yolo", "voc", "coco")
        transform: Optional data transformation pipeline
        augment: Whether to apply augmentation
    """
    
    def __init__(
        self,
        images_dir: Union[str, Path],
        annotations_dir: Union[str, Path],
        annotation_format: str = "yolo",
        transform: Optional[Callable] = None,
        augment: bool = False
    ) -> None:
        pass
    
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        pass
    
    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, target) where:
            - image: Preprocessed image tensor [C, H, W]
            - target: Dictionary containing:
                - boxes: Bounding boxes [N, 4]
                - labels: Class labels [N]
                - image_id: Image identifier
        """
        pass
```

### 3.2 Dataset Factory Functions

#### Create Dataset
```python
def create_dataset(
    split: str,
    config: Dict[str, Any]
) -> AppleDataset:
    """
    Create a dataset for the specified split.
    
    Args:
        split: Dataset split ("train", "val", "test")
        config: Configuration dictionary
        
    Returns:
        AppleDataset instance
        
    Raises:
        ValueError: If split is invalid
        FileNotFoundError: If data directories don't exist
    """
    pass
```

#### Create DataLoader
```python
def create_dataloader(
    dataset: AppleDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    collate_fn: Optional[Callable] = None
) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        collate_fn: Custom collate function
        
    Returns:
        DataLoader instance
    """
    pass
```

### 3.3 Annotation Parsing Functions

#### Parse YOLO Annotation
```python
def parse_yolo_annotation(
    annotation_path: Union[str, Path],
    image_width: int,
    image_height: int
) -> List[Dict[str, Any]]:
    """
    Parse YOLO format annotation file.
    
    Args:
        annotation_path: Path to annotation file
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        List of annotation dictionaries with keys:
        - class_id: Class ID
        - bbox: Bounding box [x_center, y_center, width, height] (normalized)
        - bbox_abs: Bounding box in absolute coordinates [x1, y1, x2, y2]
        
    Raises:
        FileNotFoundError: If annotation file doesn't exist
        ValueError: If annotation format is invalid
    """
    pass
```

#### Parse VOC Annotation
```python
def parse_voc_annotation(
    annotation_path: Union[str, Path]
) -> List[Dict[str, Any]]:
    """
    Parse Pascal VOC format annotation file.
    
    Args:
        annotation_path: Path to XML annotation file
        
    Returns:
        List of annotation dictionaries with keys:
        - class_name: Class name
        - bbox: Bounding box [x1, y1, x2, y2] (absolute)
        - difficult: Difficulty flag
        - truncated: Truncation flag
    """
    pass
```

#### Parse COCO Annotation
```python
def parse_coco_annotation(
    annotation_path: Union[str, Path],
    image_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Parse COCO format annotation file.
    
    Args:
        annotation_path: Path to JSON annotation file
        image_id: Optional image ID to filter annotations
        
    Returns:
        List of annotation dictionaries
    """
    pass
```

---

## 4. Model API

### 4.1 Model Class

#### Base Model Interface
```python
class DetectionModel(nn.Module):
    """
    Base class for object detection models.
    """
    
    def __init__(
        self,
        num_classes: int,
        input_size: Tuple[int, int] = (640, 640),
        pretrained: bool = True,
        **kwargs
    ) -> None:
        """
        Initialize detection model.
        
        Args:
            num_classes: Number of classes (1 for apple)
            input_size: Input image size (width, height)
            pretrained: Whether to use pretrained weights
            **kwargs: Additional model-specific parameters
        """
        pass
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Dictionary with keys:
            - boxes: Predicted boxes [B, N, 4]
            - scores: Confidence scores [B, N]
            - classes: Class predictions [B, N]
            - features: Feature maps (optional)
        """
        pass
    
    def load_weights(
        self,
        checkpoint_path: Union[str, Path],
        strict: bool = True
    ) -> None:
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly match state dict keys
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If checkpoint format is invalid
        """
        pass
    
    def save_weights(
        self,
        checkpoint_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Save model weights to checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
            optimizer: Optional optimizer state
            epoch: Optional epoch number
            metrics: Optional metrics dictionary
        """
        pass
```

### 4.2 Model Factory Functions

#### Create Model
```python
def create_model(
    architecture: str = "yolo",
    version: str = "v5",
    size: str = "s",
    num_classes: int = 1,
    pretrained: bool = True,
    **kwargs
) -> DetectionModel:
    """
    Create a detection model.
    
    Args:
        architecture: Model architecture ("yolo", "ssd", "faster_rcnn")
        version: Model version (architecture-specific)
        size: Model size ("n", "s", "m", "l", "x")
        num_classes: Number of classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model-specific parameters
        
    Returns:
        DetectionModel instance
        
    Raises:
        ValueError: If architecture or size is invalid
    """
    pass
```

#### Load Model from Checkpoint
```python
def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Tuple[DetectionModel, Optional[Dict[str, Any]]]:
    """
    Load model from checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Target device (auto-detected if None)
        strict: Whether to strictly match state dict
        
    Returns:
        Tuple of (model, checkpoint_info) where checkpoint_info contains:
        - epoch: Training epoch
        - metrics: Training metrics
        - optimizer_state: Optimizer state (if saved)
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint is invalid
    """
    pass
```

---

## 5. Training API

### 5.1 Training Function

#### Main Training Function
```python
def train_model(
    model: DetectionModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    callbacks: Optional[List[TrainingCallback]] = None
) -> Dict[str, Any]:
    """
    Train the detection model.
    
    Args:
        model: Detection model instance
        train_loader: Training data loader
        val_loader: Optional validation data loader
        config: Training configuration dictionary
        device: Training device (auto-detected if None)
        callbacks: Optional list of training callbacks
        
    Returns:
        Dictionary containing:
        - best_model_state: Best model state dict
        - training_history: Training metrics over epochs
        - best_epoch: Epoch with best validation metric
        - best_metrics: Best validation metrics
        
    Raises:
        RuntimeError: If training fails
    """
    pass
```

### 5.2 Training Loop Components

#### Training Step
```python
def training_step(
    model: DetectionModel,
    batch: Tuple[torch.Tensor, Dict[str, Any]],
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """
    Perform a single training step.
    
    Args:
        model: Detection model
        batch: Training batch (images, targets)
        loss_fn: Loss function
        optimizer: Optimizer
        device: Training device
        
    Returns:
        Dictionary with loss values:
        - total_loss: Total loss
        - localization_loss: Localization loss
        - classification_loss: Classification loss
        - confidence_loss: Confidence loss
    """
    pass
```

#### Validation Step
```python
def validation_step(
    model: DetectionModel,
    val_loader: DataLoader,
    loss_fn: Callable,
    device: torch.device,
    metrics_fn: Optional[Callable] = None
) -> Dict[str, float]:
    """
    Perform validation on validation set.
    
    Args:
        model: Detection model
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device for computation
        metrics_fn: Optional metrics calculation function
        
    Returns:
        Dictionary with validation metrics:
        - val_loss: Validation loss
        - val_iou: Average IoU
        - val_precision: Precision
        - val_recall: Recall
    """
    pass
```

### 5.3 Loss Functions

#### Combined Loss Function
```python
def combined_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    localization_weight: float = 1.0,
    classification_weight: float = 1.0,
    confidence_weight: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Compute combined detection loss.
    
    Args:
        predictions: Model predictions dictionary
        targets: Ground truth targets dictionary
        localization_weight: Weight for localization loss
        classification_weight: Weight for classification loss
        confidence_weight: Weight for confidence loss
        
    Returns:
        Dictionary with loss components:
        - total_loss: Weighted sum of all losses
        - localization_loss: Bounding box regression loss
        - classification_loss: Class prediction loss
        - confidence_loss: Objectness loss
    """
    pass
```

#### IoU Loss
```python
def iou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    iou_type: str = "giou"
) -> torch.Tensor:
    """
    Compute IoU-based localization loss.
    
    Args:
        pred_boxes: Predicted boxes [N, 4]
        target_boxes: Target boxes [N, 4]
        iou_type: IoU type ("iou", "giou", "diou", "ciou")
        
    Returns:
        IoU loss tensor
    """
    pass
```

### 5.4 Training Callbacks

#### Callback Interface
```python
class TrainingCallback:
    """Base class for training callbacks."""
    
    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_end(
        self,
        batch_idx: int,
        loss: float
    ) -> None:
        """Called after each batch."""
        pass
```

#### Checkpoint Callback
```python
class CheckpointCallback(TrainingCallback):
    """
    Callback for saving model checkpoints.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        save_best: Whether to save best model
        monitor_metric: Metric to monitor
        monitor_mode: "min" or "max"
    """
    pass
```

---

## 6. Evaluation API

### 6.1 Evaluation Function

#### Main Evaluation Function
```python
def evaluate_model(
    model: DetectionModel,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.25,
    save_predictions: bool = True,
    save_visualizations: bool = True,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, float]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Trained detection model
        test_loader: Test data loader
        device: Evaluation device
        iou_threshold: IoU threshold for matching
        confidence_threshold: Confidence threshold
        save_predictions: Whether to save predictions
        save_visualizations: Whether to save visualizations
        output_dir: Output directory for results
        
    Returns:
        Dictionary with evaluation metrics:
        - mean_iou: Mean Intersection over Union
        - precision: Precision score
        - recall: Recall score
        - f1_score: F1 score
        - mean_ap: Mean Average Precision (if computed)
        - num_true_positives: Number of true positives
        - num_false_positives: Number of false positives
        - num_false_negatives: Number of false negatives
    """
    pass
```

### 6.2 Metric Calculation Functions

#### Calculate IoU
```python
def calculate_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    format: str = "xyxy"
) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) between boxes.
    
    Args:
        box1: First set of boxes [N, 4]
        box2: Second set of boxes [M, 4]
        format: Box format ("xyxy", "xywh", "cxcywh")
        
    Returns:
        IoU matrix [N, M]
    """
    pass
```

#### Calculate Precision and Recall
```python
def calculate_precision_recall(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_threshold: float = 0.5
) -> Tuple[float, float, Dict[str, int]]:
    """
    Calculate precision and recall.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        iou_threshold: IoU threshold for matching
        
    Returns:
        Tuple of (precision, recall, counts) where counts contains:
        - true_positives: Number of true positives
        - false_positives: Number of false positives
        - false_negatives: Number of false negatives
    """
    pass
```

#### Calculate mAP
```python
def calculate_map(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_thresholds: List[float] = None,
    num_classes: int = 1
) -> float:
    """
    Calculate mean Average Precision (mAP).
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        iou_thresholds: List of IoU thresholds (default: 0.5:0.95:0.05)
        num_classes: Number of classes
        
    Returns:
        Mean Average Precision score
    """
    pass
```

### 6.3 Matching Functions

#### Match Predictions with Ground Truth
```python
def match_predictions(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    strategy: str = "best"
) -> List[Dict[str, Any]]:
    """
    Match predictions with ground truth annotations.
    
    Args:
        predictions: List of predictions
        ground_truth: List of ground truth
        iou_threshold: IoU threshold for matching
        strategy: Matching strategy ("best", "greedy")
        
    Returns:
        List of match dictionaries with keys:
        - prediction_idx: Index of matched prediction
        - gt_idx: Index of matched ground truth
        - iou: IoU value
        - is_match: Whether IoU exceeds threshold
    """
    pass
```

---

## 7. Inference API

### 7.1 Single Image Inference

#### Detect Apples in Image
```python
def detect_apples(
    image_path: Union[str, Path],
    model: DetectionModel,
    device: Optional[torch.device] = None,
    confidence_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    input_size: Tuple[int, int] = (640, 640)
) -> Dict[str, Any]:
    """
    Detect apples in a single image.
    
    Args:
        image_path: Path to input image
        model: Trained detection model
        device: Inference device
        confidence_threshold: Confidence threshold
        nms_threshold: NMS threshold
        input_size: Input image size
        
    Returns:
        Dictionary with detection results:
        - boxes: Bounding boxes [N, 4] (absolute coordinates)
        - scores: Confidence scores [N]
        - classes: Class IDs [N]
        - num_detections: Number of detections
    """
    pass
```

#### Batch Inference
```python
def detect_apples_batch(
    image_paths: List[Union[str, Path]],
    model: DetectionModel,
    device: Optional[torch.device] = None,
    batch_size: int = 8,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Detect apples in multiple images.
    
    Args:
        image_paths: List of image paths
        model: Trained detection model
        device: Inference device
        batch_size: Batch size for processing
        **kwargs: Additional arguments for detect_apples
        
    Returns:
        List of detection result dictionaries
    """
    pass
```

### 7.2 Post-Processing Functions

#### Apply NMS
```python
def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.45,
    max_detections: Optional[int] = None
) -> torch.Tensor:
    """
    Apply Non-Maximum Suppression (NMS).
    
    Args:
        boxes: Bounding boxes [N, 4]
        scores: Confidence scores [N]
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections
        
    Returns:
        Indices of boxes to keep after NMS
    """
    pass
```

#### Filter by Confidence
```python
def filter_by_confidence(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    confidence_threshold: float = 0.25
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter detections by confidence threshold.
    
    Args:
        boxes: Bounding boxes [N, 4]
        scores: Confidence scores [N]
        classes: Class IDs [N]
        confidence_threshold: Confidence threshold
        
    Returns:
        Tuple of filtered (boxes, scores, classes)
    """
    pass
```

---

## 8. Utility APIs

### 8.1 IoU Utilities

#### Calculate IoU Matrix
```python
def calculate_iou_matrix(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    format: str = "xyxy"
) -> torch.Tensor:
    """
    Calculate IoU matrix between two sets of boxes.
    
    Args:
        boxes1: First set [N, 4]
        boxes2: Second set [M, 4]
        format: Box format
        
    Returns:
        IoU matrix [N, M]
    """
    pass
```

#### Calculate GIoU
```python
def calculate_giou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor
) -> torch.Tensor:
    """
    Calculate Generalized IoU (GIoU).
    
    Args:
        boxes1: First set of boxes [N, 4]
        boxes2: Second set of boxes [N, 4]
        
    Returns:
        GIoU values [N]
    """
    pass
```

### 8.2 Box Conversion Functions

#### Convert Box Format
```python
def convert_box_format(
    boxes: torch.Tensor,
    from_format: str,
    to_format: str
) -> torch.Tensor:
    """
    Convert bounding boxes between formats.
    
    Args:
        boxes: Input boxes [N, 4]
        from_format: Source format ("xyxy", "xywh", "cxcywh")
        to_format: Target format ("xyxy", "xywh", "cxcywh")
        
    Returns:
        Converted boxes [N, 4]
    """
    pass
```

#### Scale Boxes
```python
def scale_boxes(
    boxes: torch.Tensor,
    scale_factor: Union[float, Tuple[float, float]],
    format: str = "xyxy"
) -> torch.Tensor:
    """
    Scale bounding boxes by factor.
    
    Args:
        boxes: Input boxes [N, 4]
        scale_factor: Scale factor (single value or (w, h))
        format: Box format
        
    Returns:
        Scaled boxes [N, 4]
    """
    pass
```

---

## 9. Configuration API

### 9.1 Configuration Loading

#### Load Configuration
```python
def load_config(
    config_path: Union[str, Path],
    override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        override: Optional dictionary to override config values
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    pass
```

#### Save Configuration
```python
def save_config(
    config: Dict[str, Any],
    config_path: Union[str, Path],
    format: str = "yaml"
) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        format: File format ("yaml" or "json")
    """
    pass
```

#### Validate Configuration
```python
def validate_config(
    config: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    pass
```

---

## 10. Data Transformation API

### 10.1 Preprocessing Transforms

#### Create Preprocessing Pipeline
```python
def create_preprocessing_pipeline(
    input_size: Tuple[int, int] = (640, 640),
    normalize: bool = True,
    normalization_mean: List[float] = None,
    normalization_std: List[float] = None
) -> Callable:
    """
    Create image preprocessing pipeline.
    
    Args:
        input_size: Target image size (width, height)
        normalize: Whether to normalize
        normalization_mean: Normalization mean values
        normalization_std: Normalization std values
        
    Returns:
        Preprocessing function
    """
    pass
```

### 10.2 Augmentation Transforms

#### Create Augmentation Pipeline
```python
def create_augmentation_pipeline(
    horizontal_flip: bool = True,
    rotation: bool = True,
    scale: bool = True,
    brightness: bool = True,
    contrast: bool = True,
    **kwargs
) -> Callable:
    """
    Create data augmentation pipeline.
    
    Args:
        horizontal_flip: Enable horizontal flip
        rotation: Enable rotation
        scale: Enable scaling
        brightness: Enable brightness adjustment
        contrast: Enable contrast adjustment
        **kwargs: Additional augmentation parameters
        
    Returns:
        Augmentation function
    """
    pass
```

---

## 11. Visualization API

### 11.1 Visualization Functions

#### Visualize Detections
```python
def visualize_detections(
    image: Union[np.ndarray, torch.Tensor, PIL.Image.Image],
    boxes: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    classes: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
    ground_truth: Optional[torch.Tensor] = None,
    save_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    Visualize detection results on image.
    
    Args:
        image: Input image
        boxes: Bounding boxes [N, 4]
        scores: Optional confidence scores [N]
        classes: Optional class IDs [N]
        class_names: Optional class names
        ground_truth: Optional ground truth boxes for comparison
        save_path: Optional path to save visualization
        
    Returns:
        Visualization image as numpy array
    """
    pass
```

#### Draw Bounding Box
```python
def draw_bounding_box(
    image: np.ndarray,
    box: Union[torch.Tensor, np.ndarray, List[float]],
    label: Optional[str] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a single bounding box on image.
    
    Args:
        image: Input image
        box: Bounding box [x1, y1, x2, y2] or [x, y, w, h]
        label: Optional label text
        color: Optional box color (B, G, R)
        thickness: Line thickness
        
    Returns:
        Image with drawn box
    """
    pass
```

#### Create Comparison Visualization
```python
def create_comparison_visualization(
    image: np.ndarray,
    predictions: Dict[str, Any],
    ground_truth: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    Create side-by-side comparison of predictions and ground truth.
    
    Args:
        image: Input image
        predictions: Prediction results
        ground_truth: Ground truth annotations
        save_path: Optional path to save
        
    Returns:
        Comparison visualization
    """
    pass
```

---

## 12. Error Handling API

### 12.1 Custom Exceptions

#### Custom Exception Classes
```python
class DatasetError(Exception):
    """Base exception for dataset-related errors."""
    pass

class ModelError(Exception):
    """Base exception for model-related errors."""
    pass

class TrainingError(Exception):
    """Base exception for training-related errors."""
    pass

class InferenceError(Exception):
    """Base exception for inference-related errors."""
    pass
```

### 12.2 Error Handling Utilities

#### Safe File Loading
```python
def safe_load_image(
    image_path: Union[str, Path]
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Safely load image with error handling.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (image, error_message)
        If successful: (image_array, None)
        If failed: (None, error_message)
    """
    pass
```

---

## 13. Type Definitions

### 13.1 Common Type Aliases

```python
from typing import Union, List, Dict, Tuple, Optional, Any
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Path types
PathLike = Union[str, Path]

# Image types
ImageType = Union[np.ndarray, torch.Tensor, Image.Image]

# Box formats
BoxFormat = str  # "xyxy", "xywh", "cxcywh"

# Detection result
DetectionResult = Dict[str, Any]  # Contains boxes, scores, classes

# Annotation format
AnnotationFormat = str  # "yolo", "voc", "coco"

# Model architecture
ModelArchitecture = str  # "yolo", "ssd", "faster_rcnn"
```

### 13.2 Data Structures

#### Detection Result Structure
```python
@dataclass
class Detection:
    """Single detection result."""
    box: List[float]  # [x1, y1, x2, y2] or [x, y, w, h]
    score: float
    class_id: int
    class_name: Optional[str] = None

@dataclass
class DetectionResults:
    """Collection of detection results."""
    detections: List[Detection]
    image_id: Optional[str] = None
    image_size: Optional[Tuple[int, int]] = None
```

---

## 14. Usage Examples

### 14.1 Training Example

```python
from src.model import create_model
from src.dataset import create_dataset, create_dataloader
from src.train import train_model
from src.utils.config import load_config

# Load configuration
config = load_config("configs/config.yaml")

# Create model
model = create_model(
    architecture="yolo",
    size="s",
    num_classes=1,
    pretrained=True
)

# Create datasets
train_dataset = create_dataset("train", config)
val_dataset = create_dataset("val", config)

# Create data loaders
train_loader = create_dataloader(train_dataset, batch_size=16)
val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False)

# Train model
results = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)
```

### 14.2 Inference Example

```python
from src.model import load_model_from_checkpoint
from src.inference import detect_apples
from src.visualization import visualize_detections

# Load model
model, checkpoint_info = load_model_from_checkpoint(
    "checkpoints/best_model.pth"
)

# Detect apples
results = detect_apples(
    image_path="data/test/apple_001.jpg",
    model=model,
    confidence_threshold=0.25
)

# Visualize
image = visualize_detections(
    image="data/test/apple_001.jpg",
    boxes=results["boxes"],
    scores=results["scores"],
    save_path="results/detection.jpg"
)
```

### 14.3 Evaluation Example

```python
from src.model import load_model_from_checkpoint
from src.dataset import create_dataset, create_dataloader
from src.evaluate import evaluate_model

# Load model
model, _ = load_model_from_checkpoint("checkpoints/best_model.pth")

# Create test dataset
test_dataset = create_dataset("test", config)
test_loader = create_dataloader(test_dataset, batch_size=32, shuffle=False)

# Evaluate
metrics = evaluate_model(
    model=model,
    test_loader=test_loader,
    iou_threshold=0.5,
    save_visualizations=True,
    output_dir="results/evaluation"
)

print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"mAP: {metrics['mean_ap']:.3f}")
```

---

## 15. API Versioning

### 15.1 Version Strategy

#### Version Format
- **Major**: Breaking changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

#### Version Information
```python
__version__ = "1.0.0"
API_VERSION = "1.0"
```

---

## 16. References

### 16.1 Related Documents
- [Requirements Specification](Requirements.md)
- [System Architecture](Architecture.md)
- [Configuration Specification](Configuration_Specification.md)

### 16.2 External Resources
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Document End**

*This API specification defines all public interfaces for the Apple Detection System. All implementations should adhere to these specifications.*

