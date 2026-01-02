# Data Specification: Apple Detection Dataset

## 📋 Document Information

- **Project**: Apple Detection Using Object Detection
- **Version**: 1.0
- **Date**: January 2026
- **Status**: Draft
- **Author**: Project Team

---

## 📑 Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset Overview](#2-dataset-overview)
3. [Data Structure](#3-data-structure)
4. [Image Specifications](#4-image-specifications)
5. [Annotation Formats](#5-annotation-formats)
6. [Annotation Guidelines](#6-annotation-guidelines)
7. [Data Collection Guidelines](#7-data-collection-guidelines)
8. [Dataset Organization](#8-dataset-organization)
9. [Data Validation](#9-data-validation)
10. [Data Preprocessing](#10-data-preprocessing)
11. [Dataset Splits](#11-dataset-splits)
12. [Data Quality Standards](#12-data-quality-standards)
13. [Data Augmentation](#13-data-augmentation)
14. [Data Statistics](#14-data-statistics)
15. [Data Maintenance](#15-data-maintenance)

---

## 1. Introduction

### 1.1 Purpose
This document specifies the data requirements, formats, and standards for the Apple Detection dataset. It defines how images and annotations should be structured, collected, and validated to ensure consistent and high-quality training data.

### 1.2 Scope
This specification covers:
- Image format and quality requirements
- Annotation formats (YOLO, Pascal VOC, COCO)
- Data organization and structure
- Annotation guidelines and best practices
- Data validation procedures
- Preprocessing requirements

### 1.3 Target Audience
- Data collectors and annotators
- Dataset curators
- Developers implementing data loaders
- Quality assurance personnel

---

## 2. Dataset Overview

### 2.1 Dataset Purpose
The dataset is designed to train and evaluate an object detection model capable of identifying and localizing apples in digital images.

### 2.2 Dataset Characteristics

#### Object Class
- **Primary Class**: Apple
- **Class ID**: 0 (for single-class detection)
- **Class Name**: "apple"

#### Detection Task
- **Type**: Single-class object detection
- **Output**: Bounding boxes with confidence scores
- **Multiple Instances**: Yes (multiple apples per image allowed)

### 2.3 Dataset Size Requirements

#### Minimum Dataset
- **Training Images**: 100+ images
- **Validation Images**: 20+ images
- **Test Images**: 20+ images
- **Total**: 140+ images
- **Total Annotations**: 200+ bounding boxes (minimum)

#### Recommended Dataset
- **Training Images**: 500+ images
- **Validation Images**: 100+ images
- **Test Images**: 100+ images
- **Total**: 700+ images
- **Total Annotations**: 1000+ bounding boxes

#### Ideal Dataset
- **Training Images**: 1000+ images
- **Validation Images**: 200+ images
- **Test Images**: 200+ images
- **Total**: 1400+ images
- **Total Annotations**: 2000+ bounding boxes

### 2.4 Data Diversity Requirements

#### Required Diversity
- ✅ Different lighting conditions (bright, dim, natural, artificial)
- ✅ Various backgrounds (trees, baskets, markets, plain backgrounds)
- ✅ Different apple varieties (red, green, yellow, mixed)
- ✅ Multiple scales (close-up, medium, far away)
- ✅ Different orientations (various angles)
- ✅ Multiple apples per image (1-10+ apples)

#### Recommended Diversity
- ✅ Different image qualities (high-res, medium-res)
- ✅ Various camera angles (front, side, top, bottom)
- ✅ Occlusion cases (partially hidden apples)
- ✅ Different seasons/environments
- ✅ Various image sources (photos, stock images)

---

## 3. Data Structure

### 3.1 Directory Structure

#### Standard Structure
```
data/
├── images/
│   ├── train/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   ├── val/
│   │   ├── image_101.jpg
│   │   ├── image_102.jpg
│   │   └── ...
│   └── test/
│       ├── image_201.jpg
│       ├── image_202.jpg
│       └── ...
├── annotations/
│   ├── train/
│   │   ├── image_001.txt (YOLO) or image_001.xml (VOC)
│   │   ├── image_002.txt
│   │   └── ...
│   ├── val/
│   │   ├── image_101.txt
│   │   ├── image_102.txt
│   │   └── ...
│   └── test/
│       ├── image_201.txt
│       ├── image_202.txt
│       └── ...
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

#### Alternative Structure (COCO Format)
```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/
│   ├── train_annotations.json (COCO format)
│   ├── val_annotations.json
│   └── test_annotations.json
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### 3.2 File Naming Conventions

#### Image Files
- **Format**: `{prefix}_{number}.{extension}`
- **Examples**: 
  - `apple_001.jpg`
  - `image_042.png`
  - `train_123.jpg`
- **Rules**:
  - Use consistent naming pattern
  - Avoid special characters (spaces, symbols)
  - Use lowercase letters and numbers
  - Include leading zeros for sorting (001, 002, ...)

#### Annotation Files
- **YOLO**: Same name as image, `.txt` extension
  - `apple_001.jpg` → `apple_001.txt`
- **VOC**: Same name as image, `.xml` extension
  - `apple_001.jpg` → `apple_001.xml`
- **COCO**: Single JSON file per split
  - `train_annotations.json`

### 3.3 Split Files

#### Split File Format
Each split file (train.txt, val.txt, test.txt) contains one image filename per line:

```
apple_001.jpg
apple_002.jpg
apple_003.jpg
...
```

#### Split File Location
- Stored in `data/splits/` directory
- Used to organize dataset splits
- Can be generated programmatically or manually

---

## 4. Image Specifications

### 4.1 Image Format Requirements

#### Supported Formats
- **JPEG** (.jpg, .jpeg) - Recommended for photos
- **PNG** (.png) - Recommended for screenshots, graphics
- **BMP** (.bmp) - Supported but not recommended

#### Format Recommendations
- **JPEG**: Best for photographs, smaller file size
- **PNG**: Best for lossless quality, larger file size
- **Avoid**: GIF, TIFF (unless necessary)

### 4.2 Image Quality Requirements

#### Resolution
- **Minimum**: 224x224 pixels
- **Recommended**: 640x640 pixels or higher
- **Maximum**: 4K (3840x2160) - will be resized during preprocessing
- **Aspect Ratio**: Any (will be handled during preprocessing)

#### Image Quality
- **Clarity**: Clear, not blurry
- **Focus**: Well-focused on apples
- **Exposure**: Reasonable exposure (not over/under-exposed)
- **Noise**: Minimal digital noise
- **Compression**: Reasonable quality (JPEG quality > 80)

### 4.3 Image Content Requirements

#### Required Content
- ✅ At least one visible apple in the image
- ✅ Apples should be clearly identifiable
- ✅ Reasonable image composition

#### Image Characteristics
- **Color Space**: RGB (3 channels)
- **Bit Depth**: 8-bit per channel (24-bit color)
- **Orientation**: Any (rotation handled during preprocessing)

### 4.4 Image Validation Rules

#### Valid Image Criteria
- ✅ File can be opened and decoded
- ✅ Has 3 color channels (RGB)
- ✅ Resolution meets minimum requirements
- ✅ Not corrupted or damaged
- ✅ Contains at least one apple (visually verified)

#### Invalid Image Criteria
- ❌ Corrupted file
- ❌ Unsupported format
- ❌ Grayscale only (unless intentional)
- ❌ Resolution too low (< 224x224)
- ❌ No apples visible

---

## 5. Annotation Formats

### 5.1 YOLO Format

#### File Structure
- **Extension**: `.txt`
- **One file per image**
- **Same filename as image** (different extension)

#### Format Specification
```
class_id center_x center_y width height
class_id center_x center_y width height
...
```

#### Coordinate System
- **Normalized coordinates** (0.0 to 1.0)
- **center_x**: X-coordinate of box center (normalized)
- **center_y**: Y-coordinate of box center (normalized)
- **width**: Box width (normalized)
- **height**: Box height (normalized)

#### Example
```
0 0.5 0.5 0.3 0.4
0 0.2 0.7 0.15 0.2
```

This represents:
- First apple: center at (50%, 50%), size 30% width × 40% height
- Second apple: center at (20%, 70%), size 15% width × 20% height

#### YOLO Format Rules
- One line per object
- Class ID is 0 (for single-class detection)
- All coordinates normalized to [0, 1]
- Empty file if no objects in image

### 5.2 Pascal VOC Format

#### File Structure
- **Extension**: `.xml`
- **One file per image**
- **XML structure**

#### Format Specification
```xml
<annotation>
    <folder>images</folder>
    <filename>apple_001.jpg</filename>
    <path>path/to/apple_001.jpg</path>
    <source>
        <database>Apple Detection Dataset</database>
    </source>
    <size>
        <width>640</width>
        <height>480</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>apple</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>100</xmin>
            <ymin>150</ymin>
            <xmax>300</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
    <object>
        <!-- Additional objects -->
    </object>
</annotation>
```

#### Coordinate System
- **Absolute pixel coordinates**
- **xmin, ymin**: Top-left corner
- **xmax, ymax**: Bottom-right corner
- **Origin**: Top-left corner (0, 0)

#### VOC Format Rules
- Multiple `<object>` tags for multiple apples
- Coordinates must be within image bounds
- xmin < xmax, ymin < ymax
- All coordinates are integers (pixels)

### 5.3 COCO Format

#### File Structure
- **Extension**: `.json`
- **Single file per split** (train, val, test)
- **JSON structure**

#### Format Specification
```json
{
    "info": {
        "description": "Apple Detection Dataset",
        "version": "1.0",
        "year": 2026
    },
    "licenses": [...],
    "images": [
        {
            "id": 1,
            "file_name": "apple_001.jpg",
            "width": 640,
            "height": 480
        },
        ...
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 150, 200, 250],
            "area": 50000,
            "iscrowd": 0
        },
        ...
    ],
    "categories": [
        {
            "id": 1,
            "name": "apple",
            "supercategory": "fruit"
        }
    ]
}
```

#### Coordinate System
- **Bounding box**: [x, y, width, height]
- **x, y**: Top-left corner (absolute pixels)
- **width, height**: Box dimensions (absolute pixels)
- **area**: Box area in pixels²

#### COCO Format Rules
- Bounding box format: [x, y, width, height]
- All coordinates are absolute pixels
- Each annotation linked to image via `image_id`
- Category ID: 1 (for apple class)

### 5.4 Format Conversion

#### Conversion Requirements
- System MUST support conversion between formats
- Conversion MUST preserve accuracy
- Coordinate transformations MUST be correct

#### Conversion Examples

**YOLO to VOC:**
```
Normalized → Absolute pixels
center_x, center_y, width, height → xmin, ymin, xmax, ymax
```

**VOC to YOLO:**
```
Absolute pixels → Normalized
xmin, ymin, xmax, ymax → center_x, center_y, width, height
```

**COCO to YOLO:**
```
[x, y, width, height] → [center_x, center_y, width, height]
```

---

## 6. Annotation Guidelines

### 6.1 Bounding Box Placement

#### General Rules
- **Tight Fit**: Box should tightly enclose the apple
- **Complete Coverage**: Include entire visible apple
- **Minimal Background**: Minimize background pixels in box
- **Consistent Style**: Use same annotation style throughout

#### Box Boundaries
- **Top Edge**: Just above the topmost visible part of apple
- **Bottom Edge**: Just below the bottommost visible part
- **Left Edge**: Just left of the leftmost visible part
- **Right Edge**: Just right of the rightmost visible part

### 6.2 Edge Cases

#### Partially Visible Apples
- **> 50% Visible**: Annotate if more than half is visible
- **< 50% Visible**: Consider omitting or marking as "difficult"
- **Occluded**: Include if clearly identifiable as apple

#### Overlapping Apples
- **Separate Boxes**: Each apple gets its own bounding box
- **Overlap Allowed**: Boxes can overlap if apples overlap
- **Clear Separation**: Ensure boxes don't merge into one

#### Multiple Scales
- **Small Apples**: Still annotate if clearly identifiable
- **Large Apples**: Ensure box includes entire apple
- **Mixed Scales**: Handle each apple independently

### 6.3 Annotation Quality Standards

#### Accuracy Requirements
- **Pixel Accuracy**: Box edges within 2-3 pixels of apple edge
- **Consistency**: Similar apples annotated similarly
- **Completeness**: All visible apples annotated

#### Common Mistakes to Avoid
- ❌ Boxes too large (too much background)
- ❌ Boxes too small (cutting off apple)
- ❌ Missing apples (incomplete annotation)
- ❌ Wrong objects (annotating non-apples)
- ❌ Inconsistent box placement

### 6.4 Annotation Tools

#### Recommended Tools
- **LabelImg**: Popular YOLO/VOC annotation tool
- **CVAT**: Web-based annotation platform
- **Labelbox**: Commercial annotation platform
- **VGG Image Annotator (VIA)**: Web-based, free

#### Tool Requirements
- Support bounding box annotation
- Export to YOLO/VOC/COCO format
- Quality control features
- Batch processing capabilities

---

## 7. Data Collection Guidelines

### 7.1 Image Collection

#### Sources
- **Personal Photos**: Photos taken specifically for project
- **Stock Images**: Free stock photo websites
- **Public Datasets**: Existing fruit/vegetable datasets
- **Web Scraping**: With proper permissions and licenses

#### Collection Criteria
- **Diversity**: Collect diverse images (see Section 2.4)
- **Quality**: High-quality, clear images
- **Relevance**: Images must contain apples
- **Legal**: Ensure proper licensing and permissions

### 7.2 Image Selection

#### Selection Criteria
- ✅ Clear, well-lit images
- ✅ Apples are main or significant subject
- ✅ Reasonable resolution
- ✅ Diverse scenarios
- ✅ Various apple types

#### Rejection Criteria
- ❌ Blurry or low-quality images
- ❌ No apples visible
- ❌ Extremely small apples (< 20 pixels)
- ❌ Copyright/licensing issues
- ❌ Duplicate or near-duplicate images

### 7.3 Data Collection Workflow

#### Step-by-Step Process
1. **Identify Source**: Determine image source
2. **Download/Collect**: Obtain images
3. **Validate**: Check image quality and content
4. **Organize**: Place in appropriate directory
5. **Annotate**: Create bounding box annotations
6. **Validate Annotations**: Check annotation quality
7. **Add to Dataset**: Include in dataset splits

---

## 8. Dataset Organization

### 8.1 Directory Organization

#### Standard Organization
```
data/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
├── annotations/
│   ├── train/          # Training annotations
│   ├── val/            # Validation annotations
│   └── test/           # Test annotations
└── splits/
    ├── train.txt       # Training image list
    ├── val.txt         # Validation image list
    └── test.txt        # Test image list
```

### 8.2 File Matching

#### Image-Annotation Matching
- **YOLO/VOC**: Annotation file must have same name as image
  - `apple_001.jpg` ↔ `apple_001.txt` or `apple_001.xml`
- **COCO**: Annotations linked via `image_id` and `file_name`
- **Validation**: System must verify all images have annotations

### 8.3 Dataset Metadata

#### Required Metadata
- **Dataset Name**: Apple Detection Dataset
- **Version**: 1.0
- **Creation Date**: [Date]
- **Total Images**: [Count]
- **Total Annotations**: [Count]
- **Format**: YOLO/VOC/COCO

#### Optional Metadata
- **License**: Dataset license information
- **Contributors**: List of contributors
- **Description**: Dataset description
- **Statistics**: Dataset statistics (see Section 14)

---

## 9. Data Validation

### 9.1 Image Validation

#### Validation Checks
1. **File Integrity**: File can be opened and decoded
2. **Format**: Supported image format
3. **Resolution**: Meets minimum resolution requirements
4. **Channels**: Has 3 color channels (RGB)
5. **Content**: Contains at least one apple (visual check)

#### Validation Script Requirements
- Check all images in dataset
- Report invalid images
- Generate validation report
- Handle errors gracefully

### 9.2 Annotation Validation

#### Validation Checks
1. **File Exists**: Annotation file exists for each image
2. **Format Valid**: Annotation format is correct
3. **Coordinates Valid**: Coordinates are within image bounds
4. **Box Dimensions**: Width and height are positive
5. **Box Relationships**: xmin < xmax, ymin < ymax
6. **Class Labels**: Class IDs are valid (0 for apple)

#### Coordinate Validation Rules

**YOLO Format:**
- All values in [0, 1] range
- center_x ± width/2 in [0, 1]
- center_y ± height/2 in [0, 1]

**VOC Format:**
- All coordinates are integers
- xmin, ymin, xmax, ymax within image bounds
- xmin < xmax, ymin < ymax

**COCO Format:**
- Bounding box: [x, y, width, height]
- x, y, width, height are positive
- x + width ≤ image_width
- y + height ≤ image_height

### 9.3 Dataset Consistency Validation

#### Consistency Checks
1. **Image-Annotation Match**: All images have annotations
2. **Split Consistency**: Images in splits exist
3. **No Duplicates**: No duplicate images across splits
4. **Complete Coverage**: All images in splits are accounted for
5. **Format Consistency**: All annotations use same format

### 9.4 Validation Tools

#### Recommended Tools
- **Custom Python Scripts**: Dataset-specific validation
- **LabelImg Validation**: Built-in validation features
- **COCO API**: Validation tools for COCO format
- **OpenCV**: Image validation utilities

---

## 10. Data Preprocessing

### 10.1 Image Preprocessing

#### Required Preprocessing
1. **Resize**: Resize to fixed resolution (e.g., 640x640)
2. **Normalization**: Normalize pixel values to [0, 1] or standardize
3. **Color Space**: Ensure RGB color space
4. **Data Type**: Convert to float32 for model input

#### Preprocessing Pipeline
```python
# Example preprocessing steps
1. Load image (RGB)
2. Resize to target size (maintain aspect ratio or pad)
3. Normalize: pixel_values / 255.0
4. Convert to tensor format
```

### 10.2 Annotation Preprocessing

#### Required Preprocessing
1. **Format Conversion**: Convert to model's required format
2. **Coordinate Transformation**: Transform to model's coordinate system
3. **Validation**: Validate coordinates after transformation
4. **Normalization**: Normalize if required by model

#### Coordinate Transformation
- **YOLO → Model**: May need to convert normalized to absolute
- **VOC → Model**: May need to convert absolute to normalized
- **COCO → Model**: May need format conversion

### 10.3 Preprocessing Configuration

#### Configurable Parameters
- **Target Resolution**: Image resize dimensions
- **Normalization Method**: [0,1] or ImageNet stats
- **Aspect Ratio Handling**: Stretch, pad, or crop
- **Color Augmentation**: Brightness, contrast, etc.

---

## 11. Dataset Splits

### 11.1 Split Strategy

#### Standard Split
- **Training**: 70% of images
- **Validation**: 15% of images
- **Test**: 15% of images

#### Alternative Split (Small Dataset)
- **Training**: 80% of images
- **Validation**: 10% of images
- **Test**: 10% of images

### 11.2 Split Requirements

#### Split Criteria
- **Random Split**: Random assignment to splits
- **Stratified Split**: Ensure diversity in each split
- **No Overlap**: Images appear in only one split
- **Balanced**: Similar distribution across splits

#### Split Validation
- ✅ No duplicate images across splits
- ✅ All images assigned to a split
- ✅ Reasonable distribution (not too imbalanced)
- ✅ Diversity maintained in each split

### 11.3 Split File Format

#### Format Specification
Each split file contains image filenames (one per line):
```
apple_001.jpg
apple_002.jpg
apple_003.jpg
...
```

#### Split File Generation
- Can be generated randomly
- Can be manually curated
- Should be reproducible (use random seed)

---

## 12. Data Quality Standards

### 12.1 Image Quality Standards

#### Minimum Quality
- **Resolution**: ≥ 224x224 pixels
- **Clarity**: Clear, not blurry
- **Exposure**: Reasonable exposure
- **Focus**: Well-focused on subject

#### High Quality
- **Resolution**: ≥ 640x640 pixels
- **Clarity**: Very clear, sharp
- **Exposure**: Well-exposed
- **Composition**: Good composition

### 12.2 Annotation Quality Standards

#### Minimum Quality
- **Accuracy**: Boxes within 5 pixels of apple edge
- **Completeness**: All clearly visible apples annotated
- **Consistency**: Reasonably consistent annotation style

#### High Quality
- **Accuracy**: Boxes within 2-3 pixels of apple edge
- **Completeness**: All apples annotated (including partial)
- **Consistency**: Very consistent annotation style
- **Validation**: All annotations validated

### 12.3 Dataset Quality Metrics

#### Quality Indicators
- **Annotation Coverage**: % of images with annotations
- **Average Objects per Image**: Mean number of apples per image
- **Box Accuracy**: Average IoU with ground truth (if available)
- **Consistency Score**: Measure of annotation consistency

---

## 13. Data Augmentation

### 13.1 Augmentation Strategy

#### Purpose
- Increase dataset diversity
- Improve model generalization
- Reduce overfitting
- Handle various conditions

### 13.2 Augmentation Techniques

#### Geometric Augmentations
- **Horizontal Flip**: Mirror image horizontally
- **Rotation**: Small angle rotations (±15°)
- **Scaling**: Random zoom in/out (0.8x - 1.2x)
- **Translation**: Small shifts

#### Color Augmentations
- **Brightness**: Adjust brightness (±20%)
- **Contrast**: Adjust contrast (±20%)
- **Saturation**: Adjust color saturation
- **Hue**: Slight hue shifts

#### Advanced Augmentations (Optional)
- **Cutout**: Random rectangular cutouts
- **Mixup**: Blend two images
- **Mosaic**: Combine multiple images

### 13.3 Augmentation Considerations

#### Important Notes
- **Coordinate Transformation**: Must update bounding boxes
- **Validation**: Ensure boxes remain valid after augmentation
- **Realism**: Augmentations should be realistic
- **Balance**: Don't over-augment (can hurt performance)

#### Augmentation Pipeline
```python
# Example augmentation order
1. Geometric transformations (flip, rotate, scale)
2. Update bounding box coordinates
3. Color augmentations (brightness, contrast)
4. Final validation of coordinates
```

---

## 14. Data Statistics

### 14.1 Dataset Statistics

#### Recommended Statistics to Track
- **Total Images**: Count of images in dataset
- **Total Annotations**: Count of bounding boxes
- **Images per Split**: Count for train/val/test
- **Average Objects per Image**: Mean apples per image
- **Object Distribution**: Histogram of objects per image

#### Image Statistics
- **Resolution Distribution**: Distribution of image sizes
- **Format Distribution**: JPEG vs PNG vs others
- **Aspect Ratio Distribution**: Distribution of aspect ratios

#### Annotation Statistics
- **Box Size Distribution**: Distribution of box sizes
- **Box Position Distribution**: Where boxes appear in images
- **Box Aspect Ratio**: Distribution of box aspect ratios

### 14.2 Statistics Generation

#### Tools
- **Custom Scripts**: Python scripts to calculate statistics
- **Pandas**: Data analysis and statistics
- **Matplotlib**: Visualization of statistics

#### Statistics Report
Should include:
- Summary statistics (counts, means, etc.)
- Distribution plots
- Quality metrics
- Recommendations

---

## 15. Data Maintenance

### 15.1 Version Control

#### Dataset Versioning
- **Version Numbers**: Track dataset versions (v1.0, v1.1, etc.)
- **Changelog**: Document changes between versions
- **Backup**: Keep backups of previous versions

### 15.2 Dataset Updates

#### Update Process
1. **Review Changes**: Identify what needs updating
2. **Validate Changes**: Ensure quality maintained
3. **Update Metadata**: Update version and changelog
4. **Regenerate Splits**: If images added/removed
5. **Re-validate**: Run validation checks

### 15.3 Dataset Documentation

#### Required Documentation
- **README**: Dataset overview and usage
- **Statistics Report**: Dataset statistics
- **Annotation Guidelines**: How to annotate
- **Changelog**: Version history

---

## 16. Data Access and Usage

### 16.1 Data Access

#### Access Methods
- **Local Storage**: Files stored locally
- **Cloud Storage**: Optional cloud backup
- **Version Control**: Git LFS for large files (if applicable)

### 16.2 Data Usage

#### Usage Guidelines
- **Training**: Use training split for model training
- **Validation**: Use validation split for hyperparameter tuning
- **Test**: Use test split only for final evaluation
- **No Leakage**: Never use test set during training

### 16.3 Data Sharing

#### Sharing Considerations
- **License**: Ensure proper licensing
- **Privacy**: Remove any sensitive information
- **Format**: Provide in standard formats
- **Documentation**: Include usage documentation

---

## 17. Troubleshooting

### 17.1 Common Issues

#### Image Issues
- **Problem**: Corrupted images
- **Solution**: Re-download or re-collect images

#### Annotation Issues
- **Problem**: Missing annotations
- **Solution**: Create missing annotation files

#### Format Issues
- **Problem**: Wrong annotation format
- **Solution**: Convert to correct format

#### Coordinate Issues
- **Problem**: Coordinates out of bounds
- **Solution**: Validate and fix coordinates

### 17.2 Validation Errors

#### Common Validation Errors
- Missing annotation files
- Invalid coordinate ranges
- Format mismatches
- Inconsistent file naming

#### Error Resolution
1. Identify error type
2. Locate problematic files
3. Fix issues
4. Re-run validation
5. Verify fixes

---

## 18. References

### 18.1 Related Documents
- [Requirements Specification](Requirements.md)
- [Project Overview](Project_Overview.md)
- [Configuration Specification](Configuration_Spec.md) (to be created)

### 18.2 External Resources
- [YOLO Format Documentation](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)
- [Pascal VOC Format](http://host.robots.ox.ac.uk/pascal/VOC/)
- [COCO Format Documentation](https://cocodataset.org/#format-data)
- [LabelImg Tool](https://github.com/tzutalin/labelImg)

---

## 19. Appendix

### 19.1 Annotation Examples

#### Example 1: Single Apple (YOLO)
**Image**: `apple_001.jpg` (640x480 pixels)
**Annotation**: `apple_001.txt`
```
0 0.5 0.5 0.3 0.4
```

#### Example 2: Multiple Apples (YOLO)
**Image**: `apple_002.jpg` (640x480 pixels)
**Annotation**: `apple_002.txt`
```
0 0.3 0.4 0.2 0.25
0 0.7 0.6 0.18 0.22
0 0.5 0.8 0.15 0.2
```

#### Example 3: Single Apple (VOC)
**Image**: `apple_001.jpg` (640x480 pixels)
**Annotation**: `apple_001.xml`
```xml
<annotation>
    <filename>apple_001.jpg</filename>
    <size>
        <width>640</width>
        <height>480</height>
    </size>
    <object>
        <name>apple</name>
        <bndbox>
            <xmin>224</xmin>
            <ymin>144</ymin>
            <xmax>416</xmax>
            <ymax>336</ymax>
        </bndbox>
    </object>
</annotation>
```

### 19.2 Coordinate Conversion Examples

#### YOLO to VOC Conversion
**YOLO**: `0 0.5 0.5 0.3 0.4` (Image: 640x480)
**VOC**: 
- center_x = 0.5 × 640 = 320
- center_y = 0.5 × 480 = 240
- width = 0.3 × 640 = 192
- height = 0.4 × 480 = 192
- xmin = 320 - 96 = 224
- ymin = 240 - 96 = 144
- xmax = 320 + 96 = 416
- ymax = 240 + 96 = 336

### 19.3 Validation Checklist

#### Pre-Training Checklist
- [ ] All images validated
- [ ] All annotations validated
- [ ] Format consistency verified
- [ ] Split files generated
- [ ] Statistics calculated
- [ ] Documentation complete

---

**Document End**

*This data specification serves as the definitive guide for dataset creation, organization, and validation for the Apple Detection project.*

