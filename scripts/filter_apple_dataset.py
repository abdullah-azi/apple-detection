"""
Filter Fruit Detection Dataset to Extract Only Apple Images
This script filters a multi-fruit YOLO dataset to extract only apple images.
"""

import os
import shutil
import random
from pathlib import Path
from collections import Counter


def find_dataset_structure(dataset_path):
    """Find the structure of the YOLO dataset."""
    dataset_path = Path(dataset_path)
    
    # Common YOLO dataset structures
    possible_structures = [
        (dataset_path / 'images', dataset_path / 'labels'),
        (dataset_path / 'train' / 'images', dataset_path / 'train' / 'labels'),
        (dataset_path / 'images' / 'train', dataset_path / 'labels' / 'train'),
        (dataset_path / 'train', dataset_path / 'train'),
    ]
    
    for img_path, lbl_path in possible_structures:
        if img_path.exists():
            # Try to find labels in common locations
            label_locations = [
                lbl_path,
                img_path.parent / 'labels',
                dataset_path / 'labels',
                img_path.parent.parent / 'labels',
            ]
            
            for label_path in label_locations:
                if label_path.exists():
                    return img_path, label_path
    
    # If no standard structure found, return what exists
    if (dataset_path / 'images').exists():
        return dataset_path / 'images', dataset_path / 'labels'
    
    return None, None


def analyze_classes(labels_dir, sample_size=100):
    """Analyze class distribution in the dataset."""
    print("\n📊 Analyzing class distribution...")
    
    class_counts = Counter()
    annotation_files = list(labels_dir.glob('*.txt'))
    
    if not annotation_files:
        # Try subdirectories
        annotation_files = list(labels_dir.rglob('*.txt'))
    
    sample_files = annotation_files[:min(sample_size, len(annotation_files))]
    
    for ann_file in sample_files:
        try:
            with open(ann_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_counts[parts[0]] += 1
        except Exception as e:
            continue
    
    print("Class distribution (from sample):")
    fruit_names = {
        '0': 'Apple', 
        '1': 'Grapes', 
        '2': 'Pineapple',
        '3': 'Orange', 
        '4': 'Banana', 
        '5': 'Watermelon'
    }
    
    for class_id in sorted(class_counts.keys()):
        fruit_name = fruit_names.get(class_id, f'Class {class_id}')
        print(f"  Class {class_id} ({fruit_name}): {class_counts[class_id]} boxes")
    
    return '0'  # Apple is typically class 0


def filter_apple_annotations(input_ann_path, output_ann_path, apple_class_id='0'):
    """
    Filter annotation file to keep only apple bounding boxes.
    Returns True if file contains at least one apple, False otherwise.
    """
    apple_boxes = []
    try:
        with open(input_ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0] == apple_class_id:  # Only apple boxes
                    apple_boxes.append(line.strip())
        
        if apple_boxes:
            # Write filtered annotation (only apple boxes)
            with open(output_ann_path, 'w') as f:
                for box in apple_boxes:
                    f.write(box + '\n')
            return True
    except Exception as e:
        print(f"Error processing {input_ann_path}: {e}")
    
    return False


def find_corresponding_annotation(image_path, labels_dir):
    """Find the corresponding annotation file for an image."""
    possible_paths = [
        labels_dir / (image_path.stem + '.txt'),
        labels_dir / image_path.name.replace(image_path.suffix, '.txt'),
        image_path.parent.parent / 'labels' / (image_path.stem + '.txt'),
        image_path.parent / (image_path.stem + '.txt'),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def filter_and_prepare_dataset(dataset_path, output_dir, apple_class_id='0', 
                                train_split=0.7, val_split=0.15, test_split=0.15, 
                                seed=42):
    """
    Filter dataset to extract only apple images and prepare for training.
    
    Args:
        dataset_path: Path to the downloaded fruit detection dataset
        output_dir: Path where filtered dataset will be saved
        apple_class_id: Class ID for apples (usually '0')
        train_split: Proportion for training set
        val_split: Proportion for validation set
        test_split: Proportion for test set
        seed: Random seed for reproducibility
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    
    # Create output directory structure
    print("📁 Creating output directory structure...")
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'annotations' / split).mkdir(parents=True, exist_ok=True)
    
    # Find dataset structure
    print("\n🔍 Finding dataset structure...")
    images_dir, labels_dir = find_dataset_structure(dataset_path)
    
    if not images_dir or not images_dir.exists():
        print(f"❌ Error: Could not find images directory in {dataset_path}")
        print("   Please check the dataset path and structure.")
        return False
    
    if not labels_dir or not labels_dir.exists():
        print(f"⚠️  Warning: Could not find labels directory.")
        print(f"   Looking for labels in: {labels_dir}")
        print("   Trying to find labels in subdirectories...")
        # Try to find labels anywhere
        labels_dir = dataset_path
        label_files = list(dataset_path.rglob('*.txt'))
        if label_files:
            labels_dir = label_files[0].parent
            print(f"   Found labels in: {labels_dir}")
        else:
            print("❌ Error: No annotation files found!")
            return False
    
    print(f"✅ Found images in: {images_dir}")
    print(f"✅ Found labels in: {labels_dir}")
    
    # Analyze classes
    apple_class_id = analyze_classes(labels_dir)
    print(f"\n✅ Using class_id '{apple_class_id}' for Apple")
    
    # Find all images
    print("\n🔍 Finding all images...")
    all_images = (
        list(images_dir.glob('*.jpg')) + 
        list(images_dir.glob('*.png')) + 
        list(images_dir.glob('*.JPG')) + 
        list(images_dir.glob('*.PNG')) +
        list(images_dir.rglob('*.jpg')) +
        list(images_dir.rglob('*.png'))
    )
    
    # Remove duplicates
    all_images = list(set(all_images))
    print(f"Found {len(all_images)} total images")
    
    # Filter images that have apple annotations
    print("\n🍎 Filtering for apple images...")
    apple_images = []
    
    for img_path in all_images:
        ann_path = find_corresponding_annotation(img_path, labels_dir)
        
        if ann_path and ann_path.exists():
            # Check if this image has apples
            temp_ann = output_dir / 'temp_check.txt'
            if filter_apple_annotations(ann_path, temp_ann, apple_class_id):
                apple_images.append((img_path, ann_path))
            if temp_ann.exists():
                temp_ann.unlink()  # Clean up
    
    print(f"✅ Found {len(apple_images)} images containing apples")
    
    if len(apple_images) == 0:
        print("❌ Error: No apple images found!")
        print("   Please check:")
        print("   1. The dataset contains apple images")
        print("   2. The apple class ID is correct (try '0' or '1')")
        print("   3. The annotation files are in the correct format")
        return False
    
    # Split dataset
    print("\n📦 Splitting dataset...")
    random.seed(seed)
    random.shuffle(apple_images)
    
    train_count = int(train_split * len(apple_images))
    val_count = int(val_split * len(apple_images))
    
    train_data = apple_images[:train_count]
    val_data = apple_images[train_count:train_count + val_count]
    test_data = apple_images[train_count + val_count:]
    
    print(f"  Train: {len(train_data)} images ({len(train_data)/len(apple_images)*100:.1f}%)")
    print(f"  Val:   {len(val_data)} images ({len(val_data)/len(apple_images)*100:.1f}%)")
    print(f"  Test:  {len(test_data)} images ({len(test_data)/len(apple_images)*100:.1f}%)")
    
    # Copy filtered data
    print("\n📋 Copying filtered data...")
    
    def copy_split(data_list, split_name):
        """Copy images and filtered annotations for a split"""
        copied = 0
        for img_path, ann_path in data_list:
            # Copy image
            dest_img = output_dir / 'images' / split_name / img_path.name
            shutil.copy(img_path, dest_img)
            
            # Copy and filter annotation (only apple boxes)
            dest_ann = output_dir / 'annotations' / split_name / (img_path.stem + '.txt')
            filter_apple_annotations(ann_path, dest_ann, apple_class_id)
            copied += 1
        
        print(f"  ✅ {split_name}: {copied} apple images copied")
        return copied
    
    copy_split(train_data, 'train')
    copy_split(val_data, 'val')
    copy_split(test_data, 'test')
    
    # Verify final dataset
    print("\n✅ Final Dataset Verification:")
    total_images = 0
    total_boxes = 0
    
    for split in ['train', 'val', 'test']:
        img_dir = output_dir / 'images' / split
        ann_dir = output_dir / 'annotations' / split
        
        img_count = len(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
        ann_count = len(list(ann_dir.glob('*.txt')))
        
        # Count total apple boxes
        boxes = 0
        for ann_file in ann_dir.glob('*.txt'):
            with open(ann_file, 'r') as f:
                boxes += len([l for l in f if l.strip()])
        
        total_images += img_count
        total_boxes += boxes
        
        print(f"\n{split.upper()}:")
        print(f"  Images: {img_count}")
        print(f"  Annotations: {ann_count}")
        print(f"  Total apple bounding boxes: {boxes}")
        if img_count > 0:
            print(f"  Avg boxes per image: {boxes/img_count:.2f}")
    
    print(f"\n📊 Summary:")
    print(f"  Total images: {total_images}")
    print(f"  Total apple boxes: {total_boxes}")
    print(f"  Average boxes per image: {total_boxes/total_images:.2f}" if total_images > 0 else "")
    
    print("\n🎉 Dataset ready! Only apple images with apple bounding boxes.")
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python filter_apple_dataset.py <dataset_path> <output_path>")
        print("Example: python filter_apple_dataset.py /content/dataset /content/apple-detection/data")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    output_path = sys.argv[2]
    
    success = filter_and_prepare_dataset(dataset_path, output_path)
    sys.exit(0 if success else 1)

