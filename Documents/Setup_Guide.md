# Setup Guide: Apple Detection Project

## 📋 Document Information

- **Project**: Apple Detection Using Object Detection
- **Version**: 1.0
- **Date**: January 2026
- **Purpose**: Step-by-step guide to set up the development environment

---

## 📑 Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [System Requirements](#2-system-requirements)
3. [Step 1: Clone/Download Project](#3-step-1-clonedownload-project)
4. [Step 2: Python Environment Setup](#4-step-2-python-environment-setup)
5. [Step 3: Install Dependencies](#5-step-3-install-dependencies)
6. [Step 4: Verify Installation](#6-step-4-verify-installation)
7. [Step 5: Configure Project](#7-step-5-configure-project)
8. [Step 6: Prepare Dataset](#8-step-6-prepare-dataset)
9. [Step 7: Test Setup](#9-step-7-test-setup)
10. [Troubleshooting](#10-troubleshooting)
11. [Quick Start Checklist](#11-quick-start-checklist)

---

## 1. Prerequisites

### 1.1 Required Knowledge
- Basic Python programming
- Familiarity with command line/terminal
- Basic understanding of machine learning concepts (helpful but not required)

### 1.2 Required Software
- **Python 3.8 or higher** (3.9+ recommended)
- **Git** (for version control, optional)
- **Text Editor/IDE** (VS Code, PyCharm, or any editor)

### 1.3 Optional but Recommended
- **CUDA-capable GPU** (for faster training)
- **CUDA Toolkit** (if using GPU)
- **Jupyter Notebook** (for interactive development)

---

## 2. System Requirements

### 2.1 Minimum Requirements

#### CPU-Only Setup
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **OS**: Windows 10/11, Linux, or macOS

#### GPU Setup (Recommended)
- **CPU**: Multi-core processor
- **RAM**: 16GB+ recommended
- **GPU**: NVIDIA GPU with CUDA support (4GB+ VRAM)
- **CUDA**: Version 11.0 or higher
- **Storage**: 20GB+ free space (for CUDA toolkit)

### 2.2 Check Your System

#### Check Python Version
```bash
python --version
# Should show Python 3.8 or higher
```

#### Check GPU (if available)
```bash
# Windows
nvidia-smi

# Linux/Mac
lspci | grep -i nvidia
```

---

## 3. Step 1: Clone/Download Project

### 3.1 Option A: Clone from GitHub

#### If you have Git installed:
```bash
# Navigate to your projects directory
cd "D:\Documents\Computer Vision\Projects"

# Clone the repository
git clone https://github.com/abdullah-azi/apple-detection.git

# Navigate into project directory
cd apple-detection
```

### 3.2 Option B: Download ZIP

1. Go to: https://github.com/abdullah-azi/apple-detection
2. Click "Code" → "Download ZIP"
3. Extract to your desired location
4. Navigate into the extracted folder

### 3.3 Verify Project Structure

After cloning/downloading, you should see:
```
apple-detection/
├── Documents/
├── data/
├── src/
├── configs/
├── checkpoints/
├── results/
├── notebooks/
├── README.md
└── .gitignore
```

---

## 4. Step 2: Python Environment Setup

### 4.1 Create Virtual Environment

#### Why Use Virtual Environment?
- Isolates project dependencies
- Prevents conflicts with other projects
- Makes dependency management easier

#### Windows (PowerShell)
```powershell
# Navigate to project directory
cd "D:\Documents\Computer Vision\Projects\apple-detection"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**Note**: If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Windows (Command Prompt)
```cmd
cd "D:\Documents\Computer Vision\Projects\apple-detection"
python -m venv venv
venv\Scripts\activate.bat
```

#### Linux/macOS
```bash
cd ~/path/to/apple-detection
python3 -m venv venv
source venv/bin/activate
```

### 4.2 Verify Virtual Environment

After activation, you should see `(venv)` in your terminal prompt:
```bash
(venv) PS D:\Documents\Computer Vision\Projects\apple-detection>
```

### 4.3 Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 5. Step 3: Install Dependencies

### 5.1 Create requirements.txt

First, create a `requirements.txt` file in the project root:

```txt
# Core Dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=9.0.0

# Data Processing
pyyaml>=6.0
tqdm>=4.64.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.12.0

# Optional: Data Augmentation
albumentations>=1.3.0

# Optional: Experiment Tracking
# wandb>=0.13.0
# tensorboard>=2.10.0

# Development Tools (Optional)
# jupyter>=1.0.0
# ipykernel>=6.0.0
# black>=22.0.0
# pylint>=2.15.0
```

### 5.2 Install Dependencies

#### Install All Dependencies
```bash
pip install -r requirements.txt
```

#### Install with GPU Support (PyTorch)

**For CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CPU-only:**
```bash
pip install torch torchvision
```

**Check PyTorch installation:**
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 5.3 Install Dependencies Step-by-Step (Alternative)

If you prefer to install individually:

```bash
# Core packages
pip install torch torchvision
pip install numpy opencv-python Pillow
pip install pyyaml tqdm
pip install matplotlib seaborn

# Optional
pip install albumentations
```

---

## 6. Step 4: Verify Installation

### 6.1 Create Verification Script

Create a file `verify_setup.py` in the project root:

```python
"""Verify that all dependencies are installed correctly."""

import sys

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_imports():
    """Check if required packages can be imported."""
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'Torchvision',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'matplotlib': 'Matplotlib',
    }
    
    results = {}
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✅ {name}")
            results[name] = True
        except ImportError:
            print(f"❌ {name} - Not installed")
            results[name] = False
    
    return all(results.values())

def check_pytorch():
    """Check PyTorch installation and CUDA."""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available (CPU-only mode)")
        
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def main():
    """Run all verification checks."""
    print("=" * 50)
    print("Verifying Setup...")
    print("=" * 50)
    
    checks = []
    checks.append(check_python_version())
    checks.append(check_imports())
    checks.append(check_pytorch())
    
    print("=" * 50)
    if all(checks):
        print("✅ All checks passed! Setup is complete.")
    else:
        print("❌ Some checks failed. Please review the errors above.")
    print("=" * 50)

if __name__ == "__main__":
    main()
```

### 6.2 Run Verification

```bash
python verify_setup.py
```

**Expected Output:**
```
==================================================
Verifying Setup...
==================================================
✅ Python 3.9.7
✅ PyTorch
✅ Torchvision
✅ NumPy
✅ OpenCV
✅ Pillow
✅ PyYAML
✅ tqdm
✅ Matplotlib
✅ PyTorch 2.0.0
✅ CUDA available: NVIDIA GeForce RTX 3060
   CUDA version: 11.8
==================================================
✅ All checks passed! Setup is complete.
==================================================
```

---

## 7. Step 5: Configure Project

### 7.1 Create Configuration File

Create `configs/config.yaml`:

```yaml
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
  annotation_format: "yolo"
  class_names: ["apple"]
  num_classes: 1

# Model Configuration
model:
  architecture: "yolo"
  version: "v5"
  size: "s"
  pretrained: true
  input_size: [640, 640]

# Training Configuration
training:
  num_epochs: 100
  batch_size: 16
  learning_rate: 0.001
  optimizer: "adam"
  num_workers: 4

# Paths
paths:
  checkpoints_dir: "checkpoints"
  results_dir: "results"
  logs_dir: "logs"

# Hardware
hardware:
  device: "auto"  # "auto", "cpu", or "cuda"
```

### 7.2 Create Directory Structure

```bash
# Create necessary directories
mkdir -p data/images/train
mkdir -p data/images/val
mkdir -p data/images/test
mkdir -p data/annotations/train
mkdir -p data/annotations/val
mkdir -p data/annotations/test
mkdir -p data/splits
mkdir -p checkpoints
mkdir -p results
mkdir -p logs
```

**Windows PowerShell:**
```powershell
New-Item -ItemType Directory -Force -Path data/images/train
New-Item -ItemType Directory -Force -Path data/images/val
New-Item -ItemType Directory -Force -Path data/images/test
New-Item -ItemType Directory -Force -Path data/annotations/train
New-Item -ItemType Directory -Force -Path data/annotations/val
New-Item -ItemType Directory -Force -Path data/annotations/test
New-Item -ItemType Directory -Force -Path data/splits
New-Item -ItemType Directory -Force -Path checkpoints
New-Item -ItemType Directory -Force -Path results
New-Item -ItemType Directory -Force -Path logs
```

### 7.3 Create Placeholder Files

Create `.gitkeep` files to preserve empty directories:

```bash
# Create .gitkeep files
touch data/images/train/.gitkeep
touch data/images/val/.gitkeep
touch data/images/test/.gitkeep
touch data/annotations/train/.gitkeep
touch data/annotations/val/.gitkeep
touch data/annotations/test/.gitkeep
touch checkpoints/.gitkeep
touch results/.gitkeep
```

---

## 8. Step 6: Prepare Dataset

### 8.1 Dataset Requirements

- **Format**: Images in JPEG or PNG format
- **Annotations**: YOLO format (.txt files) recommended
- **Structure**: See [Data Specification](Data_Specification.md) for details

### 8.2 Organize Your Dataset

#### Option A: Manual Organization
1. Place training images in `data/images/train/`
2. Place validation images in `data/images/val/`
3. Place test images in `data/images/test/`
4. Place corresponding annotations in `data/annotations/` subdirectories

#### Option B: Use Split Script
Create a script to automatically split your dataset (if you have all images in one folder).

### 8.3 Annotation Format

#### YOLO Format Example
For each image, create a `.txt` file with the same name:

**Example: `apple_001.jpg` → `apple_001.txt`**
```
0 0.5 0.5 0.3 0.4
0 0.2 0.7 0.15 0.2
```

Format: `class_id center_x center_y width height` (all normalized 0-1)

### 8.4 Create Split Files

Create `data/splits/train.txt`, `val.txt`, and `test.txt` with image filenames:

**train.txt:**
```
apple_001.jpg
apple_002.jpg
apple_003.jpg
...
```

---

## 9. Step 7: Test Setup

### 9.1 Test Data Loading

Create `test_setup.py`:

```python
"""Test basic setup and data loading."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test if modules can be imported."""
    try:
        # These will be created during implementation
        # from src.dataset import AppleDataset
        # from src.model import create_model
        print("✅ Module imports (will be available after implementation)")
        return True
    except ImportError as e:
        print(f"⚠️  Module imports not yet available: {e}")
        return True  # Expected during initial setup

def test_config_loading():
    """Test configuration loading."""
    try:
        import yaml
        config_path = Path("configs/config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            print("✅ Configuration file loaded")
            return True
        else:
            print("⚠️  Configuration file not found (create configs/config.yaml)")
            return False
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

def test_directories():
    """Test if required directories exist."""
    required_dirs = [
        "data/images",
        "data/annotations",
        "checkpoints",
        "results",
        "logs"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - Create this directory")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Setup...")
    print("=" * 50)
    
    tests = []
    tests.append(test_imports())
    tests.append(test_config_loading())
    tests.append(test_directories())
    
    print("=" * 50)
    if all(tests):
        print("✅ Setup test passed!")
    else:
        print("⚠️  Some setup steps need attention")
    print("=" * 50)

if __name__ == "__main__":
    main()
```

### 9.2 Run Test

```bash
python test_setup.py
```

---

## 10. Troubleshooting

### 10.1 Common Issues

#### Issue: Python Not Found
**Error**: `'python' is not recognized as an internal or external command`

**Solution**:
1. Install Python from python.org
2. Add Python to PATH during installation
3. Or use `py` command instead: `py -m venv venv`

#### Issue: Virtual Environment Activation Fails
**Error**: `ExecutionPolicy` error on Windows

**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Issue: pip Install Fails
**Error**: `ERROR: Could not find a version that satisfies the requirement`

**Solutions**:
1. Upgrade pip: `python -m pip install --upgrade pip`
2. Use specific version: `pip install torch==2.0.0`
3. Check Python version compatibility

#### Issue: CUDA Not Available
**Error**: `CUDA not available` even with GPU

**Solutions**:
1. Install CUDA-compatible PyTorch:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
2. Verify CUDA installation: `nvidia-smi`
3. Check PyTorch CUDA support:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

#### Issue: Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size in config
2. Use smaller model size
3. Enable gradient checkpointing
4. Use CPU if GPU memory is insufficient

#### Issue: Import Errors
**Error**: `ModuleNotFoundError: No module named 'xxx'`

**Solution**:
```bash
pip install <module_name>
# Or reinstall all:
pip install -r requirements.txt
```

### 10.2 Getting Help

#### Check Documentation
- Review [Project Overview](Project_Overview.md)
- Check [Requirements](Requirements.md)
- See [Data Specification](Data_Specification.md)

#### Verify Installation
```bash
python verify_setup.py
```

#### Check Logs
- Review error messages carefully
- Check Python version compatibility
- Verify file paths are correct

---

## 11. Quick Start Checklist

### Setup Checklist

- [ ] **Step 1**: Clone/download project
- [ ] **Step 2**: Create virtual environment
- [ ] **Step 3**: Activate virtual environment
- [ ] **Step 4**: Install dependencies
- [ ] **Step 5**: Verify installation (`python verify_setup.py`)
- [ ] **Step 6**: Create configuration file
- [ ] **Step 7**: Create directory structure
- [ ] **Step 8**: Prepare dataset (images + annotations)
- [ ] **Step 9**: Test setup (`python test_setup.py`)
- [ ] **Step 10**: Ready to start development!

### Next Steps After Setup

1. **Review Documentation**:
   - Read [Project Overview](Project_Overview.md)
   - Review [Requirements](Requirements.md)
   - Check [Architecture](Architecture.md)

2. **Prepare Dataset**:
   - Collect images with apples
   - Create annotations (YOLO format)
   - Organize into train/val/test splits

3. **Start Implementation**:
   - Begin with data loading module
   - Implement model architecture
   - Create training script

---

## 12. Additional Resources

### 12.1 Useful Commands

#### Activate Virtual Environment
```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
venv\Scripts\activate.bat

# Linux/macOS
source venv/bin/activate
```

#### Deactivate Virtual Environment
```bash
deactivate
```

#### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

#### Check Installed Packages
```bash
pip list
```

### 12.2 IDE Setup

#### VS Code
1. Install Python extension
2. Select Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
3. Choose virtual environment: `.\venv\Scripts\python.exe`

#### PyCharm
1. Open project
2. File → Settings → Project → Python Interpreter
3. Add interpreter → Existing → Select `venv\Scripts\python.exe`

### 12.3 Jupyter Notebook Setup

#### Install Jupyter
```bash
pip install jupyter ipykernel
```

#### Add Virtual Environment to Jupyter
```bash
python -m ipykernel install --user --name=apple-detection
```

#### Launch Jupyter
```bash
jupyter notebook
```

---

## 13. Verification Scripts

### 13.1 Complete Verification Script

Save as `verify_setup.py` (already provided in Step 6)

### 13.2 Quick Test Script

Save as `quick_test.py`:

```python
"""Quick test of basic functionality."""

import torch
import numpy as np
import cv2
from PIL import Image

print("Testing basic imports...")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"OpenCV: {cv2.__version__}")

# Test tensor creation
x = torch.randn(3, 3)
print(f"\n✅ Tensor creation works: {x.shape}")

# Test CUDA (if available)
if torch.cuda.is_available():
    x_gpu = x.cuda()
    print(f"✅ CUDA tensor creation works: {x_gpu.device}")
else:
    print("⚠️  CUDA not available (CPU-only)")

print("\n✅ All basic tests passed!")
```

Run with: `python quick_test.py`

---

## 14. Environment Variables (Optional)

### 14.1 Set Environment Variables

#### Windows (PowerShell)
```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONPATH = "D:\Documents\Computer Vision\Projects\apple-detection"
```

#### Windows (Command Prompt)
```cmd
set CUDA_VISIBLE_DEVICES=0
set PYTHONPATH=D:\Documents\Computer Vision\Projects\apple-detection
```

#### Linux/macOS
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/apple-detection
```

---

## 15. Summary

### Setup Complete When:
- ✅ Virtual environment created and activated
- ✅ All dependencies installed
- ✅ Verification script passes
- ✅ Configuration file created
- ✅ Directory structure ready
- ✅ Dataset prepared (or ready to prepare)

### You're Ready To:
- Start implementing the code
- Load and preprocess data
- Train models
- Run inference
- Evaluate results

---

**Setup Guide End**

*Follow this guide step-by-step to set up your development environment. If you encounter issues, refer to the Troubleshooting section.*

