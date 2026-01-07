# Quick Setup Guide: Python 3.13.7 CPU-Only

## 🚀 Quick Start (3 Steps)

### Step 1: Create Virtual Environment
```powershell
# Navigate to project
cd "D:\Documents\Computer Vision\Projects\apple-detection"

# Create virtual environment
python -m venv venv

# Activate (PowerShell)
.\venv\Scripts\Activate.ps1

# Or activate (CMD)
venv\Scripts\activate.bat
```

**If you get execution policy error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 2: Run Automated Setup Script
```powershell
python setup_python313_cpu.py
```

This script will:
- ✅ Check Python 3.13.7
- ✅ Upgrade pip
- ✅ Install CPU-only PyTorch (optimized for CPU)
- ✅ Install all dependencies
- ✅ Create directory structure
- ✅ Verify installation

### Step 3: Verify Setup
```powershell
python verify_setup.py
python test_setup.py
```

---

## 📋 Manual Setup (Alternative)

If you prefer manual setup or the script fails:

### 1. Upgrade pip
```powershell
python -m pip install --upgrade pip
```

### 2. Install CPU-only PyTorch
```powershell
# Recommended: CPU-only version (smaller, faster)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Other Dependencies
```powershell
pip install numpy>=1.21.0 opencv-python>=4.5.0 Pillow>=9.0.0
pip install pyyaml>=6.0 tqdm>=4.64.0
pip install matplotlib>=3.5.0 seaborn>=0.12.0
pip install albumentations>=1.3.0
```

Or install from requirements.txt:
```powershell
pip install -r requirements.txt
```

---

## ⚠️ Python 3.13.7 Compatibility Notes

### Known Considerations:
1. **Python 3.13 is very new** (released October 2024)
2. **Some packages may need updates** for full compatibility
3. **Most packages work**, but you may encounter:
   - Warnings about deprecated features
   - Some packages needing to be installed from source
   - Occasional compatibility issues with very old packages

### If You Encounter Issues:

#### Option 1: Update Packages
```powershell
pip install --upgrade <package-name>
```

#### Option 2: Install from Source
```powershell
pip install --no-binary <package-name> <package-name>
```

#### Option 3: Use Python 3.11 or 3.12
If compatibility issues persist, consider:
- Python 3.12.7 (recommended alternative)
- Python 3.11.9 (stable, well-tested)

---

## 🔧 Configuration

Your CPU-optimized configuration is in: `configs/config_cpu.yaml`

**Key settings for CPU-only:**
- Model size: `n` (nano - smallest)
- Batch size: `4` (small for CPU)
- Input size: `416x416` (smaller than 640x640)
- Num workers: `2` (matches 2 CPU cores)
- Device: `cpu` (explicitly set)

---

## ✅ Verification Checklist

After setup, verify:

- [ ] Python 3.13.7 detected
- [ ] Virtual environment active
- [ ] PyTorch installed (CPU-only)
- [ ] All dependencies installed
- [ ] Configuration file exists (`configs/config_cpu.yaml`)
- [ ] Directory structure created
- [ ] Verification script passes

---

## 🐛 Troubleshooting

### Issue: Package Installation Fails
**Solution:**
```powershell
# Try upgrading pip first
python -m pip install --upgrade pip

# Then try installing with --no-cache-dir
pip install --no-cache-dir <package-name>
```

### Issue: PyTorch Installation Fails
**Solution:**
```powershell
# Try standard installation instead of CPU-only
pip install torch torchvision

# Or install specific version
pip install torch==2.1.0 torchvision==0.16.0
```

### Issue: Import Errors
**Solution:**
```powershell
# Reinstall package
pip uninstall <package-name>
pip install <package-name>

# Or reinstall all
pip install -r requirements.txt --force-reinstall
```

### Issue: Out of Memory
**Solution:**
- Reduce batch size to 2-4 in config
- Use smaller model (nano)
- Reduce input image size (320x320)
- Close other applications

---

## 📚 Next Steps

1. **Prepare Dataset:**
   - Place images in `data/images/train/`, `val/`, `test/`
   - Place annotations in `data/annotations/` subdirectories

2. **Review Configuration:**
   - Check `configs/config_cpu.yaml`
   - Adjust batch size, epochs, etc. as needed

3. **Start Training:**
   - Follow project documentation for training scripts
   - Monitor memory usage during training

---

## 💡 Performance Tips for CPU Training

1. **Start Small:**
   - Use 50-100 images for initial testing
   - Train for 10-20 epochs first

2. **Optimize Settings:**
   - Batch size: 4 (or 2 if memory issues)
   - Model: nano size
   - Input: 416x416 or 320x320

3. **Be Patient:**
   - CPU training is slower (hours per epoch)
   - Train overnight or during breaks
   - Use early stopping to avoid overfitting

4. **Monitor Resources:**
   - Watch RAM usage (should be 4-8GB)
   - Close other applications
   - Use Task Manager to monitor

---

## 📞 Need Help?

1. Run verification: `python verify_setup.py`
2. Check logs for specific error messages
3. Review main Setup_Guide.md for detailed instructions
4. Check Python 3.13 release notes for compatibility info

---

**Setup Complete!** 🎉

You're ready to start working with the apple detection project on Python 3.13.7 CPU-only setup.

