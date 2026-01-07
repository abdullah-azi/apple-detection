# Google Colab Setup Guide

## 🚀 Running Apple Detection on Google Colab

Yes, you can absolutely run this project on Google Colab! In fact, Colab is a great choice because:

- ✅ **Free GPU access** (T4 GPU with 16GB VRAM)
- ✅ **No local setup required** - everything runs in the cloud
- ✅ **Pre-installed libraries** - many dependencies are already available
- ✅ **Easy sharing** - share notebooks with others

## 📋 Quick Start

### Step 1: Enable GPU Runtime

1. Open a new Colab notebook
2. Go to **Runtime** → **Change runtime type**
3. Set **Hardware accelerator** to **GPU (T4)**
4. Click **Save**

### Step 2: Upload Your Project

You have two options:

#### Option A: Upload as ZIP
1. Create a ZIP file of your project (exclude `venv/`, `__pycache__/`, `.git/`)
2. Upload it to Colab
3. Extract it to `/content/apple-detection/`

#### Option B: Clone from GitHub
If your project is on GitHub:
```python
!git clone https://github.com/yourusername/apple-detection.git
%cd apple-detection
```

### Step 3: Install Dependencies

```python
# Install PyTorch with CUDA (Colab usually has this)
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
!pip install -r requirements.txt
```

### Step 4: Upload Your Dataset

#### Option A: Upload ZIP file
```python
from google.colab import files
import zipfile

uploaded = files.upload()
# Extract to /content/apple-detection/data/
```

#### Option B: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy from Drive
!cp -r /content/drive/MyDrive/your-dataset/* /content/apple-detection/data/
```

### Step 5: Train the Model

Use the Colab-optimized configuration:

```python
!python src/train.py --config configs/config_colab.yaml
```

## 📁 Configuration Differences

The `config_colab.yaml` file is optimized for Colab with:

- **GPU usage** (`device: "cuda"`)
- **Larger batch size** (16 instead of 4)
- **Full augmentation** (rotation, saturation enabled)
- **Larger input size** (640x640 instead of 416x416)
- **Absolute paths** for Colab's file system (`/content/apple-detection/...`)

## 💾 Saving Your Work

**Important:** Colab sessions are temporary. Always save to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Save checkpoints
!cp -r /content/apple-detection/checkpoints /content/drive/MyDrive/

# Save results
!cp -r /content/apple-detection/results /content/drive/MyDrive/
```

## ⚠️ Important Notes

### Session Timeout
- Colab disconnects after ~90 minutes of inactivity
- Keep the tab active during training
- Save checkpoints frequently

### GPU Limits
- Free Colab has daily usage limits
- If you hit limits, wait a few hours or consider Colab Pro
- Monitor usage in Runtime → Manage sessions

### File System
- Colab's file system is temporary
- Files are deleted when the session ends
- Always save important files to Google Drive

### Path Differences
- Local: `data/images/train`
- Colab: `/content/apple-detection/data/images/train`
- Use absolute paths in Colab config

## 🔄 Loading Saved Checkpoints

If you saved checkpoints to Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy checkpoints back
!cp -r /content/drive/MyDrive/checkpoints /content/apple-detection/
```

## 📊 Monitoring Training

### Option 1: Print Statements
The training script should print progress automatically.

### Option 2: TensorBoard (if configured)
```python
# In a new cell
%load_ext tensorboard
%tensorboard --logdir /content/apple-detection/logs
```

## 🎯 Recommended Workflow

1. **Setup** (one time):
   - Upload project files
   - Install dependencies
   - Upload dataset

2. **Training** (each session):
   - Mount Google Drive
   - Load previous checkpoints (if resuming)
   - Start training
   - Save checkpoints to Drive periodically

3. **Inference**:
   - Load saved checkpoint from Drive
   - Run inference on test images
   - Save results to Drive

## 🆚 Colab vs Local: Key Differences

| Feature | Local (CPU) | Colab (GPU) |
|---------|-------------|-------------|
| Device | CPU | GPU (T4) |
| Batch Size | 4 | 16 |
| Input Size | 416x416 | 640x640 |
| Augmentation | Light | Full |
| Training Speed | Slow | Fast |
| Cost | Free | Free (with limits) |
| Persistence | Permanent | Temporary |

## 🐛 Troubleshooting

### GPU Not Available
- Check Runtime → Change runtime type → GPU is selected
- Verify with: `torch.cuda.is_available()`

### Out of Memory
- Reduce batch size in `config_colab.yaml`
- Use smaller model size (change `size: "s"` to `size: "n"`)

### Import Errors
- Make sure you're in the project directory: `%cd /content/apple-detection`
- Check that all dependencies are installed

### Path Errors
- Use absolute paths in Colab config
- Verify paths exist: `!ls /content/apple-detection/data/images/train`

## 📚 Additional Resources

- [Colab Setup Notebook](notebooks/colab_setup.ipynb) - Interactive setup guide
- [Colab Config](configs/config_colab.yaml) - GPU-optimized configuration
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

---

**Happy Training on Colab! 🍎🚀**

