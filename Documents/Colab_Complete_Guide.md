# Complete Google Colab Setup Guide - From Zero to Training

This guide walks you through setting up the Apple Detection project on Google Colab **from scratch**, with **no Kaggle account required**.

## 🎯 What You'll Get

- ✅ Complete setup from zero
- ✅ Download Fruit Detection Dataset (no account needed)
- ✅ Automatic filtering to extract only apple images
- ✅ Ready-to-train dataset
- ✅ GPU-optimized configuration

## 📋 Prerequisites

- Google account (for Colab)
- Internet connection
- **No Kaggle account needed!**

## 🚀 Step-by-Step Instructions

### Step 1: Open Google Colab

1. Go to https://colab.research.google.com
2. Sign in with your Google account
3. Click **File** → **New notebook**

### Step 2: Enable GPU

**IMPORTANT:** Do this first!

1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU (T4)**
3. Click **Save**

### Step 3: Upload the Setup Notebook

1. Upload the `notebooks/colab_setup.ipynb` file to Colab
2. Or copy the cells from the notebook into a new Colab notebook

### Step 4: Run the Setup Cells

Follow the cells in order:

#### Cell 1: Verify GPU
- Checks if GPU is available
- Shows GPU information

#### Cell 2: Upload Project
- Option A: Clone from GitHub (if you have a repo)
- Option B: Upload project as ZIP file

**To create ZIP:**
1. On your computer, create a ZIP of your project
2. Exclude: `venv/`, `__pycache__/`, `.git/`, `*.pyc`
3. Upload the ZIP in Colab

#### Cell 3: Install Dependencies
- Installs PyTorch with CUDA
- Installs all project dependencies
- Verifies installation

#### Cell 4: Download Dataset

You have **3 options** (no Kaggle account needed):

**Option A: Direct Download Link**
- If you have a direct download URL, paste it in the cell
- The script will download and extract automatically

**Option B: Upload ZIP File**
- Download the Fruit Detection Dataset to your computer
- Upload the ZIP file in Colab
- Script will extract it automatically

**Option C: Google Drive**
- If dataset is in Google Drive, mount Drive and point to the dataset

#### Cell 5: Filter for Apple Images
- Automatically filters the multi-fruit dataset
- Extracts ONLY images containing apples
- Removes other fruits from annotations
- Splits into train/val/test (70/15/15)

#### Cell 6: Verify Dataset
- Shows final dataset structure
- Displays image counts and bounding box statistics
- Confirms dataset is ready for training

#### Cell 7: Train Model
- Ready to start training
- Uses GPU-optimized configuration
- Uncomment the training command to start

#### Cell 8: Run Inference
- Test your trained model
- Upload test images
- View detection results

#### Cell 9: Save to Drive
- Saves checkpoints to Google Drive
- Saves results and config
- Prevents data loss

## 📥 Getting the Dataset (No Kaggle Account)

### Method 1: Direct Download

If you find a direct download link:
1. Paste the URL in Cell 4, Option A
2. Run the cell
3. Dataset downloads automatically

### Method 2: Upload ZIP

1. Download the dataset ZIP file to your computer
2. In Colab, go to Cell 4, Option B
3. Click "Choose Files"
4. Select the ZIP file
5. Upload and extract automatically

### Method 3: Google Drive

1. Upload dataset ZIP to Google Drive
2. In Colab, mount Google Drive (Cell 4, Option C)
3. Point to the dataset location
4. Extract if needed

## 🔍 What the Filtering Script Does

The filtering process:

1. **Finds dataset structure** - Automatically detects images/ and labels/ folders
2. **Analyzes classes** - Checks which class ID represents apples (usually 0)
3. **Filters images** - Keeps only images that contain apples
4. **Filters annotations** - Removes bounding boxes for other fruits
5. **Splits dataset** - Creates train/val/test splits (70/15/15)
6. **Organizes output** - Creates proper directory structure for your project

## 📊 Expected Results

After filtering, you should have:

- **Train set**: ~70% of apple images
- **Val set**: ~15% of apple images  
- **Test set**: ~15% of apple images
- **All annotations**: Only apple bounding boxes (class 0)

Example output:
```
TRAIN:
  Images: 1200
  Annotations: 1200
  Apple bounding boxes: 1850
  Avg boxes per image: 1.54

VAL:
  Images: 250
  Annotations: 250
  Apple bounding boxes: 390
  Avg boxes per image: 1.56

TEST:
  Images: 250
  Annotations: 250
  Apple bounding boxes: 385
  Avg boxes per image: 1.54
```

## ⚠️ Important Notes

### Session Timeout
- Colab sessions disconnect after ~90 minutes of inactivity
- **Keep the tab active during training**
- Save checkpoints frequently to Drive

### GPU Limits
- Free Colab has daily usage limits
- If you hit limits, wait a few hours
- Consider Colab Pro for more GPU time

### File Persistence
- Colab files are **temporary**
- Files are deleted when session ends
- **Always save to Google Drive**

### Dataset Size
- The Fruit Detection Dataset is large (~8479 images)
- Filtering may take a few minutes
- Be patient during the filtering process

## 🐛 Troubleshooting

### GPU Not Available
- Check Runtime → Change runtime type → GPU is selected
- Verify with: `torch.cuda.is_available()`

### Dataset Not Found
- Check the dataset path
- Verify the ZIP was extracted correctly
- Try re-uploading the dataset

### No Apple Images Found
- Check if the dataset actually contains apples
- Verify the class ID (try '0' or '1')
- Check annotation file format

### Import Errors
- Make sure all dependencies are installed
- Check that PROJECT_DIR is set correctly
- Verify project structure

### Out of Memory
- Reduce batch size in config
- Use smaller model size
- Close other Colab tabs

## 💡 Tips

1. **Save frequently** - Use the save-to-Drive cell regularly
2. **Monitor training** - Check progress in logs
3. **Test incrementally** - Start with fewer epochs to test
4. **Keep tab active** - Prevent session timeout
5. **Use Drive for large files** - Don't upload huge ZIPs directly

## 📚 Next Steps

After setup:

1. **Review dataset** - Check a few images and annotations
2. **Adjust config** - Modify `config_colab.yaml` if needed
3. **Start training** - Uncomment training command
4. **Monitor progress** - Watch training logs
5. **Save checkpoints** - Regular saves to Drive
6. **Test model** - Run inference on test images
7. **Evaluate** - Check model performance

## 🎉 Success Checklist

- [ ] GPU enabled and verified
- [ ] Project uploaded/extracted
- [ ] Dependencies installed
- [ ] Dataset downloaded/uploaded
- [ ] Dataset filtered for apples
- [ ] Train/val/test splits created
- [ ] Dataset verified and ready
- [ ] Config file ready
- [ ] Ready to train!

---

**Happy Training! 🍎🚀**

For questions or issues, check the troubleshooting section or review the notebook cells for detailed comments.

