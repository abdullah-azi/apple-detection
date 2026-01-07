# 🚀 Quick Start: Google Colab Setup

## What Was Created

I've created a **complete Colab setup** that works from scratch with **no Kaggle account needed**:

### Files Created:

1. **`notebooks/colab_setup.ipynb`** - Complete setup notebook
   - Step-by-step cells
   - Automatic dataset filtering
   - Ready to use

2. **`scripts/filter_apple_dataset.py`** - Standalone filtering script
   - Can be used independently
   - Filters multi-fruit dataset to apples only

3. **`configs/config_colab.yaml`** - GPU-optimized configuration
   - Already created in previous step

4. **`Documents/Colab_Complete_Guide.md`** - Detailed documentation

## 🎯 How to Use

### Step 1: Open Colab
1. Go to https://colab.research.google.com
2. Sign in with Google account
3. Click **File** → **Upload notebook**
4. Upload `notebooks/colab_setup.ipynb`

### Step 2: Enable GPU
**IMPORTANT:** Do this first!
- **Runtime** → **Change runtime type** → **GPU (T4)** → **Save**

### Step 3: Run Cells in Order

The notebook has these main sections:

1. **Cell 1**: Verify GPU ✅
2. **Cell 2**: Upload project (ZIP or GitHub)
3. **Cell 3**: Install dependencies
4. **Cell 4**: Download/upload dataset (3 options, no Kaggle needed!)
5. **Cell 5**: Filter dataset for apples only 🍎
6. **Cell 6**: Verify filtered dataset
7. **Cell 7**: Train model
8. **Cell 8**: Run inference
9. **Cell 9**: Save to Google Drive

## 📥 Getting the Dataset (No Kaggle Account)

You have **3 options**:

### Option 1: Direct Download Link
- If you have a download URL, paste it in Cell 4, Option A
- Script downloads and extracts automatically

### Option 2: Upload ZIP File (Easiest)
1. Download the Fruit Detection Dataset ZIP to your computer
2. In Colab, go to Cell 4, Option B
3. Click "Choose Files"
4. Select the ZIP file
5. Done! It extracts automatically

### Option 3: Google Drive
1. Upload dataset ZIP to Google Drive
2. Mount Drive in Colab (Cell 4, Option C)
3. Point to dataset location

## 🍎 What the Filtering Does

The script automatically:
- ✅ Finds all images with apples
- ✅ Removes other fruits from annotations
- ✅ Keeps only apple bounding boxes
- ✅ Splits into train/val/test (70/15/15)
- ✅ Creates proper directory structure

**Result:** You get a clean apple-only dataset ready for training!

## ⚡ Quick Checklist

- [ ] Open Colab
- [ ] Enable GPU
- [ ] Upload `colab_setup.ipynb`
- [ ] Run Cell 1 (verify GPU)
- [ ] Run Cell 2 (upload project)
- [ ] Run Cell 3 (install dependencies)
- [ ] Run Cell 4 (download/upload dataset)
- [ ] Run Cell 5 (filter for apples)
- [ ] Run Cell 6 (verify dataset)
- [ ] Run Cell 7 (train model)
- [ ] Run Cell 9 (save to Drive)

## 💡 Tips

1. **Keep tab active** - Prevents session timeout
2. **Save frequently** - Use Cell 9 to save to Drive
3. **Check GPU** - Make sure GPU is enabled before training
4. **Be patient** - Filtering large datasets takes time

## 🐛 Common Issues

**GPU not available?**
- Check Runtime → Change runtime type → GPU

**Dataset not found?**
- Verify the ZIP was uploaded/extracted
- Check the path in the cell output

**No apple images found?**
- Check if dataset actually contains apples
- Verify class ID (usually '0')

## 📚 More Help

- See `Documents/Colab_Complete_Guide.md` for detailed instructions
- Check notebook cell comments for explanations
- Review `scripts/filter_apple_dataset.py` for filtering logic

---

**You're all set! Just open the notebook and follow the cells! 🎉**

