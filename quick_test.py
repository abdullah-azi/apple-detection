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

