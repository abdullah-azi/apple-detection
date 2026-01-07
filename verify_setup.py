"""Verify that all dependencies are installed correctly."""

import sys

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        print(f"✅ Python {version_str}")
        
        # Special note for Python 3.13
        if version.major == 3 and version.minor == 13:
            print("   ⚠️  Python 3.13 is very new - some packages may need updates")
            print("   💡 If you encounter issues, consider using Python 3.11 or 3.12")
        
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

