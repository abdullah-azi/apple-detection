"""
Setup script for Python 3.13.7 CPU-only systems.
This script automates the setup process for your specific configuration.
"""

import sys
import subprocess
import os
from pathlib import Path

def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")

def check_python_version():
    """Check if Python 3.13.7 is being used."""
    version = sys.version_info
    if version.major == 3 and version.minor == 13:
        print(f"✅ Detected Python {version.major}.{version.minor}.{version.micro}")
        print("   ⚠️  Python 3.13 is very new - some packages may need updates")
        print("   💡 If you encounter compatibility issues, consider Python 3.11 or 3.12")
        return True
    elif version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        print("   ℹ️  This script is optimized for Python 3.13.7")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} detected")
        print("   Need Python 3.8 or higher")
        return False

def check_venv():
    """Check if virtual environment is activated."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is active")
        return True
    else:
        print("⚠️  Virtual environment not detected")
        print("   💡 Create and activate venv first:")
        print("      python -m venv venv")
        print("      .\\venv\\Scripts\\Activate.ps1  (PowerShell)")
        print("      venv\\Scripts\\activate.bat     (CMD)")
        return False

def upgrade_pip():
    """Upgrade pip to latest version."""
    print("\n📦 Upgrading pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ pip upgraded successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to upgrade pip")
        return False

def install_cpu_pytorch():
    """Install CPU-only PyTorch (recommended for CPU systems)."""
    print("\n📦 Installing CPU-only PyTorch...")
    print("   This is optimized for CPU-only systems (smaller, faster installation)")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
        print("✅ CPU-only PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install PyTorch")
        print("   💡 Trying standard PyTorch installation...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision"
            ])
            print("✅ PyTorch installed (standard version)")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install PyTorch")
            return False

def install_requirements():
    """Install requirements from requirements.txt."""
    print("\n📦 Installing requirements from requirements.txt...")
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("⚠️  requirements.txt not found")
        return False
    
    try:
        # Install requirements, skipping torch/torchvision (already installed)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", str(requirements_file),
            "--no-deps"  # Skip dependencies to avoid conflicts
        ])
        # Then install with dependencies for non-torch packages
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "numpy>=1.21.0",
            "opencv-python>=4.5.0",
            "Pillow>=9.0.0",
            "pyyaml>=6.0",
            "tqdm>=4.64.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "albumentations>=1.3.0"
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        print("   💡 Try installing manually: pip install -r requirements.txt")
        return False

def verify_installation():
    """Run verification script."""
    print("\n🔍 Running verification...")
    verify_script = Path("verify_setup.py")
    
    if verify_script.exists():
        try:
            subprocess.check_call([sys.executable, str(verify_script)])
            return True
        except subprocess.CalledProcessError:
            print("⚠️  Verification script found issues")
            return False
    else:
        print("⚠️  verify_setup.py not found")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directory structure...")
    directories = [
        "data/images/train",
        "data/images/val",
        "data/images/test",
        "data/annotations/train",
        "data/annotations/val",
        "data/annotations/test",
        "data/splits",
        "checkpoints",
        "results",
        "logs"
    ]
    
    created = 0
    for dir_path in directories:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        if not any(path.iterdir()):  # If empty, create .gitkeep
            (path / ".gitkeep").touch(exist_ok=True)
        created += 1
    
    print(f"✅ Created {created} directories")
    return True

def main():
    """Main setup function."""
    print_header("Python 3.13.7 CPU-Only Setup Script")
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup aborted: Python version check failed")
        return False
    
    # Check virtual environment
    venv_active = check_venv()
    if not venv_active:
        response = input("\nContinue anyway? (y/n): ").lower()
        if response != 'y':
            print("\n⚠️  Setup aborted. Please activate virtual environment first.")
            return False
    
    # Upgrade pip
    upgrade_pip()
    
    # Install CPU-only PyTorch
    if not install_cpu_pytorch():
        print("\n❌ Setup aborted: PyTorch installation failed")
        return False
    
    # Install other requirements
    install_requirements()
    
    # Create directories
    create_directories()
    
    # Verify installation
    verify_installation()
    
    print_header("Setup Complete!")
    print("✅ Next steps:")
    print("   1. Verify setup: python verify_setup.py")
    print("   2. Test setup: python test_setup.py")
    print("   3. Check config: configs/config_cpu.yaml")
    print("   4. Prepare your dataset in data/ directory")
    print("\n💡 Tips for Python 3.13.7:")
    print("   - If packages fail, try: pip install --upgrade <package>")
    print("   - Some packages may need to be installed from source")
    print("   - Consider Python 3.11/3.12 if you encounter issues")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        sys.exit(1)

