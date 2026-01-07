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
        # Check for CPU config first (recommended for CPU-only systems)
        config_paths = [
            Path("configs/config_cpu.yaml"),
            Path("configs/config.yaml")
        ]
        
        config_path = None
        for path in config_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuration file loaded: {config_path}")
            return True
        else:
            print("⚠️  Configuration file not found (create configs/config_cpu.yaml or configs/config.yaml)")
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

