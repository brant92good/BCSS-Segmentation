#!/usr/bin/env python3
"""
Quick validation script to check project setup
"""

import os
import sys
from pathlib import Path


def check_structure():
    """Check if project structure is complete."""
    print("=" * 60)
    print("BCSS Segmentation - Project Setup Validator")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    
    # Required directories
    required_dirs = [
        'src',
        'config',
        'ckpt',
        'logs',
        'output'
    ]
    
    # Required Python files
    required_files = {
        'src/__init__.py': 'Package init',
        'src/dataset.py': 'Dataset classes',
        'src/model.py': 'U-Net model',
        'src/losses.py': 'Loss functions',
        'src/augmentation.py': 'Data augmentation',
        'src/utils.py': 'Utility functions',
        'config/__init__.py': 'Config package',
        'config/config.py': 'Configuration',
        'train_main.py': 'Training entry point',
        'visualize_da.py': 'Visualization entry point',
        'requirements.txt': 'Dependencies',
        'README.md': 'Documentation',
        'USAGE.md': 'Usage guide'
    }
    
    # Optional files
    optional_files = {
        'setup.sh': 'Setup script',
        'PROJECT_STRUCTURE.md': 'Structure documentation',
        'COMPLETION_SUMMARY.md': 'Completion summary',
        '__main__.py': 'Unified entry point'
    }
    
    all_ok = True
    
    # Check directories
    print("\n📁 Checking directories...")
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (missing)")
            all_ok = False
    
    # Check required files
    print("\n📄 Checking required files...")
    for file_path, description in required_files.items():
        full_path = base_dir / file_path
        if full_path.exists() and full_path.is_file():
            size = full_path.stat().st_size
            print(f"  ✅ {file_path:<30} ({size:>8} bytes) - {description}")
        else:
            print(f"  ❌ {file_path:<30} (missing) - {description}")
            all_ok = False
    
    # Check optional files
    print("\n📌 Checking optional files...")
    for file_path, description in optional_files.items():
        full_path = base_dir / file_path
        if full_path.exists() and full_path.is_file():
            size = full_path.stat().st_size
            print(f"  ✅ {file_path:<30} ({size:>8} bytes) - {description}")
        else:
            print(f"  ⚠️  {file_path:<30} (optional) - {description}")
    
    # Check Python syntax
    print("\n🐍 Checking Python files syntax...")
    import py_compile
    
    for file_path in required_files.keys():
        if file_path.endswith('.py'):
            full_path = base_dir / file_path
            try:
                py_compile.compile(str(full_path), doraise=True)
                print(f"  ✅ {file_path} (syntax OK)")
            except py_compile.PyCompileError as e:
                print(f"  ❌ {file_path} (syntax error: {e})")
                all_ok = False
    
    # Check imports
    print("\n📦 Checking imports...")
    sys.path.insert(0, str(base_dir))
    
    try:
        from config.config import (
            TRAIN_IMAGE_PATH, VAL_IMAGE_PATH, BATCH_SIZE,
            N_CLASSES, MAX_EPOCHS
        )
        print(f"  ✅ config.config (imports OK)")
        print(f"     - N_CLASSES={N_CLASSES}, BATCH_SIZE={BATCH_SIZE}, MAX_EPOCHS={MAX_EPOCHS}")
    except ImportError as e:
        print(f"  ❌ config.config (import error: {e})")
        all_ok = False
    
    try:
        from src.model import UNet
        print(f"  ✅ src.model (UNet imports OK)")
    except ImportError as e:
        print(f"  ❌ src.model (import error: {e})")
        all_ok = False
    
    try:
        from src.dataset import BCSSDataset, BCSSDatasetTest
        print(f"  ✅ src.dataset (BCSSDataset imports OK)")
    except ImportError as e:
        print(f"  ❌ src.dataset (import error: {e})")
        all_ok = False
    
    try:
        from src.losses import DiceLoss, pixel_accuracy, mIoU
        print(f"  ✅ src.losses (DiceLoss imports OK)")
    except ImportError as e:
        print(f"  ❌ src.losses (import error: {e})")
        all_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ PROJECT SETUP COMPLETE - Ready to use!")
        print("\nQuick start:")
        print("  # Train with Kaggle submission")
        print("  python train_main.py")
        print("\n  # Visualize data augmentation")
        print("  python visualize_da.py --mode single")
        print("\n  # View help")
        print("  python train_main.py --help")
        print("  python visualize_da.py --help")
    else:
        print("⚠️  Some issues detected - please review above")
        return 1
    
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(check_structure())
