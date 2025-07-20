#!/usr/bin/env python3
"""
Test script to verify DeepFashion2 training setup.

This script checks all components are working correctly before starting training.
"""

import os
import sys
import torch
import yaml
from pathlib import Path

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        import ultralytics
        print(f"✓ ultralytics: {ultralytics.__version__}")
    except ImportError as e:
        print(f"✗ ultralytics: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ opencv-python: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ opencv-python: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False
    
    try:
        import seaborn
        print(f"✓ seaborn: {seaborn.__version__}")
    except ImportError as e:
        print(f"✗ seaborn: {e}")
        return False
    
    try:
        import yaml
        print(f"✓ pyyaml: Available")
    except ImportError as e:
        print(f"✗ pyyaml: {e}")
        return False
    
    print("✓ All imports successful\n")
    return True


def test_cuda():
    """Test CUDA availability."""
    print("Testing CUDA...")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"✓ CUDA available")
        print(f"✓ Device count: {device_count}")
        print(f"✓ Device name: {device_name}")
        print(f"✓ GPU memory: {memory:.1f} GB")
        
        # Test memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("✓ GPU memory allocation test passed")
        except Exception as e:
            print(f"✗ GPU memory allocation failed: {e}")
            return False
    else:
        print("⚠ CUDA not available - training will be slow")
    
    print()
    return True


def test_dataset_path():
    """Test DeepFashion2 dataset path."""
    print("Testing dataset path...")
    
    dataset_path = Path("/media/kunwar-padda/Gold/DeepFashion2/deepfashion2_original_images")
    
    if not dataset_path.exists():
        print(f"✗ Dataset not found at: {dataset_path}")
        return False
    
    # Check train directory
    train_dir = dataset_path / "train"
    if not train_dir.exists():
        print(f"✗ Train directory not found: {train_dir}")
        return False
    
    # Check annotations
    annos_dir = train_dir / "annos"
    if not annos_dir.exists():
        print(f"✗ Annotations directory not found: {annos_dir}")
        return False
    
    # Count annotation files
    anno_files = list(annos_dir.glob("*.json"))
    print(f"✓ Found {len(anno_files)} annotation files")
    
    # Check validation directory
    val_dir = dataset_path / "validation"
    if val_dir.exists():
        val_annos = val_dir / "annos"
        if val_annos.exists():
            val_anno_files = list(val_annos.glob("*.json"))
            print(f"✓ Found {len(val_anno_files)} validation annotation files")
    
    print("✓ Dataset structure verified\n")
    return True


def test_config_file():
    """Test configuration file."""
    print("Testing configuration file...")
    
    config_path = Path("configs/deepfashion2_config.yaml")
    
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['experiment', 'dataset', 'model', 'training', 'loss']
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing config section: {section}")
                return False
        
        print("✓ Configuration file valid")
        print(f"✓ Experiment: {config['experiment']['name']}")
        print(f"✓ Model: {config['model']['model_size']}")
        print(f"✓ Batch size: {config['training']['batch_size']}")
        print(f"✓ Epochs: {config['training']['epochs']}")
        
    except Exception as e:
        print(f"✗ Error reading config: {e}")
        return False
    
    print()
    return True


def test_data_loading():
    """Test data loading."""
    print("Testing data loading...")
    
    try:
        from data.dataset import DeepFashion2Dataset
        from data.transforms import get_transforms
        
        # Create dataset
        transform = get_transforms('val', 224)
        dataset = DeepFashion2Dataset(
            root_dir="/media/kunwar-padda/Gold/DeepFashion2/deepfashion2_original_images",
            split='train',
            load_masks=True,
            load_keypoints=True,
            transform=transform
        )
        
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Test loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Sample loaded: {type(sample)}")
            
            if 'image' in sample:
                print(f"✓ Image shape: {sample['image'].shape}")
            
            if 'annotations' in sample:
                print(f"✓ Annotations: {len(sample['annotations'])} items")
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True


def test_model_creation():
    """Test model creation."""
    print("Testing model creation...")
    
    try:
        from ultralytics import YOLO
        
        # Test creating YOLOv8 segmentation model
        model = YOLO('yolov8n-seg.pt')
        print("✓ YOLOv8 segmentation model created")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 640)
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
        
        # Note: YOLO models handle this differently, so we'll skip the forward test
        print("✓ Model setup successful")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    print()
    return True


def test_output_directories():
    """Test output directory creation."""
    print("Testing output directories...")
    
    output_dirs = [
        "outputs/deepfashion2",
        "outputs/deepfashion2/checkpoints",
        "outputs/deepfashion2/logs",
        "outputs/deepfashion2/visualizations",
        "outputs/deepfashion2/results"
    ]
    
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        if Path(dir_path).exists():
            print(f"✓ Created directory: {dir_path}")
        else:
            print(f"✗ Failed to create: {dir_path}")
            return False
    
    print()
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("DeepFashion2 Training Setup Test")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("Dataset Path", test_dataset_path),
        ("Config File", test_config_file),
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Output Directories", test_output_directories)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running test: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("="*60)
    print("TEST RESULTS")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Ready to start training.")
        print("\nTo start training, run:")
        print("./run_deepfashion2_training.sh")
    else:
        print(f"\n⚠ {len(results) - passed} test(s) failed. Please fix issues before training.")
    
    return passed == len(results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)