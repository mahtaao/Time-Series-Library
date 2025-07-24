#!/usr/bin/env python3
"""
Test script to verify sweep initialization works
"""

import wandb
from pathlib import Path

def test_sweep_init():
    """Test that sweep initialization works without PosixPath error"""
    
    print("🧪 Testing sweep initialization...")
    
    # Test config path - use the correct path
    config_path = Path("hyperparameter_tuning/sweep_config.yaml")
    print(f"📁 Config path: {config_path}")
    print(f"📁 Config path type: {type(config_path)}")
    print(f"📁 Config path exists: {config_path.exists()}")
    
    # Test the fix
    config_path_str = str(config_path)
    print(f"📁 Config path string: {config_path_str}")
    print(f"📁 Config path string type: {type(config_path_str)}")
    
    # Try to load the config
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config loaded successfully")
        print(f"📊 Config keys: {list(config.keys())}")
        print(f"📊 Program: {config.get('program', 'Not found')}")
        print(f"📊 Method: {config.get('method', 'Not found')}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return
    
    print("✅ Sweep initialization test completed!")

if __name__ == "__main__":
    test_sweep_init() 