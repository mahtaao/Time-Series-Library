#!/usr/bin/env python3
"""
Simple test script to verify the crypto transformer setup works
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_loading():
    """Test that we can load the preprocessed data"""
    print("🔍 Testing data loading...")
    
    try:
        # Load preprocessed data
        variables = pd.read_pickle('./crypto/data/preprocessed.pkl')
        
        print(f"✅ Data loaded successfully")
        print(f"   Train shape: {variables['ts_train'].shape}")
        print(f"   Val shape: {variables['ts_val'].shape}")
        print(f"   Test shape: {variables['ts_test'].shape}")
        
        # Check features
        features = [col for col in variables['ts_train'].columns if col != 'label']
        print(f"   Features: {len(features)}")
        print(f"   Target: label")
        
        return True, len(features)
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False, 0

def test_crypto_dataset():
    """Test our custom crypto dataset"""
    print("\n🔍 Testing CryptoDataset...")
    
    try:
        from crypto.run_crypto_transformer import CryptoDataset, create_crypto_args
        
        # Create args
        args = create_crypto_args()
        args.enc_in = 22
        args.dec_in = 22
        args.c_out = 1
        args.seq_len = 96
        args.label_len = 48
        args.pred_len = 24
        
        # Create dataset
        dataset = CryptoDataset(
            args=args,
            root_path='./crypto/data/',
            flag='train',
            size=[args.seq_len, args.label_len, args.pred_len],
            features='M',
            target='label'
        )
        
        print(f"✅ Dataset created successfully")
        print(f"   Dataset size: {len(dataset)}")
        
        # Test getting a sample
        sample = dataset[0]
        print(f"   Sample shapes:")
        print(f"     seq_x: {sample[0].shape}")
        print(f"     seq_y: {sample[1].shape}")
        print(f"     seq_x_mark: {sample[2].shape}")
        print(f"     seq_y_mark: {sample[3].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return False

def test_model_creation():
    """Test that we can create a transformer model"""
    print("\n🔍 Testing model creation...")
    
    try:
        from models import Transformer
        
        # Create config
        class Config:
            def __init__(self):
                self.enc_in = 22
                self.dec_in = 22
                self.c_out = 1
                self.seq_len = 96
                self.label_len = 48
                self.pred_len = 24
                self.d_model = 256
                self.n_heads = 8
                self.e_layers = 2
                self.d_layers = 1
                self.d_ff = 1024
                self.dropout = 0.1
                self.activation = 'gelu'
                self.factor = 5
                self.moving_avg = 25
                self.features = 'M'
                self.target = 'OT'
                self.freq = 'h'
                self.embed = 'timeF'
                self.distil = True
        
        config = Config()
        
        # Create model
        model = Transformer.Model(config)
        
        print(f"✅ Model created successfully")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, config.seq_len, config.enc_in)
        x_mark = torch.randn(batch_size, config.seq_len, 4)  # time features
        y = torch.randn(batch_size, config.label_len, config.dec_in)
        y_mark = torch.randn(batch_size, config.label_len, 4)  # time features
        
        with torch.no_grad():
            output = model(x, x_mark, y, y_mark)
        
        print(f"   Output shape: {output.shape}")
        print(f"   Expected shape: ({batch_size}, {config.pred_len}, {config.c_out})")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return False

def test_simple_training():
    """Test a simple training run"""
    print("\n🔍 Testing simple training...")
    
    try:
        # Import our custom experiment
        from crypto.crypto_exp import CryptoExp_Long_Term_Forecast
        from crypto.run_crypto_transformer import create_crypto_args
        
        # Create args for quick test
        args = create_crypto_args()
        args.model = 'Transformer'
        args.enc_in = 22
        args.dec_in = 22
        args.c_out = 1
        args.seq_len = 96
        args.label_len = 48
        args.pred_len = 24
        args.d_model = 128  # Smaller for quick test
        args.n_heads = 4
        args.e_layers = 1
        args.d_layers = 1
        args.d_ff = 512
        args.batch_size = 16
        args.train_epochs = 2  # Very short for testing
        args.learning_rate = 0.001
        args.patience = 1
        
        # Create experiment
        exp = CryptoExp_Long_Term_Forecast(args)
        
        print(f"✅ Experiment created successfully")
        
        # Test data loading
        train_data, train_loader = exp._get_data('train')
        val_data, val_loader = exp._get_data('val')
        
        print(f"   Train loader: {len(train_loader)} batches")
        print(f"   Val loader: {len(val_loader)} batches")
        
        # Test one training step
        exp.model.train()
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            
            # Forward pass
            outputs = exp.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            
            print(f"   Forward pass successful")
            print(f"   Input shape: {batch_x.shape}")
            print(f"   Output shape: {outputs.shape}")
            break
        
        return True
        
    except Exception as e:
        print(f"❌ Failed training test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Crypto Transformer Setup")
    print("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Dataset Creation", test_crypto_dataset),
        ("Model Creation", test_model_creation),
        ("Simple Training", test_simple_training),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            if test_name == "Data Loading":
                success, n_features = test_func()
                results[test_name] = success
                if success:
                    print(f"   Number of features: {n_features}")
            else:
                success = test_func()
                results[test_name] = success
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print(f"{'='*50}")
    
    all_passed = True
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} : {status}")
        if not success:
            all_passed = False
    
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Setup is ready! You can now run:")
        print("   python crypto/run_crypto_transformer.py --model Transformer")
        print("   python crypto/run_transformer_models.py")
    else:
        print("\n⚠️  Please fix the failing tests before proceeding")

if __name__ == '__main__':
    main() 