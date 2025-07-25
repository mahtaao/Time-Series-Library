#!/usr/bin/env python3
"""
Quick test script to verify W&B setup for crypto forecasting
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_wandb_import():
    """Test if W&B can be imported"""
    try:
        import wandb
        print(f"✅ W&B imported successfully (version: {wandb.__version__})")
        return True
    except ImportError as e:
        print(f"❌ W&B import failed: {e}")
        return False

def test_wandb_login():
    """Test W&B login"""
    try:
        import wandb
        # This will prompt for login if not already logged in
        api = wandb.Api()
        print("✅ W&B API connection successful")
        return True
    except Exception as e:
        print(f"❌ W&B login failed: {e}")
        print("Please run: wandb login")
        return False

def simulate_training_metrics():
    """Simulate what training metrics would look like"""
    print("\n📊 Simulating training metrics...")
    
    # Simulate training data
    epochs = 10
    train_losses = [1.0 - 0.08*i + np.random.normal(0, 0.02) for i in range(epochs)]
    val_losses = [0.95 - 0.06*i + np.random.normal(0, 0.03) for i in range(epochs)]
    learning_rates = [0.001 * (0.9**i) for i in range(epochs)]
    
    print(f"  Epochs: {epochs}")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final val loss: {val_losses[-1]:.4f}")
    print(f"  Final learning rate: {learning_rates[-1]:.6f}")
    
    return train_losses, val_losses, learning_rates

def simulate_test_results():
    """Simulate what test results would look like"""
    print("\n🔍 Simulating test results...")
    
    # Simulate predictions vs ground truth
    n_samples = 100
    pred_len = 24
    
    # Generate realistic crypto price predictions
    np.random.seed(42)
    base_price = 50000  # Bitcoin-like price
    preds = np.array([base_price + np.random.normal(0, 1000) + np.cumsum(np.random.normal(0, 50, pred_len)) 
                     for _ in range(n_samples)])
    trues = preds + np.random.normal(0, 200, (n_samples, pred_len))
    
    # Calculate metrics
    mae = np.mean(np.abs(preds - trues))
    mse = np.mean((preds - trues) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"  Samples: {n_samples}")
    print(f"  Prediction length: {pred_len}")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    
    return preds, trues, mae, mse, rmse

def create_sample_plots():
    """Create sample plots that would be logged to W&B"""
    print("\n📈 Creating sample plots...")
    
    # Create sample prediction plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss plot
    epochs = range(10)
    train_loss = [1.0 - 0.08*i + np.random.normal(0, 0.02) for i in epochs]
    val_loss = [0.95 - 0.06*i + np.random.normal(0, 0.03) for i in epochs]
    
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sample predictions
    time_steps = range(24)
    sample_pred = [50000 + i*50 + np.random.normal(0, 100) for i in time_steps]
    sample_true = [50000 + i*45 + np.random.normal(0, 80) for i in time_steps]
    
    axes[0, 1].plot(time_steps, sample_pred, 'b-', label='Prediction', linewidth=2)
    axes[0, 1].plot(time_steps, sample_true, 'r-', label='Ground Truth', linewidth=2)
    axes[0, 1].set_title('Sample Prediction')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Price')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate decay
    lr_decay = [0.001 * (0.9**i) for i in epochs]
    axes[1, 0].plot(epochs, lr_decay, 'g-', linewidth=2)
    axes[1, 0].set_title('Learning Rate Decay')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Metrics comparison
    models = ['Transformer', 'TimesNet', 'iTransformer', 'DLinear']
    mae_scores = [764.43, 712.18, 789.25, 823.67]  # Sample MAE values
    
    axes[1, 1].bar(models, mae_scores, color=['blue', 'green', 'orange', 'red'])
    axes[1, 1].set_title('Model Comparison (MAE)')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('test_plots', exist_ok=True)
    plot_path = 'test_plots/sample_wandb_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Sample plots saved: {plot_path}")
    return plot_path

def main():
    """Run all tests"""
    print("🧪 Testing W&B Setup for Crypto Forecasting")
    print("=" * 50)
    
    # Test W&B import
    if not test_wandb_import():
        print("\n❌ Please install W&B: pip install wandb")
        return
    
    # Test W&B login
    if not test_wandb_login():
        print("\n❌ Please login to W&B: wandb login")
        return
    
    # Simulate metrics
    train_losses, val_losses, learning_rates = simulate_training_metrics()
    
    # Simulate test results
    preds, trues, mae, mse, rmse = simulate_test_results()
    
    # Create sample plots
    plot_path = create_sample_plots()
    
    print("\n" + "=" * 50)
    print("✅ W&B Setup Test Complete!")
    print("=" * 50)
    
    print("\n📋 What will be logged to W&B:")
    print("  🎯 Training Metrics:")
    print("    - Train/validation loss curves")
    print("    - Learning rate decay")
    print("    - Training speed (iterations/second)")
    print("    - GPU utilization")
    
    print("\n  📊 Test Results:")
    print("    - MSE, MAE, RMSE, MAPE, MSPE")
    print("    - Prediction vs ground truth plots")
    print("    - Interactive tables with sample predictions")
    print("    - Model comparison across runs")
    
    print("\n  🔧 Configuration:")
    print("    - All hyperparameters")
    print("    - Model architecture details")
    print("    - Data preprocessing settings")
    
    print("\n🚀 Next steps:")
    print("1. Update W&B username in batch scripts")
    print("2. Run training: sbatch crypto/run_crypto_transformer.sbatch")
    print("3. Test checkpoint: sbatch crypto/test_crypto_checkpoint.sbatch")
    print("4. View results at: https://wandb.ai/your-username/crypto-forecasting")

if __name__ == '__main__':
    main() 