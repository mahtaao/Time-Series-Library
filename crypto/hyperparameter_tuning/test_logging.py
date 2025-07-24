#!/usr/bin/env python3
"""
Test script to verify W&B logging fixes
"""

import wandb
import time

def test_logging():
    """Test that logging works without step monotonicity issues"""
    
    # Initialize wandb
    wandb.init(
        project="crypto-forecasting",
        entity="mahta-milaquebec",
        name="test_logging_fix",
        config={"test": True}
    )
    
    print("🧪 Testing W&B logging fixes...")
    
    # Simulate training epochs
    for epoch in range(5):
        print(f"📊 Logging epoch {epoch + 1}")
        
        # Simulate metrics
        train_loss = 1.0 - (epoch * 0.1)
        val_loss = 1.1 - (epoch * 0.08)
        test_loss = 1.2 - (epoch * 0.09)
        
        # Single consolidated log call (like the fix)
        wandb.log({
            "epoch": epoch + 1,
            "loss/train": train_loss,
            "loss/val": val_loss,
            "loss/test": test_loss,
            "training/loss/train": train_loss,
            "training/loss/val": val_loss,
            "training/loss/test": test_loss,
            "training/learning_rate": 0.001,
            "training/gradient_norm": 0.5,
            "training/epoch_time": 30.0,
        }, step=epoch + 1)
        
        time.sleep(1)  # Small delay
    
    print("✅ Logging test completed successfully!")
    wandb.finish()

if __name__ == "__main__":
    test_logging() 