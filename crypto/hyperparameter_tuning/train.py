#!/usr/bin/env python3
"""
W&B Sweep Entry Point for Crypto Forecasting
This script serves as the entry point for hyperparameter optimization sweeps.
Based on https://docs.wandb.ai/guides/sweeps/
"""

import os
import sys
import subprocess
import wandb
import argparse
from pathlib import Path

def validate_config(config):
    """Validate sweep configuration and apply model-specific constraints"""
    
    # Handle Mamba model constraints
    if config.model == "Mamba":
        if config.d_ff > 256:
            print(f"⚠️  Mamba model detected - capping d_ff at 256 (was {config.d_ff})")
            config.d_ff = 256
    
    # Validate parameter ranges
    if config.batch_size > 64:
        print(f"⚠️  Large batch size detected ({config.batch_size}), may cause memory issues")
    
    if config.d_ff > 1024:
        print(f"⚠️  Large d_ff detected ({config.d_ff}), may cause memory issues")
    
    # Ensure GPU is available
    if not config.use_gpu:
        raise ValueError("GPU is required for this sweep")
    
    return config

def build_command_args(config):
    """Build command line arguments for the training script"""
    
    # Base command
    cmd_args = [
        sys.executable, "crypto/run_crypto_transformer.py",
    ]
    
    # Required parameters
    required_params = {
        "task_name": config.task_name,
        "is_training": config.is_training,
        "model_id": config.model_id,
        "model": config.model,
        "data": config.data,
        "root_path": config.root_path,
        "data_path": config.data_path,
        "features": config.features,
        "target": config.target,
        "freq": config.freq,
        "checkpoints": config.checkpoints,
        "seq_len": config.seq_len,
        "label_len": config.label_len,
        "pred_len": config.pred_len,
        "enc_in": config.enc_in,
        "dec_in": config.dec_in,
        "c_out": config.c_out,
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "e_layers": config.e_layers,
        "d_layers": config.d_layers,
        "d_ff": config.d_ff,
        "moving_avg": config.moving_avg,
        "factor": config.factor,
        "dropout": config.dropout,
        "embed": config.embed,
        "activation": config.activation,
        "num_workers": config.num_workers,
        "itr": config.itr,
        "train_epochs": config.train_epochs,
        "batch_size": config.batch_size,
        "patience": config.patience,
        "learning_rate": config.learning_rate,
        "loss": config.loss,
        "lradj": config.lradj,
        "use_amp": str(config.use_amp).lower(),
        "inverse": str(config.inverse).lower(),
        "use_dtw": str(config.use_dtw).lower(),
        "use_wandb": str(config.use_wandb).lower(),
        "wandb_project": config.wandb_project,
        "wandb_entity": config.wandb_entity,
        "use_gpu": str(config.use_gpu).lower(),
        "device": config.device,
        "use_multi_gpu": str(config.use_multi_gpu).lower(),
        "devices": config.devices,
    }
    
    # Add required parameters
    for key, value in required_params.items():
        cmd_args.extend([f"--{key}", str(value)])
    
    # Add optional parameters if they exist
    optional_params = ["expand", "d_conv", "warmup_steps", "grad_clip"]
    for param in optional_params:
        if hasattr(config, param):
            cmd_args.extend([f"--{param}", str(getattr(config, param))])
    
    # Add boolean flags
    if config.distil:
        cmd_args.append("--distil")
    
    # Add descriptive name for this run
    run_name = f"sweep_{config.model}_{config.d_model}_{config.n_heads}_{config.e_layers}"
    cmd_args.extend(["--des", run_name])
    
    return cmd_args

def main():
    """Main entry point for sweep runs"""
    
    # Initialize wandb run
    wandb.init()
    config = wandb.config
    
    print(f"🚀 Starting sweep run for {config.model}")
    print(f"📊 Configuration: d_model={config.d_model}, n_heads={config.n_heads}, e_layers={config.e_layers}")
    print(f"🎯 Optimizing for: {wandb.sweep.config.metric.name}")
    
    # Validate and apply constraints
    try:
        config = validate_config(config)
    except ValueError as e:
        print(f"❌ Configuration validation failed: {e}")
        wandb.finish(exit_code=1)
        return
    
    # Build command arguments
    cmd_args = build_command_args(config)
    
    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = config.devices
    
    print(f"🔧 Running command: {' '.join(cmd_args[:10])}...")
    print(f"🖥️  GPU device: {config.device}")
    print(f"🧠 Model: {config.model} (d_ff={config.d_ff})")
    
    # Run the training script
    try:
        result = subprocess.run(
            cmd_args, 
            env=env, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        print("✅ Training completed successfully")
        if result.stdout:
            print("📤 STDOUT:", result.stdout[-500:])  # Last 500 chars
        
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out after 1 hour")
        wandb.finish(exit_code=1)
        return
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        print("📤 STDOUT:", e.stdout[-500:] if e.stdout else "No output")
        print("📥 STDERR:", e.stderr[-500:] if e.stderr else "No errors")
        
        # Log error to wandb
        wandb.log({"error": True, "error_message": str(e.stderr)})
        wandb.finish(exit_code=1)
        return
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        wandb.log({"error": True, "error_message": str(e)})
        wandb.finish(exit_code=1)
        return
    
    print("🎉 Sweep run completed successfully!")
    wandb.finish()

if __name__ == "__main__":
    main() 