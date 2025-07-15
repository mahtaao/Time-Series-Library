#!/usr/bin/env python3
"""
Test script for crypto forecasting checkpoints with W&B integration
"""

import os
import sys
import torch
import random
import numpy as np
import argparse
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto.crypto_exp import CryptoExp_Long_Term_Forecast
from utils.print_args import print_args
from utils.metrics import metric

def plot_predictions(preds, trues, save_path, title="Predictions vs Ground Truth"):
    """Create and save prediction plots"""
    plt.figure(figsize=(15, 8))
    
    # Plot first few samples
    n_samples = min(5, preds.shape[0])
    for i in range(n_samples):
        plt.subplot(n_samples, 1, i+1)
        plt.plot(trues[i, :, -1], label='Ground Truth', linewidth=2)
        plt.plot(preds[i, :, -1], label='Prediction', linewidth=2)
        plt.title(f'Sample {i+1}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def create_wandb_table(preds, trues, sample_indices=None):
    """Create W&B table for detailed analysis"""
    if sample_indices is None:
        sample_indices = range(min(10, preds.shape[0]))
    
    table_data = []
    for i in sample_indices:
        # Calculate metrics for this sample
        sample_pred = preds[i, :, -1]  # Last feature (target)
        sample_true = trues[i, :, -1]
        
        mae = np.mean(np.abs(sample_pred - sample_true))
        mse = np.mean((sample_pred - sample_true) ** 2)
        rmse = np.sqrt(mse)
        
        # Create time series data
        time_steps = list(range(len(sample_pred)))
        
        table_data.append({
            "sample_id": i,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "predictions": wandb.plot.line_series(
                xs=time_steps,
                ys=[sample_pred, sample_true],
                keys=["Prediction", "Ground Truth"],
                title=f"Sample {i} Predictions",
                xname="Time Step"
            )
        })
    
    return wandb.Table(data=table_data, columns=["sample_id", "mae", "mse", "rmse", "predictions"])

def main():
    parser = argparse.ArgumentParser(description='Test Crypto Forecasting Checkpoint')
    
    # Model and checkpoint
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to the checkpoint directory')
    parser.add_argument('--model', type=str, default='Transformer',
                       help='Model name (should match checkpoint)')
    
    # W&B settings
    parser.add_argument('--wandb_project', type=str, default='crypto-forecasting',
                       help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='W&B run name (auto-generated if not provided)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='W&B entity/username')
    
    # Testing parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--save_plots', action='store_true', default=True,
                       help='Save prediction plots')
    parser.add_argument('--plot_samples', type=int, default=5,
                       help='Number of samples to plot')
    
    args = parser.parse_args()
    
    # Initialize W&B
    if args.wandb_run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.wandb_run_name = f"test_{args.model}_{timestamp}"
    
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        config=vars(args)
    )
    
    print(f"Testing checkpoint: {args.checkpoint_path}")
    print(f"W&B run: {wandb.run.name}")
    
    # Create test arguments (matching training args)
    test_args = argparse.Namespace()
    test_args.task_name = 'long_term_forecast'
    test_args.is_training = 0  # Testing mode
    test_args.model_id = 'crypto_test'
    test_args.model = args.model
    test_args.data = 'custom'
    test_args.root_path = './crypto/dataset/'
    test_args.data_path = 'crypto.csv'
    test_args.features = 'M'
    test_args.target = 'label'
    test_args.freq = 'h'
    test_args.checkpoints = './checkpoints/'
    
    # Model parameters (should match training)
    test_args.seq_len = 96
    test_args.label_len = 48
    test_args.pred_len = 24
    test_args.enc_in = 24
    test_args.dec_in = 24
    test_args.c_out = 1
    test_args.d_model = 256
    test_args.n_heads = 8
    test_args.e_layers = 2
    test_args.d_layers = 1
    test_args.d_ff = 1024
    test_args.moving_avg = 25
    test_args.factor = 1
    test_args.distil = True
    test_args.dropout = 0.1
    test_args.embed = 'timeF'
    test_args.activation = 'gelu'
    
    # Optimization
    test_args.num_workers = 10
    test_args.itr = 1
    test_args.batch_size = args.batch_size
    test_args.learning_rate = 0.0001
    test_args.des = 'test'
    test_args.loss = 'MSE'
    test_args.lradj = 'type1'
    test_args.use_amp = False
    test_args.inverse = True
    test_args.use_dtw = False
    
    # GPU
    test_args.use_gpu = True
    test_args.gpu = 0
    test_args.use_multi_gpu = False
    test_args.devices = '0,1,2,3'
    
    # Check GPU availability
    if torch.cuda.is_available():
        test_args.device = torch.device('cuda:{}'.format(test_args.gpu))
        print('GPU available: {}'.format(torch.cuda.get_device_name(test_args.gpu)))
    else:
        print('ERROR: No GPU available!')
        sys.exit(1)
    
    print('Args in experiment:')
    print_args(test_args)
    
    # Create experiment
    Exp = CryptoExp_Long_Term_Forecast
    exp = Exp(test_args)
    
    # Extract setting from checkpoint path
    checkpoint_dir = os.path.basename(args.checkpoint_path.rstrip('/'))
    setting = checkpoint_dir
    
    print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    
    # Test the model
    preds, trues = exp.test(setting, test=1)  # test=1 means load checkpoint
    
    # Calculate metrics
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    
    print(f'Test Results:')
    print(f'  MSE: {mse:.6f}')
    print(f'  MAE: {mae:.6f}')
    print(f'  RMSE: {rmse:.6f}')
    print(f'  MAPE: {mape:.6f}')
    print(f'  MSPE: {mspe:.6f}')
    
    # Log metrics to W&B
    wandb.log({
        "test/mse": mse,
        "test/mae": mae,
        "test/rmse": rmse,
        "test/mape": mape,
        "test/mspe": mspe,
        "test/pred_shape": preds.shape,
        "test/true_shape": trues.shape
    })
    
    # Create and log prediction plots
    if args.save_plots:
        plots_dir = f'./test_plots/{setting}/'
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_path = os.path.join(plots_dir, 'predictions.png')
        plot_predictions(preds, trues, plot_path, f"{args.model} Predictions")
        
        # Log plot to W&B
        wandb.log({"predictions_plot": wandb.Image(plot_path)})
        print(f"Prediction plot saved: {plot_path}")
    
    # Create W&B table for detailed analysis
    wandb_table = create_wandb_table(preds, trues)
    wandb.log({"detailed_predictions": wandb_table})
    
    # Save predictions and ground truth as numpy arrays
    results_dir = f'./test_results/{setting}/'
    os.makedirs(results_dir, exist_ok=True)
    
    np.save(os.path.join(results_dir, 'predictions.npy'), preds)
    np.save(os.path.join(results_dir, 'ground_truth.npy'), trues)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'metric': ['MSE', 'MAE', 'RMSE', 'MAPE', 'MSPE'],
        'value': [mse, mae, rmse, mape, mspe]
    })
    
    summary_df.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)
    
    # Log summary table to W&B
    wandb.log({"metrics_summary": wandb.Table(dataframe=summary_df)})
    
    print(f"Results saved to: {results_dir}")
    print("Testing completed!")
    
    wandb.finish()

if __name__ == '__main__':
    main() 