#!/usr/bin/env python3
"""
Test script for crypto forecasting checkpoints - simplified for inference
"""

import os
import sys
import torch
import random
import numpy as np
import argparse
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

def main():
    parser = argparse.ArgumentParser(description='Test Crypto Forecasting Checkpoint')
    
    # Model and checkpoint
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to the checkpoint directory')
    parser.add_argument('--model', type=str, default='Transformer',
                       help='Model name (should match checkpoint)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--save_plots', action='store_true', default=False,
                       help='Save prediction plots')
    
    args = parser.parse_args()
    
    print(f"Testing checkpoint: {args.checkpoint_path}")
    
    # Create test arguments
    test_args = argparse.Namespace()
    test_args.task_name = 'long_term_forecast'
    test_args.is_training = 0
    test_args.model_id = 'crypto_test'
    test_args.model = args.model
    test_args.data = 'custom'
    test_args.root_path = './crypto/dataset/'
    test_args.data_path = 'crypto.csv'
    test_args.features = 'M'
    test_args.target = 'label'
    test_args.freq = 'h'
    test_args.checkpoints = './checkpoints/'
    test_args.use_gpu = True
    test_args.gpu = 0
    test_args.use_multi_gpu = False
    test_args.devices = '0,1,2,3'
    test_args.device = torch.device('cuda:{}'.format(test_args.gpu)) if torch.cuda.is_available() else torch.device('cpu')
    
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
    print(f'Metrics saved to: {os.path.join(results_dir, "metrics.csv")}')
    
    # Save predictions in DRW competition submission format
    print(f'\nSaving predictions in DRW competition format...')
    
    # preds shape is (n_samples, pred_len, n_features) - we want the last feature (target)
    test_predictions = preds[:, :, -1]  # Take the last feature (target variable)
    
    print(f'Test predictions shape: {test_predictions.shape}')
    
    # Create submission DataFrame in DRW format
    try:
        sample_sub = pd.read_csv('./data/drw-crypto-market-prediction/sample_submission.csv')
        submission = pd.DataFrame({
            sample_sub.columns[0]: sample_sub.iloc[:len(test_predictions), 0],
            'prediction': test_predictions.flatten()  # Flatten to 1D array
        })
    except FileNotFoundError:
        # If sample submission not found, create simple format
        submission = pd.DataFrame({
            'id': range(len(test_predictions.flatten())),
            'prediction': test_predictions.flatten()
        })
    
    submission_file = os.path.join(results_dir, f'{args.model.lower()}_submission.csv')
    submission.to_csv(submission_file, index=False)
    print(f'DRW submission saved to: {submission_file}')
    
    # Save plots if requested
    if args.save_plots:
        plot_file = os.path.join(results_dir, 'predictions_plot.png')
        plot_predictions(preds, trues, plot_file)
        print(f'Prediction plots saved to: {plot_file}')
    
    print(f'\nAll results saved to: {results_dir}')
    print(f'{args.model} testing completed successfully!')

if __name__ == '__main__':
    main() 