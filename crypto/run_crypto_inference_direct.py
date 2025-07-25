#!/usr/bin/env python3
"""
Generic crypto inference script that can work with any model checkpoint
"""

import os
import sys
import torch
import random
import numpy as np
import argparse
import pandas as pd
import re
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crypto.crypto_exp import CryptoExp_Long_Term_Forecast
from utils.print_args import print_args
from utils.metrics import metric

def parse_checkpoint_name(checkpoint_name):
    """
    Parse checkpoint name to extract model parameters
    Format: {MODEL}_job{JOB_ID}
    """
    print(f"Parsing checkpoint name: {checkpoint_name}")
    
    # Extract model name (before _job)
    model_match = re.search(r'^([^_]*)_job', checkpoint_name)
    model = model_match.group(1) if model_match else 'iTransformer'
    
    # Extract job ID (after job)
    job_match = re.search(r'job(\d+)', checkpoint_name)
    job_id = job_match.group(1) if job_match else '0'
    
    # For simplified checkpoint names, we use default parameters
    # These should match the training configuration
    seq_len = 96
    label_len = 48
    pred_len = 24
    d_model = 256
    n_heads = 8
    e_layers = 2
    d_layers = 1
    d_ff = 256
    factor = 1
    embed = 'timeF'
    distil = True
    
    # Model-specific parameter validation
    if model == 'Mamba':
        print("Mamba model detected - capping d_ff at 256 for CUDA compatibility")
        d_ff = 256
    
    return {
        'model': model,
        'seq_len': seq_len,
        'label_len': label_len,
        'pred_len': pred_len,
        'd_model': d_model,
        'n_heads': n_heads,
        'e_layers': e_layers,
        'd_layers': d_layers,
        'd_ff': d_ff,
        'factor': factor,
        'embed': embed,
        'distil': distil == 'True'
    }

def main():
    # Set random seeds for reproducibility
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    print("Starting generic crypto inference...")
    print(f"Timestamp: {datetime.now()}")
    
    # Checkpoint configuration - CHANGE THIS TO USE DIFFERENT CHECKPOINTS
    checkpoint_name = "iTransformer_job46505243"
    checkpoint_path = f"./checkpoints/{checkpoint_name}"
    
    print(f"Loading checkpoint: {checkpoint_name}")
    print(f"Checkpoint path: {checkpoint_path}")
    
    # Verify checkpoint exists
    if not os.path.exists(os.path.join(checkpoint_path, 'checkpoint.pth')):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}/checkpoint.pth")
        return
    
    print("Checkpoint found successfully!")
    
    # Parse checkpoint name to get model parameters
    params = parse_checkpoint_name(checkpoint_name)
    
    print("Extracted parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Create test arguments
    test_args = argparse.Namespace()
    
    # Basic config
    test_args.task_name = 'long_term_forecast'
    test_args.is_training = 0  # Testing mode
    test_args.model_id = 'crypto_test'
    test_args.model = params['model']
    test_args.data = 'custom'
    
    # Data loader
    test_args.root_path = './crypto/dataset/'
    test_args.data_path = 'crypto.csv'
    test_args.features = 'M'
    test_args.target = 'label'
    test_args.freq = 'h'
    test_args.checkpoints = './checkpoints/'
    
    # Forecasting task
    test_args.seq_len = params['seq_len']
    test_args.label_len = params['label_len']
    test_args.pred_len = params['pred_len']
    
    # Model define
    test_args.enc_in = 24
    test_args.dec_in = 24
    test_args.c_out = 1
    test_args.d_model = params['d_model']
    test_args.n_heads = params['n_heads']
    test_args.e_layers = params['e_layers']
    test_args.d_layers = params['d_layers']
    test_args.d_ff = params['d_ff']
    test_args.moving_avg = 25
    test_args.factor = params['factor']
    test_args.distil = params['distil']
    test_args.dropout = 0.1
    test_args.embed = params['embed']
    test_args.activation = 'gelu'
    
    # Optimization
    test_args.num_workers = 4
    test_args.itr = 1
    test_args.batch_size = 32
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
        print('Using CPU for inference')
        test_args.device = torch.device('cpu')
    
    print('Model configuration:')
    print_args(test_args)
    
    # Create experiment
    print(f'\n>>>>>>>Loading model from checkpoint: {checkpoint_name}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    
    try:
        exp = CryptoExp_Long_Term_Forecast(test_args)
        
        # Test the model (this loads the checkpoint and runs inference)
        preds, trues = exp.test(checkpoint_name, test=1)  # test=1 means load checkpoint
        
        print(f'Inference completed!')
        print(f'Predictions shape: {preds.shape}')
        print(f'Ground truth shape: {trues.shape}')
        
        # Calculate metrics
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        print(f'\nTest Results:')
        print(f'  MSE: {mse:.6f}')
        print(f'  MAE: {mae:.6f}')
        print(f'  RMSE: {rmse:.6f}')
        print(f'  MAPE: {mape:.6f}')
        print(f'  MSPE: {mspe:.6f}')
        
        # Save results
        print(f'\nSaving results...')
        
        # Create results directory
        results_dir = f'./inference_results/{checkpoint_name}/'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save predictions and ground truth as numpy arrays
        np.save(os.path.join(results_dir, 'predictions.npy'), preds)
        np.save(os.path.join(results_dir, 'ground_truth.npy'), trues)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame({
            'metric': ['MSE', 'MAE', 'RMSE', 'MAPE', 'MSPE'],
            'value': [mse, mae, rmse, mape, mspe]
        })
        
        summary_df.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)
        
        # Save predictions in submission format
        test_predictions = preds[:, :, -1]  # Take the last feature (target variable)
        
        print(f'Test predictions shape: {test_predictions.shape}')
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'id': range(len(test_predictions)),
            'prediction': test_predictions.flatten()  # Flatten to 1D array
        })
        
        submission_file = os.path.join(results_dir, f'{params["model"].lower()}_submission.csv')
        submission.to_csv(submission_file, index=False)
        print(f'Submission saved to: {submission_file}')
        
        # Also save in the same format as the original XGBoost
        try:
            sample_sub = pd.read_csv('./data/drw-crypto-market-prediction/sample_submission.csv')
            submission_formatted = pd.DataFrame({
                sample_sub.columns[0]: sample_sub.iloc[:len(test_predictions), 0],
                'prediction': test_predictions.flatten()
            })
            submission_formatted.to_csv(os.path.join(results_dir, f'{params["model"].lower()}_submission_formatted.csv'), index=False)
            formatted_file = f"{params['model'].lower()}_submission_formatted.csv"
            print(f'Formatted submission saved to: {os.path.join(results_dir, formatted_file)}')
        except Exception as e:
            print(f"Warning: Could not create formatted submission: {e}")
        
        print(f'\nAll results saved to: {results_dir}')
        print(f'{params["model"]} inference completed successfully!')
        
    except Exception as e:
        print(f"ERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Job finished at: {datetime.now()}")

if __name__ == '__main__':
    main() 