#!/usr/bin/env python3
"""
Improved experiment class for crypto forecasting with proper Mamba model handling
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to path to import from Time-Series-Library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from crypto.crypto_data_factory import crypto_data_provider
from utils.tools import EarlyStopping, adjust_learning_rate

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

class CryptoExp_Long_Term_Forecast_Improved(Exp_Long_Term_Forecast):
    """
    Improved experiment class for crypto forecasting with proper model validation
    """
    
    def __init__(self, args):
        # Validate model-specific parameters before building the model
        self._validate_model_parameters(args)
        
        super().__init__(args)
        
        # Initialize W&B if requested and available
        self.use_wandb = getattr(args, 'use_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb and not wandb.run:
            wandb.init(
                project=getattr(args, 'wandb_project', 'crypto-forecasting'),
                entity=getattr(args, 'wandb_entity', 'mahta-milaquebec'),
                name=getattr(args, 'wandb_run_name', None),
                config=vars(args)
            )
        
        # Logging configuration
        self.log_interval = getattr(args, 'log_interval', 10)  # Log every 10 epochs
        self.log_predictions = getattr(args, 'log_predictions', True)  # Log sample predictions
        self.max_pred_samples = getattr(args, 'max_pred_samples', 5)  # Max samples to log
        
        # Training history for detailed logging
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'lr': [],
            'grad_norm': [],
            'epoch_time': []
        }
    
    def _validate_model_parameters(self, args):
        """Validate model-specific parameters to prevent runtime errors"""
        
        # Mamba model validation
        if args.model == 'Mamba':
            # Check if mamba_ssm is available
            try:
                import mamba_ssm
            except ImportError:
                raise ImportError("Mamba model requires mamba_ssm package. Install with: pip install mamba-ssm")
            
            # Validate d_ff parameter for Mamba (d_state limitation)
            if hasattr(args, 'd_ff') and args.d_ff > 256:
                print(f"Warning: Mamba model has d_state <= 256 limitation. Capping d_ff from {args.d_ff} to 256")
                args.d_ff = 256
            
            # Validate required Mamba parameters
            if not hasattr(args, 'expand'):
                print("Warning: Mamba model requires 'expand' parameter. Setting default value of 2")
                args.expand = 2
            
            if not hasattr(args, 'd_conv'):
                print("Warning: Mamba model requires 'd_conv' parameter. Setting default value of 4")
                args.d_conv = 4
        
        # General parameter validation
        if hasattr(args, 'd_ff') and args.d_ff > 1024:
            print(f"Warning: Large d_ff value ({args.d_ff}) may cause memory issues. Consider reducing.")
        
        if hasattr(args, 'train_epochs') and args.train_epochs > 200:
            print(f"Warning: Large number of epochs ({args.train_epochs}) may take very long to train.")
        
        if hasattr(args, 'patience') and args.patience > 50:
            print(f"Warning: Large patience value ({args.patience}) may prevent early stopping.")
    
    def _get_data(self, flag):
        """
        Override the data loading method to use our custom crypto data provider
        """
        if flag == 'test':
            shuffle_flag = False
            drop_last = False
            batch_size = self.args.batch_size
            freq = self.args.freq
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = self.args.batch_size
            freq = self.args.freq
        
        # Use our custom crypto data provider
        data_set, data_loader = crypto_data_provider(self.args, flag)
        
        return data_set, data_loader
    
    def _calc_grad_norm(self, model):
        """Calculate the L2 norm of gradients"""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)
    
    def _log_predictions(self, train_loader, epoch):
        """Log sample predictions to W&B using W&B's native plotting"""
        if not self.use_wandb or not self.log_predictions:
            return
        
        self.model.eval()
        sample_data = None
        sample_pred = None
        sample_true = None
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if i >= self.max_pred_samples:
                    break
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Get predictions
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                if sample_data is None:
                    sample_data = batch_x.cpu().numpy()
                    sample_pred = outputs.cpu().numpy()
                    sample_true = batch_y.cpu().numpy()
                else:
                    sample_data = np.concatenate([sample_data, batch_x.cpu().numpy()], axis=0)
                    sample_pred = np.concatenate([sample_pred, outputs.cpu().numpy()], axis=0)
                    sample_true = np.concatenate([sample_true, batch_y.cpu().numpy()], axis=0)
        
        self.model.train()
        
        # Get number of features
        num_features = sample_pred.shape[-1]
        
        # Log sample predictions for each feature
        for i in range(min(3, sample_pred.shape[0])):
            # Log predictions for each feature
            for feature_idx in range(num_features):
                # Get sequences for this feature
                input_seq = sample_data[i, -self.args.pred_len:, feature_idx]
                pred_seq = sample_pred[i, :, feature_idx]
                true_seq = sample_true[i, :, feature_idx]
                
                # Calculate prediction metrics
                mse = np.mean((pred_seq - true_seq) ** 2)
                mae = np.mean(np.abs(pred_seq - true_seq))
                mape = np.mean(np.abs((true_seq - pred_seq) / (true_seq + 1e-8))) * 100
                
                # Log prediction metrics
                wandb.log({
                    f"predictions/rmse/sample_{i+1}_feature_{feature_idx}_e{epoch}": mse,
                    f"predictions/mae/sample_{i+1}_feature_{feature_idx}_e{epoch}": mae,
                    f"predictions/mape/sample_{i+1}_feature_{feature_idx}_e{epoch}": mape,
                }, step=epoch + 1)
                
                # Create time series data for visualization
                time_steps = list(range(len(pred_seq)))
                
                # Log time series plot showing prediction vs ground truth
                wandb.log({
                    f"predictions/time_series/sample_{i+1}_feature_{feature_idx}_e{epoch}": wandb.plot.line_series(
                        xs=time_steps,
                        ys=[pred_seq, true_seq],
                        keys=["Prediction", "Ground Truth"],
                        title=f"Sample {i+1} Feature {feature_idx} Predictions vs Ground Truth - Epoch {epoch+1}",
                        xname="Time Step"
                    )
                }, step=epoch + 1)
    
    def _log_metrics(self, epoch, train_loss, val_loss, test_loss, optimizer, epoch_time, grad_norm):
        """Log detailed metrics to W&B"""
        if not self.use_wandb:
            return
        
        # Store in history
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['test_loss'].append(test_loss)
        self.history['lr'].append(optimizer.param_groups[0]['lr'])
        self.history['grad_norm'].append(grad_norm)
        self.history['epoch_time'].append(epoch_time)
        
        # Calculate additional metrics
        if len(self.history['train_loss']) > 1:
            train_loss_delta = train_loss - self.history['train_loss'][-2]
            val_loss_delta = val_loss - self.history['val_loss'][-2]
            overfit_ratio = val_loss / train_loss if train_loss > 0 else 0
        else:
            train_loss_delta = 0
            val_loss_delta = 0
            overfit_ratio = 0
        
        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "loss/train": train_loss,
            "loss/val": val_loss,
            "loss/test": test_loss,
            "train/lr": optimizer.param_groups[0]['lr'],
            "train/grad_norm": grad_norm,
            "train/epoch_time": epoch_time,
            "train/loss_delta": train_loss_delta,
            "val/loss_delta": val_loss_delta,
            "train/overfit_ratio": overfit_ratio,
            "loss/train_avg_5": np.mean(self.history['train_loss'][-5:]) if len(self.history['train_loss']) >= 5 else train_loss,
            "loss/val_avg_5": np.mean(self.history['val_loss'][-5:]) if len(self.history['val_loss']) >= 5 else val_loss,
        })
        
        # Log training curves using organized sections
        wandb.log({
            # Training Losses
            "training/loss/train": train_loss,
            "training/loss/val": val_loss,
            "training/loss/test": test_loss,
            
            # Training Metrics
            "training/learning_rate": optimizer.param_groups[0]['lr'],
            "training/gradient_norm": grad_norm,
            "training/epoch_time": epoch_time,
            "training/val_train_ratio": val_loss / train_loss if train_loss > 0 else 0,
            "training/overfitting_score": (val_loss - train_loss) / train_loss if train_loss > 0 else 0,
        }, step=epoch + 1)  # Use epoch number as step for clarity
    
    def train(self, setting):
        """Override train method to add detailed W&B logging with less frequency"""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_grad_norms = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # Calculate gradient norm before stepping
                    grad_norm = self._calc_grad_norm(self.model)
                    epoch_grad_norms.append(grad_norm)
                    optimizer.step()

            epoch_time = time.time() - epoch_time
            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # Log metrics to W&B every epoch
            if self.use_wandb:
                self._log_metrics(epoch, train_loss, vali_loss, test_loss, optimizer, epoch_time, avg_grad_norm)
                
                # Log sample predictions every 20 epochs or first few epochs
                if (epoch + 1) % 20 == 0 or epoch < 3:
                    self._log_predictions(train_loader, epoch + 1)
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model 