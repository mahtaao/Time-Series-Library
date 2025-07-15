#!/usr/bin/env python3
"""
Custom experiment class for crypto forecasting that uses our custom data provider
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

class CryptoExp_Long_Term_Forecast(Exp_Long_Term_Forecast):
    """
    Custom experiment class for crypto forecasting that uses our custom data provider
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        # Initialize W&B if requested and available
        self.use_wandb = getattr(args, 'use_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb and not wandb.run:
            wandb.init(
                project=getattr(args, 'wandb_project', 'crypto-forecasting'),
                entity=getattr(args, 'wandb_entity', None),
                name=getattr(args, 'wandb_run_name', None),
                config=vars(args)
            )
    
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
    
    def train(self, setting):
        """Override train method to add W&B logging"""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

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
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # Log to W&B
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": vali_loss,
                    "test/loss": test_loss,
                    "train/learning_rate": model_optim.param_groups[0]['lr']
                })
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model 