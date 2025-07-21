#!/usr/bin/env python3
"""
Custom data factory for crypto data that integrates with Time-Series-Library
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

# Add the parent directory to path to import from Time-Series-Library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CryptoDataset(Dataset):
    """Custom dataset class for crypto data that works with Time-Series-Library"""
    
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path='crypto_data.csv',
                 target='label', scale=True, timeenc=0, freq='h'):
        """
        Initialize crypto dataset
        
        Args:
            args: Arguments object
            root_path: Path to data directory
            flag: 'train', 'val', or 'test'
            size: [seq_len, label_len, pred_len]
            features: 'M' for multivariate, 'S' for univariate
            data_path: Path to CSV file
            target: Target column name
            scale: Whether to scale the data
            timeenc: Time encoding type
            freq: Frequency for time features
        """
        self.args = args
        
        # Set sequence parameters
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # Set dataset type
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        
        self.__read_data__()
    
    def __read_data__(self):
        """Read and preprocess the crypto data"""
        # Load preprocessed data
        print(f"Loading data from {os.path.join(self.root_path, self.data_path)}")
        
        # Load the preprocessed pickle files directly
        if self.set_type == 0:  # train
            df_raw = pd.read_pickle(os.path.join(self.root_path, 'ts_train.pkl'))
        elif self.set_type == 1:  # val
            df_raw = pd.read_pickle(os.path.join(self.root_path, 'ts_val.pkl'))
        else:  # test
            df_raw = pd.read_pickle(os.path.join(self.root_path, 'ts_test.pkl'))
        
        # Create a dummy date column for compatibility
        df_raw = df_raw.reset_index()
        df_raw['date'] = pd.date_range(start='2020-01-01', periods=len(df_raw), freq='h')
        
        # Ensure target column exists
        if self.target not in df_raw.columns:
            raise ValueError(f"Target column '{self.target}' not found in data")
        
        # Reorder columns: date, features, target
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        
        # Prepare data based on features type
        if self.features == 'M' or self.features == 'MS':
            # For multivariate, include all features including target (for autoregressive prediction)
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        
        # Scale the data
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # Create time features
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            from data_provider.data_loader import time_features
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp
        
        print(f"Dataset {self.set_type} loaded: {len(self.data_x)} samples, {self.data_x.shape[1]} features")
        print(f"Data X shape: {self.data_x.shape}, Data Y shape: {self.data_y.shape}, Time features shape: {self.data_stamp.shape}")
    
    def __getitem__(self, index):
        """Get a single sample"""
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def crypto_data_provider(args, flag):
    """
    Data provider for crypto forecasting that works with Time-Series-Library
    
    Args:
        args: Arguments object
        flag: 'train', 'val', or 'test'
    
    Returns:
        dataset, dataloader: Dataset and DataLoader objects
    """
    timeenc = 0 if args.embed != 'timeF' else 1
    
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq
    
    # Create crypto dataset
    data_set = CryptoDataset(
        args=args,
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True,
        timeenc=timeenc,
        freq=freq
    )
    
    print(f"{flag} dataset size: {len(data_set)}")
    
    # Create data loader
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    
    return data_set, data_loader 