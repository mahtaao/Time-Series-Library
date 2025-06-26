import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class H5EEGLoader(Dataset):
    """
    Dataset class for H5 EEG files.
    Loads EEG data from H5 files and prepares it for time series forecasting.
    """
    
    def __init__(self, args, root_path, flag='train', size=None,
                 features='M', data_path=None, target=None, scale=True, 
                 timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # init
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
        """Load and preprocess H5 EEG data."""
        self.scaler = StandardScaler()
        
        # Get all H5 files in the directory
        h5_files = [f for f in os.listdir(self.root_path) if f.endswith('.h5')]
        h5_files.sort()
        
        print(f"Found {len(h5_files)} H5 files in {self.root_path}")
        
        # Load all data
        all_data = []
        for h5_file in h5_files:
            file_path = os.path.join(self.root_path, h5_file)
            print(f"Loading {h5_file}...")
            
            with h5py.File(file_path, 'r') as f:
                # Load power_data which has shape (samples, 32, 32, 1001)
                data = f['power_data'][:]
                print(f"  Shape: {data.shape}")
                
                # Reshape to (samples, features, timepoints)
                # Flatten the 32x32 spatial dimensions into features
                samples, height, width, timepoints = data.shape
                data_reshaped = data.reshape(samples, height * width, timepoints)
                
                # Transpose to get (samples, timepoints, features)
                data_reshaped = data_reshaped.transpose(0, 2, 1)
                
                all_data.append(data_reshaped)
        
        # Concatenate all data
        self.data = np.concatenate(all_data, axis=0)
        print(f"Combined data shape: {self.data.shape}")
        
        # Split data into train/val/test
        num_samples = self.data.shape[0]
        num_train = int(num_samples * 0.7)
        num_test = int(num_samples * 0.2)
        num_vali = num_samples - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, num_samples - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, num_samples]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Select features
        if self.features == 'M' or self.features == 'MS':
            # Use all features (all EEG channels)
            data_x = self.data[border1:border2]
        elif self.features == 'S':
            # Use only target feature (if specified)
            if self.target is not None:
                target_idx = int(self.target)
                data_x = self.data[border1:border2, :, target_idx:target_idx+1]
            else:
                # Use first channel as default
                data_x = self.data[border1:border2, :, 0:1]

        # Scale the data
        if self.scale:
            train_data = self.data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.reshape(-1, train_data.shape[-1]))
            data_x = self.scaler.transform(data_x.reshape(-1, data_x.shape[-1])).reshape(data_x.shape)

        # Prepare time features (simple sequential indices for now)
        time_steps = data_x.shape[1]
        time_features = np.arange(time_steps).reshape(1, -1, 1)
        time_features = np.repeat(time_features, data_x.shape[0], axis=0)

        self.data_x = data_x
        self.data_y = data_x  # For forecasting, target is the same as input
        self.data_stamp = time_features

    def __getitem__(self, index):
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