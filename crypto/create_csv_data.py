#!/usr/bin/env python3
"""
Convert pickle data to CSV format for Time-Series-Library
"""

import pandas as pd
import os

# Load pickle data
print("Loading pickle data...")
ts_train = pd.read_pickle('./crypto/dataset/ts_train.pkl')
ts_val = pd.read_pickle('./crypto/dataset/ts_val.pkl')
ts_test = pd.read_pickle('./crypto/dataset/ts_test.pkl')

# Combine all data
print("Combining data...")
all_data = pd.concat([ts_train, ts_val, ts_test], ignore_index=True)

# Add date column
all_data['date'] = pd.date_range(start='2020-01-01', periods=len(all_data), freq='h')

# Reorder columns: date first, then features, then target
cols = list(all_data.columns)
cols.remove('date')
cols.remove('label')
all_data = all_data[['date'] + cols + ['label']]

# Save as CSV
output_path = './crypto/dataset/crypto.csv'
print(f"Saving to {output_path}...")
all_data.to_csv(output_path, index=False)

print(f"✅ CSV file created: {output_path}")
print(f"   Shape: {all_data.shape}")
print(f"   Columns: {list(all_data.columns)}") 