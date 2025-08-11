#!/usr/bin/env python3
"""
Quick script to examine crypto data structure without loading large files
"""

import pandas as pd
import os

def quick_data_check():
    """Quick check of data files and structure"""
    print("🔍 QUICK CRYPTO DATA CHECK")
    print("=" * 50)
    
    # Check file sizes
    data_dir = "crypto/dataset"
    files = [
        "preprocessed.pkl",
        "ts_train.pkl", 
        "ts_val.pkl",
        "ts_test.pkl",
        "train.parquet",
        "test.parquet",
        "sample_submission.csv"
    ]
    
    print("📁 File sizes:")
    for file in files:
        path = os.path.join(data_dir, file)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024*1024)
            print(f"  {file}: {size_mb:.1f} MB")
        else:
            print(f"  {file}: Not found")
    
    # Try to load just the sample submission (smallest file)
    print(f"\n📊 Sample submission data:")
    try:
        sample_df = pd.read_csv('crypto/dataset/sample_submission.csv')
        print(f"  Shape: {sample_df.shape}")
        print(f"  Columns: {list(sample_df.columns)}")
        print(f"  First few rows:")
        print(sample_df.head(3))
    except Exception as e:
        print(f"  Error: {e}")
    
    # Try to peek at pickle structure without loading full data
    print(f"\n🔍 Pickle file structure:")
    try:
        # Just get column names from train pickle
        ts_train = pd.read_pickle('crypto/dataset/ts_train.pkl')
        print(f"  ts_train shape: {ts_train.shape}")
        print(f"  Columns: {list(ts_train.columns)}")
        print(f"  Data types:")
        for col, dtype in ts_train.dtypes.items():
            print(f"    {col}: {dtype}")
        print(f"  Sample values (first 3 rows, first 5 cols):")
        print(ts_train.iloc[:3, :5])
    except Exception as e:
        print(f"  Error loading pickle: {e}")
    
    print(f"\n✅ Quick check complete!")

if __name__ == "__main__":
    quick_data_check() 