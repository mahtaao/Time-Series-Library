#!/usr/bin/env python3
"""
Script to examine crypto data values from pickle and parquet files
"""

import pandas as pd
import numpy as np
import os

def examine_pickle_data():
    """Examine the processed pickle data"""
    print("🔍 EXAMINING PROCESSED PICKLE DATA")
    print("=" * 50)
    
    # Load the preprocessed data
    try:
        variables = pd.read_pickle('crypto/dataset/preprocessed.pkl')
        ts_train = variables['ts_train']
        ts_val = variables['ts_val'] 
        ts_test = variables['ts_test']
        
        print(f"✅ Successfully loaded preprocessed data")
        print(f"📊 Data shapes:")
        print(f"  ts_train: {ts_train.shape}")
        print(f"  ts_val:   {ts_val.shape}")
        print(f"  ts_test:  {ts_test.shape}")
        
        print(f"\n📋 Column names:")
        for i, col in enumerate(ts_train.columns):
            print(f"  {i+1:2d}. {col}")
            
        print(f"\n📈 Sample data (first 5 rows):")
        print(ts_train.head())
        
        print(f"\n📊 Data statistics:")
        print(ts_train.describe())
        
        return ts_train, ts_val, ts_test
        
    except Exception as e:
        print(f"❌ Error loading preprocessed data: {e}")
        return None, None, None

def examine_parquet_data():
    """Examine the original parquet data"""
    print("\n🔍 EXAMINING ORIGINAL PARQUET DATA")
    print("=" * 50)
    
    try:
        # Load train parquet (this might be large)
        print("📖 Loading train.parquet...")
        train_df = pd.read_parquet('crypto/dataset/train.parquet')
        
        print(f"✅ Successfully loaded train.parquet")
        print(f"📊 Train data shape: {train_df.shape}")
        print(f"📋 Train columns: {list(train_df.columns)}")
        
        print(f"\n📈 Sample train data (first 3 rows):")
        print(train_df.head(3))
        
        print(f"\n📊 Train data info:")
        print(train_df.info())
        
        # Load test parquet
        print("\n📖 Loading test.parquet...")
        test_df = pd.read_parquet('crypto/dataset/test.parquet')
        
        print(f"✅ Successfully loaded test.parquet")
        print(f"📊 Test data shape: {test_df.shape}")
        print(f"📋 Test columns: {list(test_df.columns)}")
        
        print(f"\n📈 Sample test data (first 3 rows):")
        print(test_df.head(3))
        
        return train_df, test_df
        
    except Exception as e:
        print(f"❌ Error loading parquet data: {e}")
        return None, None

def examine_sample_submission():
    """Examine the sample submission CSV"""
    print("\n🔍 EXAMINING SAMPLE SUBMISSION")
    print("=" * 50)
    
    try:
        sample_df = pd.read_csv('crypto/dataset/sample_submission.csv')
        
        print(f"✅ Successfully loaded sample_submission.csv")
        print(f"📊 Shape: {sample_df.shape}")
        print(f"📋 Columns: {list(sample_df.columns)}")
        
        print(f"\n📈 Sample submission data:")
        print(sample_df.head())
        
        return sample_df
        
    except Exception as e:
        print(f"❌ Error loading sample submission: {e}")
        return None

if __name__ == "__main__":
    print("🚀 CRYPTO DATA EXAMINATION SCRIPT")
    print("=" * 50)
    
    # Examine processed pickle data
    ts_train, ts_val, ts_test = examine_pickle_data()
    
    # Examine original parquet data
    train_df, test_df = examine_parquet_data()
    
    # Examine sample submission
    sample_df = examine_sample_submission()
    
    print("\n✅ DATA EXAMINATION COMPLETE!")
    print("=" * 50)
    
    if ts_train is not None:
        print(f"💡 Key insights:")
        print(f"  - Processed data has {ts_train.shape[1]} features")
        print(f"  - Target variable: 'label'")
        print(f"  - Data spans {len(ts_train) + len(ts_val) + len(ts_test):,} time steps")
        print(f"  - Each time step represents 1 minute of crypto market data") 