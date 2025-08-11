#!/usr/bin/env python3
"""
Script to show sample values for each of the 22 features in the crypto dataset
"""

import pandas as pd
import numpy as np

def show_feature_samples():
    """Show sample values for each feature"""
    print("🔍 CRYPTO DATA FEATURE SAMPLES")
    print("=" * 60)
    
    try:
        # Load the training data
        print("📖 Loading ts_train.pkl...")
        ts_train = pd.read_pickle('crypto/dataset/ts_train.pkl')
        
        print(f"✅ Loaded data with shape: {ts_train.shape}")
        print(f"📊 Features: {len(ts_train.columns)} total")
        print()
        
        # Separate features by type
        x_features = [col for col in ts_train.columns if col.startswith('X')]
        trading_features = [col for col in ts_train.columns if not col.startswith('X') and col != 'label']
        target_feature = 'label'
        
        print(f"🎯 Feature Breakdown:")
        print(f"  X-features (engineered): {len(x_features)}")
        print(f"  Trading features: {len(trading_features)}")
        print(f"  Target feature: 1")
        print()
        
        # Show samples for X-features
        print("🔧 X-FEATURES (Engineered Features):")
        print("-" * 40)
        for i, feature in enumerate(x_features, 1):
            sample_values = ts_train[feature].head(5).tolist()
            stats = {
                'mean': ts_train[feature].mean(),
                'std': ts_train[feature].std(),
                'min': ts_train[feature].min(),
                'max': ts_train[feature].max()
            }
            print(f"{i:2d}. {feature}:")
            print(f"    Sample values: {[f'{x:.6f}' for x in sample_values]}")
            print(f"    Stats - Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}, Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            print()
        
        # Show samples for trading features
        print("📈 TRADING FEATURES (Raw Market Data):")
        print("-" * 40)
        for i, feature in enumerate(trading_features, 1):
            sample_values = ts_train[feature].head(5).tolist()
            stats = {
                'mean': ts_train[feature].mean(),
                'std': ts_train[feature].std(),
                'min': ts_train[feature].min(),
                'max': ts_train[feature].max()
            }
            print(f"{i:2d}. {feature}:")
            print(f"    Sample values: {[f'{x:.6f}' for x in sample_values]}")
            print(f"    Stats - Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}, Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            print()
        
        # Show target feature
        print("🎯 TARGET FEATURE:")
        print("-" * 40)
        sample_values = ts_train[target_feature].head(5).tolist()
        stats = {
            'mean': ts_train[target_feature].mean(),
            'std': ts_train[target_feature].std(),
            'min': ts_train[target_feature].min(),
            'max': ts_train[target_feature].max()
        }
        print(f"1. {target_feature}:")
        print(f"   Sample values: {[f'{x:.6f}' for x in sample_values]}")
        print(f"   Stats - Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}, Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        print()
        
        # Show correlation with target
        print("🔗 TOP 10 FEATURES BY TARGET CORRELATION:")
        print("-" * 40)
        feature_correlations = []
        for feature in ts_train.columns:
            if feature != 'label':
                corr = ts_train[feature].corr(ts_train['label'])
                feature_correlations.append((feature, abs(corr)))
        
        feature_correlations.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, corr) in enumerate(feature_correlations[:10], 1):
            feature_type = "X-feature" if feature.startswith('X') else "Trading"
            print(f"{i:2d}. {feature}: {corr:.6f} ({feature_type})")
        
        print()
        print("✅ FEATURE SAMPLES COMPLETE!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    show_feature_samples() 