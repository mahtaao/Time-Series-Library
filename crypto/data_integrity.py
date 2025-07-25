#!/usr/bin/env python3
"""
Data Integrity and Visualization Module for Crypto Forecasting

This module provides comprehensive data validation, anomaly detection,
and visualization capabilities using Weights & Biases.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

class DataIntegrityChecker:
    """
    Comprehensive data integrity checker for crypto forecasting data
    """
    
    def __init__(self, use_wandb: bool = True):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.issues = []
        self.warnings = []
        self.data_stats = {}
        
    def check_data_integrity(self, data_path: str, split: str = 'train') -> Dict:
        """
        Comprehensive data integrity check
        
        Args:
            data_path: Path to the data file
            split: Data split ('train', 'val', 'test')
            
        Returns:
            Dictionary containing integrity check results
        """
        print(f"🔍 Checking data integrity for {split} split...")
        
        # Load data
        if data_path.endswith('.pkl'):
            df = pd.read_pickle(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Create timestamp index
        df = self._create_timestamp_index(df)
        
        # Run all checks
        results = {
            'split': split,
            'shape': df.shape,
            'columns': list(df.columns),
            'timestamp_range': {
                'start': df.index.min(),
                'end': df.index.max(),
                'duration': df.index.max() - df.index.min()
            },
            'missing_values': self._check_missing_values(df),
            'duplicates': self._check_duplicates(df),
            'outliers': self._check_outliers(df),
            'data_types': self._check_data_types(df),
            'value_ranges': self._check_value_ranges(df),
            'correlations': self._check_correlations(df),
            'stationarity': self._check_stationarity(df),
            'anomalies': self._check_anomalies(df),
            'summary_stats': self._get_summary_stats(df)
        }
        
        # Log to W&B if available and initialized
        if self.use_wandb and wandb.run:
            self._log_integrity_results(results, split)
        
        return results
    
    def _create_timestamp_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create proper timestamp index for the data"""
        # Reset index if it's not already a datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(drop=True)
            # Create hourly timestamps starting from 2020-01-01 (fixed frequency)
            df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='h')
        
        return df
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        issues = []
        if missing.sum() > 0:
            issues.append(f"Found {missing.sum()} missing values across all columns")
            for col, count in missing.items():
                if count > 0:
                    issues.append(f"  - {col}: {count} ({missing_pct[col]:.2f}%)")
        
        return {
            'total_missing': missing.sum(),
            'missing_by_column': missing.to_dict(),
            'missing_pct_by_column': missing_pct.to_dict(),
            'issues': issues
        }
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate timestamps"""
        duplicates = df.index.duplicated().sum()
        
        issues = []
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate timestamps")
        
        return {
            'duplicate_timestamps': duplicates,
            'issues': issues
        }
    
    def _check_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> Dict:
        """Check for outliers using IQR method"""
        outliers = {}
        issues = []
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if col == 'label':  # Skip target variable for now
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_pct = (outlier_count / len(df)) * 100
            
            outliers[col] = {
                'count': outlier_count,
                'percentage': outlier_pct,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            if outlier_pct > 5:  # Flag if more than 5% outliers
                issues.append(f"  - {col}: {outlier_count} outliers ({outlier_pct:.2f}%)")
        
        if issues:
            issues.insert(0, f"Found outliers in {len([k for k, v in outliers.items() if v['count'] > 0])} columns")
        
        return {
            'outliers_by_column': outliers,
            'issues': issues
        }
    
    def _check_data_types(self, df: pd.DataFrame) -> Dict:
        """Check data types and consistency"""
        dtypes = df.dtypes.to_dict()
        issues = []
        
        # Check for non-numeric columns (except target)
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            issues.append(f"Non-numeric columns found: {non_numeric}")
        
        # Check for infinite values
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            issues.append(f"Found {inf_count} infinite values")
        
        return {
            'dtypes': dtypes,
            'non_numeric_columns': non_numeric,
            'infinite_values': inf_count,
            'issues': issues
        }
    
    def _check_value_ranges(self, df: pd.DataFrame) -> Dict:
        """Check value ranges and distributions"""
        ranges = {}
        issues = []
        
        for col in df.select_dtypes(include=[np.number]).columns:
            ranges[col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'median': df[col].median()
            }
            
            # Check for suspicious values
            if df[col].std() == 0:
                issues.append(f"  - {col}: Zero variance (constant values)")
            elif df[col].std() > df[col].mean() * 10:
                issues.append(f"  - {col}: Very high variance relative to mean")
        
        if issues:
            issues.insert(0, "Value range issues detected")
        
        return {
            'ranges_by_column': ranges,
            'issues': issues
        }
    
    def _check_correlations(self, df: pd.DataFrame) -> Dict:
        """Check feature correlations"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.95:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlation_pairs': high_corr_pairs,
            'issues': [f"Found {len(high_corr_pairs)} highly correlated feature pairs (>0.95)"] if high_corr_pairs else []
        }
    
    def _check_stationarity(self, df: pd.DataFrame) -> Dict:
        """Basic stationarity check using rolling statistics"""
        issues = []
        stationarity_results = {}
        
        for col in df.select_dtypes(include=[np.number]).columns[:5]:  # Check first 5 columns
            if col == 'label':
                continue
                
            # Calculate rolling mean and std
            rolling_mean = df[col].rolling(window=100).mean()
            rolling_std = df[col].rolling(window=100).std()
            
            # Check if rolling statistics are relatively stable
            mean_std = rolling_mean.std()
            std_std = rolling_std.std()
            
            stationarity_results[col] = {
                'rolling_mean_std': mean_std,
                'rolling_std_std': std_std,
                'is_stationary': mean_std < df[col].std() * 0.1  # Rough heuristic
            }
            
            if not stationarity_results[col]['is_stationary']:
                issues.append(f"  - {col}: Non-stationary (rolling mean varies significantly)")
        
        if issues:
            issues.insert(0, "Stationarity issues detected")
        
        return {
            'stationarity_by_column': stationarity_results,
            'issues': issues
        }
    
    def _check_anomalies(self, df: pd.DataFrame) -> Dict:
        """Detect anomalies using statistical methods"""
        anomalies = {}
        issues = []
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if col == 'label':
                continue
                
            # Z-score method
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            z_anomalies = (z_scores > 3).sum()
            
            # Moving average method
            ma = df[col].rolling(window=20).mean()
            ma_std = df[col].rolling(window=20).std()
            ma_anomalies = (np.abs(df[col] - ma) > 3 * ma_std).sum()
            
            anomalies[col] = {
                'z_score_anomalies': z_anomalies,
                'moving_avg_anomalies': ma_anomalies,
                'total_anomalies': max(z_anomalies, ma_anomalies)
            }
            
            if anomalies[col]['total_anomalies'] > len(df) * 0.01:  # More than 1%
                issues.append(f"  - {col}: {anomalies[col]['total_anomalies']} anomalies detected")
        
        if issues:
            issues.insert(0, "Anomalies detected")
        
        return {
            'anomalies_by_column': anomalies,
            'issues': issues
        }
    
    def _get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Get comprehensive summary statistics"""
        return {
            'total_samples': len(df),
            'total_features': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'date_range_days': (df.index.max() - df.index.min()).days
        }
    
    def _log_integrity_results(self, results: Dict, split: str):
        """Log integrity results to W&B"""
        if not self.use_wandb or not wandb.run:
            return
        
        # Log summary metrics
        wandb.log({
            f"data_integrity/{split}/total_samples": results['summary_stats']['total_samples'],
            f"data_integrity/{split}/total_features": results['summary_stats']['total_features'],
            f"data_integrity/{split}/memory_usage_mb": results['summary_stats']['memory_usage_mb'],
            f"data_integrity/{split}/missing_values": results['missing_values']['total_missing'],
            f"data_integrity/{split}/duplicate_timestamps": results['duplicates']['duplicate_timestamps'],
            f"data_integrity/{split}/infinite_values": results['data_types']['infinite_values'],
        })
        
        # Log issues as text
        all_issues = []
        for check_name, check_results in results.items():
            if isinstance(check_results, dict) and 'issues' in check_results:
                all_issues.extend(check_results['issues'])
        
        if all_issues:
            wandb.log({
                f"data_integrity/{split}/issues": wandb.Table(
                    columns=["Issue"],
                    data=[[issue] for issue in all_issues]
                )
            })

class DataVisualizer:
    """
    Data visualization using W&B for crypto forecasting data
    """
    
    def __init__(self, use_wandb: bool = True):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
    def visualize_features_vs_time(self, data_path: str, split: str = 'train', 
                                  max_features: int = 10, sample_size: int = 1000):
        """
        Visualize all features as a function of timestamp using W&B
        
        Args:
            data_path: Path to the data file
            split: Data split ('train', 'val', 'test')
            max_features: Maximum number of features to plot
            sample_size: Number of samples to use for visualization
        """
        print(f"📊 Visualizing features vs time for {split} split...")
        
        # Load data
        if data_path.endswith('.pkl'):
            df = pd.read_pickle(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Create timestamp index
        df = self._create_timestamp_index(df)
        
        # Sample data if too large - take first N points to preserve time series continuity
        if len(df) > sample_size:
            df = df.head(sample_size)  # Take first N points instead of random sampling
        
        # Get numeric features (excluding target)
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'label']
        
        # Limit number of features for visualization
        if len(numeric_cols) > max_features:
            # Select features with highest variance
            variances = df[numeric_cols].var().sort_values(ascending=False)
            numeric_cols = variances.head(max_features).index.tolist()
        
        # Create time series plots for each feature
        if self.use_wandb and wandb.run:
            self._create_wandb_time_series_plots(df, numeric_cols, split)
        
        # Create correlation heatmap
        self._create_correlation_heatmap(df, numeric_cols, split)
        
        # Create distribution plots
        self._create_distribution_plots(df, numeric_cols, split)
        
        print(f"✅ Visualization complete for {split} split")
    
    def _create_timestamp_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create proper timestamp index for the data"""
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(drop=True)
            df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='h')
        return df
    
    def _create_wandb_time_series_plots(self, df: pd.DataFrame, features: List[str], split: str):
        """Create time series plots using W&B"""
        if not self.use_wandb or not wandb.run:
            return
        
        # Create time series for each feature
        for i, feature in enumerate(features):
            # Prepare data for W&B line plot using proper datetime objects
            timestamps = df.index.tolist()  # Keep as datetime objects
            values = df[feature].tolist()
            
            # Create table for line plot with datetime objects
            data = [[ts, val] for ts, val in zip(timestamps, values)]
            table = wandb.Table(data=data, columns=["timestamp", feature])
            
            # Log individual feature plot
            wandb.log({
                f"time_series/{split}/{feature}": wandb.plot.line(
                    table, "timestamp", feature, title=f"{feature} vs Time - {split}"
                )
            })
        
        # Create combined plot for first few features using line_series
        if len(features) >= 3:
            timestamps = df.index.tolist()  # Keep as datetime objects
            feature_values = [df[feature].tolist() for feature in features[:3]]
            
            wandb.log({
                f"time_series/{split}/combined_features": wandb.plot.line_series(
                    xs=[timestamps] * 3,
                    ys=feature_values,
                    keys=features[:3],
                    title=f"Combined Features vs Time - {split}",
                    xname="Timestamp"
                )
            })
    
    def _create_correlation_heatmap(self, df: pd.DataFrame, features: List[str], split: str):
        """Create correlation heatmap using W&B"""
        if not self.use_wandb or not wandb.run:
            return
        
        # Calculate correlation matrix
        corr_matrix = df[features].corr()
        
        # Create correlation matrix as a table for W&B
        wandb.log({
            f"correlations/{split}/correlation_matrix": wandb.Table(
                columns=["feature"] + features,
                data=[[f] + [corr_matrix.loc[f, col] for col in features] for f in features]
            )
        })
    
    def _create_distribution_plots(self, df: pd.DataFrame, features: List[str], split: str):
        """Create distribution plots using W&B"""
        if not self.use_wandb or not wandb.run:
            return
        
        # Create histograms for each feature using proper W&B format
        for feature in features:
            # Create table for histogram
            data = [[val] for val in df[feature]]
            table = wandb.Table(data=data, columns=[feature])
            
            wandb.log({
                f"distributions/{split}/{feature}": wandb.plot.histogram(
                    table, feature, title=f"{feature} Distribution - {split}"
                )
            })

class DataFlowTracker:
    """
    Track and visualize how data flows into the model
    """
    
    def __init__(self, use_wandb: bool = True):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.data_flow_log = []
        
    def track_data_flow(self, dataset, dataloader, model_args, split: str = 'train'):
        """
        Track how data flows through the pipeline
        
        Args:
            dataset: Dataset object
            dataloader: DataLoader object
            model_args: Model arguments
            split: Data split
        """
        print(f"🔄 Tracking data flow for {split} split...")
        
        # Get sample batch
        sample_batch = next(iter(dataloader))
        batch_x, batch_y, batch_x_mark, batch_y_mark = sample_batch
        
        # Track data shapes and statistics
        flow_info = {
            'split': split,
            'dataset_size': len(dataset),
            'batch_size': model_args.batch_size,
            'num_batches': len(dataloader),
            'input_shape': batch_x.shape,
            'target_shape': batch_y.shape,
            'time_features_shape': batch_x_mark.shape,
            'seq_len': model_args.seq_len,
            'pred_len': model_args.pred_len,
            'label_len': model_args.label_len,
            'num_features': batch_x.shape[-1],
            'data_type': str(batch_x.dtype),
            'device': str(batch_x.device) if hasattr(batch_x, 'device') else 'CPU'
        }
        
        # Calculate statistics
        flow_info.update({
            'input_stats': {
                'mean': float(batch_x.mean()),
                'std': float(batch_x.std()),
                'min': float(batch_x.min()),
                'max': float(batch_x.max())
            },
            'target_stats': {
                'mean': float(batch_y.mean()),
                'std': float(batch_y.std()),
                'min': float(batch_y.min()),
                'max': float(batch_y.max())
            }
        })
        
        self.data_flow_log.append(flow_info)
        
        # Log to W&B
        if self.use_wandb and wandb.run:
            self._log_data_flow(flow_info)
        
        return flow_info
    
    def _log_data_flow(self, flow_info: Dict):
        """Log data flow information to W&B"""
        if not self.use_wandb or not wandb.run:
            return
        
        split = flow_info['split']
        
        # Log basic metrics
        wandb.log({
            f"data_flow/{split}/dataset_size": flow_info['dataset_size'],
            f"data_flow/{split}/batch_size": flow_info['batch_size'],
            f"data_flow/{split}/num_batches": flow_info['num_batches'],
            f"data_flow/{split}/seq_len": flow_info['seq_len'],
            f"data_flow/{split}/pred_len": flow_info['pred_len'],
            f"data_flow/{split}/num_features": flow_info['num_features'],
        })
        
        # Log input statistics
        wandb.log({
            f"data_flow/{split}/input_mean": flow_info['input_stats']['mean'],
            f"data_flow/{split}/input_std": flow_info['input_stats']['std'],
            f"data_flow/{split}/input_min": flow_info['input_stats']['min'],
            f"data_flow/{split}/input_max": flow_info['input_stats']['max'],
        })
        
        # Log target statistics
        wandb.log({
            f"data_flow/{split}/target_mean": flow_info['target_stats']['mean'],
            f"data_flow/{split}/target_std": flow_info['target_stats']['std'],
            f"data_flow/{split}/target_min": flow_info['target_stats']['min'],
            f"data_flow/{split}/target_max": flow_info['target_stats']['max'],
        })
        
        # Log data flow table
        wandb.log({
            f"data_flow/{split}/flow_info": wandb.Table(
                columns=["Property", "Value"],
                data=[
                    ["Dataset Size", str(flow_info['dataset_size'])],
                    ["Batch Size", str(flow_info['batch_size'])],
                    ["Input Shape", str(flow_info['input_shape'])],
                    ["Target Shape", str(flow_info['target_shape'])],
                    ["Sequence Length", str(flow_info['seq_len'])],
                    ["Prediction Length", str(flow_info['pred_len'])],
                    ["Number of Features", str(flow_info['num_features'])],
                    ["Data Type", flow_info['data_type']],
                    ["Device", flow_info['device']]
                ]
            )
        })

def run_comprehensive_data_analysis(data_dir: str, use_wandb: bool = True):
    """
    Run comprehensive data analysis including integrity checks and visualization
    
    Args:
        data_dir: Directory containing the data files
        use_wandb: Whether to use W&B for logging
    """
    print("🚀 Starting comprehensive data analysis...")
    
    # Initialize W&B first if requested
    if use_wandb and WANDB_AVAILABLE and not wandb.run:
        wandb.init(
            project='crypto-data-integrity',
            entity='mahta-milaquebec',
            name=f"data-integrity-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            settings=wandb.Settings(init_timeout=120)
        )
    
    # Initialize components
    integrity_checker = DataIntegrityChecker(use_wandb=use_wandb)
    visualizer = DataVisualizer(use_wandb=use_wandb)
    
    # Data files to analyze
    data_files = {
        'train': os.path.join(data_dir, 'ts_train.pkl'),
        'val': os.path.join(data_dir, 'ts_val.pkl'),
        'test': os.path.join(data_dir, 'ts_test.pkl')
    }
    
    # Run integrity checks
    integrity_results = {}
    for split, file_path in data_files.items():
        if os.path.exists(file_path):
            print(f"\n📋 Running integrity checks for {split} split...")
            integrity_results[split] = integrity_checker.check_data_integrity(file_path, split)
            
            # Print summary
            print(f"   Shape: {integrity_results[split]['shape']}")
            print(f"   Missing values: {integrity_results[split]['missing_values']['total_missing']}")
            print(f"   Duplicates: {integrity_results[split]['duplicates']['duplicate_timestamps']}")
            
            # Check for critical issues
            all_issues = []
            for check_name, check_results in integrity_results[split].items():
                if isinstance(check_results, dict) and 'issues' in check_results:
                    all_issues.extend(check_results['issues'])
            
            if all_issues:
                print(f"   ⚠️  Found {len(all_issues)} issues")
                for issue in all_issues[:3]:  # Show first 3 issues
                    print(f"      {issue}")
            else:
                print(f"   ✅ No issues found")
    
    # Run visualizations for all splits
    for split, file_path in data_files.items():
        if os.path.exists(file_path):
            print(f"\n📊 Creating visualizations for {split} split...")
            visualizer.visualize_features_vs_time(file_path, split)
    
    print(f"\n✅ Comprehensive data analysis complete!")
    
    return integrity_results

if __name__ == "__main__":
    # Example usage
    data_dir = "./crypto/dataset/"
    run_comprehensive_data_analysis(data_dir, use_wandb=True) 