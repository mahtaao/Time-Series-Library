# Hyperparameter Tuning for Crypto Forecasting

This directory contains a complete hyperparameter optimization setup using Weights & Biases Sweeps for crypto forecasting models.

## 📁 Directory Structure

```
hyperparameter_tuning/
├── sweep_config.yaml      # W&B sweep configuration
├── train.py              # Sweep entry point
├── run_sweep.py          # Python script to run sweeps
├── run_sweep.sbatch      # SLURM batch script for cluster
└── README.md             # This documentation
```

## 🚀 Quick Start

### Option 1: Run on Cluster (Recommended)
```bash
sbatch hyperparameter_tuning/run_sweep.sbatch
```

### Option 2: Run Locally
```bash
cd hyperparameter_tuning
python run_sweep.py --count 10
```

### Option 3: Initialize Only
```bash
cd hyperparameter_tuning
python run_sweep.py --init_only
```

### Option 4: Run Existing Sweep
```bash
cd hyperparameter_tuning
python run_sweep.py --run_only --sweep_id YOUR_SWEEP_ID
```

## 📊 Sweep Configuration

The sweep configuration (`sweep_config.yaml`) follows [W&B Sweeps documentation](https://docs.wandb.ai/guides/sweeps/) and includes:

### Optimization Method
- **Method**: Bayesian optimization (`bayes`)
- **Metric**: `test_loss` (minimize)
- **Early Termination**: Hyperband for poor performing runs

### Hyperparameters Swept

#### Model Architecture
- `model`: ["Transformer", "iTransformer", "FEDformer", "Mamba"]
- `d_model`: 128-512
- `n_heads`: 4-16
- `e_layers`: 1-4
- `d_layers`: 1-2
- `d_ff`: 128-1024 (automatically capped at 256 for Mamba)

#### Training Parameters
- `learning_rate`: 5e-5 to 2e-4
- `batch_size`: 16-64
- `train_epochs`: 20-100
- `patience`: 5-20
- `dropout`: 0.05-0.2

#### Sequence Parameters
- `seq_len`: 48-192
- `label_len`: 24-96
- `pred_len`: 12-48

#### Data Dimensions
- `enc_in`: 12-48
- `dec_in`: 12-48
- `c_out`: 1-2

### Fixed Parameters
- GPU usage is enforced (`use_gpu: true`)
- W&B logging is enabled
- Data paths and task configuration are fixed
- Model-specific parameters are handled automatically

## 🔧 Key Features

### 1. Model-Specific Constraints
- **Mamba Model**: `d_ff` is automatically capped at 256 due to CUDA operation limitations
- **Memory Safety**: Parameter ranges are designed to prevent OOM errors
- **GPU Required**: All runs use GPU acceleration

### 2. Error Handling
- **Timeout Protection**: 1-hour timeout per run
- **Error Logging**: Failed runs are logged to W&B
- **Graceful Degradation**: Sweep continues even if individual runs fail

### 3. W&B Integration
- **Detailed Logging**: Uses your existing `crypto_exp.py` with comprehensive metrics
- **Sweep Dashboard**: Real-time monitoring of optimization progress
- **Result Analysis**: Easy comparison of different configurations

## 📈 Expected Results

The sweep will:
- Test different model architectures (Transformer, iTransformer, FEDformer, Mamba)
- Optimize hyperparameters for your crypto forecasting task
- Find the best performing configuration
- Log all results to W&B for analysis
- Respect model-specific limitations

## 🛠️ Advanced Usage

### Custom Configuration
```bash
python run_sweep.py \
    --config custom_sweep.yaml \
    --project my-project \
    --entity my-entity \
    --count 100
```

### Monitor Existing Sweep
```bash
# Get sweep ID from W&B dashboard
python run_sweep.py --run_only --sweep_id abc123
```

### Debug Single Run
```bash
cd hyperparameter_tuning
python train.py  # Uses default parameters
```

## 🔍 Monitoring

### W&B Dashboard
- **Sweep Overview**: https://wandb.ai/mahta-milaquebec/crypto-forecasting
- **Real-time Progress**: Monitor optimization as it runs
- **Result Comparison**: Compare different configurations
- **Parameter Importance**: See which parameters matter most

### Log Files
- **SLURM Output**: `logs/crypto_sweep_*.out`
- **SLURM Errors**: `logs/crypto_sweep_*.err`
- **W&B Logs**: All training metrics and predictions

## 🚨 Troubleshooting

### Common Issues

1. **GPU Not Available**
   ```bash
   # Check GPU availability
   nvidia-smi
   echo $CUDA_VISIBLE_DEVICES
   ```

2. **Memory Errors**
   - The `d_ff` range has been reduced to prevent OOM
   - Mamba models automatically cap `d_ff` at 256

3. **Sweep Initialization Failed**
   ```bash
   # Check W&B authentication
   wandb login
   ```

4. **Training Script Not Found**
   ```bash
   # Ensure you're in the correct directory
   pwd
   ls crypto/run_crypto_transformer.py
   ```

### Debug Mode
```bash
# Test single run with default parameters
cd hyperparameter_tuning
python train.py
```

## 📚 References

- [W&B Sweeps Documentation](https://docs.wandb.ai/guides/sweeps/)
- [Bayesian Optimization](https://docs.wandb.ai/guides/sweeps/optimize)
- [Early Termination](https://docs.wandb.ai/guides/sweeps/early-termination)
- [Sweep Configuration](https://docs.wandb.ai/guides/sweeps/configuration)

## 🎯 Best Practices

1. **Start Small**: Begin with 10-20 runs to test the setup
2. **Monitor Resources**: Watch GPU memory usage
3. **Check Logs**: Review W&B dashboard regularly
4. **Iterate**: Use results to refine parameter ranges
5. **Backup**: Save successful configurations

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review W&B sweep logs
3. Check SLURM job logs
4. Verify GPU availability and memory 