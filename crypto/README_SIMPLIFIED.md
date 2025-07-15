# Simplified Crypto Forecasting with Time-Series-Library

This directory contains a **simplified and clean approach** to running transformer models on your preprocessed crypto data using the Time-Series-Library instead of the complex custom transformer implementation.

## 🎯 What Changed

**Before (Complex):**
- Custom transformer implementation with 200+ lines of code
- Manual data loading and preprocessing
- Complex training loops
- Hard to maintain and debug

**After (Simple):**
- Uses proven transformer models from Time-Series-Library
- Clean data integration with your preprocessed data
- Standard training pipeline
- Easy to experiment with different models

## 📁 Files Overview

### Core Files
- `run_crypto_transformer.py` - Main script to run any transformer model
- `crypto_exp.py` - Custom experiment class for crypto data
- `crypto_data_factory.py` - Data provider for crypto datasets
- `run_crypto_transformer.py` - Contains CryptoDataset class

### Utility Files
- `run_transformer_models.py` - Run multiple models for comparison
- `test_simple_transformer.py` - Test script to verify setup
- `README_SIMPLIFIED.md` - This file

## 🚀 Quick Start

### 1. Test the Setup
```bash
cd /home/mahta/Time-Series-Library
python crypto/test_simple_transformer.py
```

### 2. Run a Single Model
```bash
# Run Transformer
python crypto/run_crypto_transformer.py --model Transformer

# Run TimesNet (state-of-the-art)
python crypto/run_crypto_transformer.py --model TimesNet

# Run iTransformer
python crypto/run_crypto_transformer.py --model iTransformer
```

### 3. Run Multiple Models for Comparison
```bash
python crypto/run_transformer_models.py
```

## 🎛️ Available Models

The Time-Series-Library provides many state-of-the-art transformer models:

| Model | Description | Performance |
|-------|-------------|-------------|
| **TimesNet** | State-of-the-art temporal modeling | 🥇 Best overall |
| **iTransformer** | Inverted transformer architecture | 🥈 Excellent |
| **Transformer** | Standard transformer | 🥉 Good baseline |
| **DLinear** | Simple but effective | 🥉 Fast training |
| **PatchTST** | Patch-based transformer | 🥉 Good for long sequences |
| **FEDformer** | Frequency enhanced | 🥉 Good for seasonal data |
| **Autoformer** | Auto-correlation based | 🥉 Efficient |
| **LightTS** | Lightweight transformer | 🥉 Fast inference |

## ⚙️ Configuration Options

### Basic Parameters
```bash
--model Transformer          # Model name
--seq_len 96                 # Input sequence length
--pred_len 24                # Prediction length
--train_epochs 10            # Training epochs
--batch_size 32              # Batch size
--learning_rate 0.0001       # Learning rate
```

### Model-Specific Parameters
```bash
--d_model 256                # Model dimension
--n_heads 8                  # Number of attention heads
--e_layers 2                 # Encoder layers
--d_layers 1                 # Decoder layers
--d_ff 1024                  # Feedforward dimension
```

### Data Parameters
```bash
--enc_in 22                  # Number of input features
--dec_in 22                  # Number of decoder input features
--c_out 1                    # Number of output features
--features M                 # M: multivariate, S: univariate
```

## 📊 Example Commands

### Quick Test (Small Model)
```bash
python crypto/run_crypto_transformer.py \
    --model Transformer \
    --d_model 128 \
    --n_heads 4 \
    --e_layers 1 \
    --d_layers 1 \
    --train_epochs 5 \
    --batch_size 16
```

### Full Training (Large Model)
```bash
python crypto/run_crypto_transformer.py \
    --model TimesNet \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 3 \
    --d_layers 2 \
    --train_epochs 50 \
    --batch_size 32 \
    --learning_rate 0.0001
```

### Comparison Run
```bash
python crypto/run_transformer_models.py
```

## 🔧 Customization

### Adding New Models
The Time-Series-Library supports many models. To add a new one:

1. Check if it's available in `models/` directory
2. Add it to the model list in `run_transformer_models.py`
3. Run with `--model ModelName`

### Modifying Data Processing
Edit `CryptoDataset` class in `run_crypto_transformer.py` to:
- Change data loading logic
- Modify preprocessing steps
- Adjust sequence creation

### Changing Training Parameters
Modify `create_crypto_args()` function in `run_crypto_transformer.py` to:
- Add new command line arguments
- Change default values
- Add model-specific parameters

## 📈 Results and Logging

### Checkpoints
Models are saved in `./checkpoints/` with naming pattern:
```
long_term_forecast_crypto_forecast_ModelName_custom_ftM_sl96_ll48_pl24_dm256_nh8_el2_dl1_df1024_fc5_ebtimeF_dtTrue_crypto_exp_0/
```

### Logs
Training logs are saved in `./logs/` with the same naming pattern.

### Metrics
The library automatically computes and logs:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the right directory
   cd /home/mahta/Time-Series-Library
   ```

2. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or model size
   --batch_size 16 --d_model 128
   ```

3. **Data Loading Issues**
   ```bash
   # Check if preprocessed.pkl exists
   ls -la crypto/data/preprocessed.pkl
   ```

4. **Model Not Found**
   ```bash
   # Check available models
   ls models/ | grep -i transformer
   ```

### Debug Mode
Run the test script to identify issues:
```bash
python crypto/test_simple_transformer.py
```

## 🎉 Benefits of This Approach

1. **Simplicity**: Much cleaner and easier to understand
2. **Reliability**: Uses proven, tested transformer implementations
3. **Flexibility**: Easy to switch between different models
4. **Maintainability**: Standard code structure and patterns
5. **Performance**: Access to state-of-the-art models like TimesNet
6. **Extensibility**: Easy to add new models or modify existing ones

## 📚 Next Steps

1. **Start Simple**: Run the test script to verify everything works
2. **Experiment**: Try different models with `run_transformer_models.py`
3. **Optimize**: Tune hyperparameters for your specific use case
4. **Compare**: Analyze results across different models
5. **Deploy**: Use the best model for your crypto forecasting needs

This simplified approach gives you access to the latest transformer research while maintaining a clean, maintainable codebase! 