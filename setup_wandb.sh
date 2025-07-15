#!/bin/bash
# Setup script for W&B integration

echo "Setting up W&B for crypto forecasting..."

# Activate virtual environment
source ~/Time-Series-Library/.venv/bin/activate

# Install W&B and dependencies
echo "Installing W&B and dependencies..."
pip install -r requirements_wandb.txt

# Check if W&B is installed
python -c "import wandb; print('W&B version:', wandb.__version__)"

echo ""
echo "W&B Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Login to W&B: wandb login"
echo "2. Update your W&B username in the batch scripts:"
echo "   - crypto/run_crypto_transformer.sbatch"
echo "   - crypto/test_crypto_checkpoint.sbatch"
echo "3. Run training: sbatch crypto/run_crypto_transformer.sbatch"
echo "4. Test checkpoint: sbatch crypto/test_crypto_checkpoint.sbatch"
echo ""
echo "Your W&B dashboard will show:"
echo "- Training metrics (loss, learning rate)"
echo "- Test results with detailed predictions"
echo "- Interactive plots and tables"
echo "- Model comparison across runs" 