#!/bin/bash
#SBATCH --job-name=crypto_sweep
#SBATCH --account=def-bengioy
#SBATCH --time=11:55:00  # Reduced from 23:55:00 to prevent time limit issues
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/crypto_sweep_%j.out
#SBATCH --error=logs/crypto_sweep_%j.err
#SBATCH --mail-user=mahta.ramezanian@mila.quebec
#SBATCH --mail-type=ALL

# Load modules
module load httpproxy/1.0

# Activate environment
source ~/Time-Series-Library/.venv/bin/activate

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# Create logs directory
mkdir -p logs

# Print job info
echo "=========================================="
echo "🚀 Crypto Forecasting Hyperparameter Sweep"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# W&B settings
WANDB_PROJECT="crypto-forecasting"
WANDB_ENTITY="mahta-milaquebec"
SWEEP_CONFIG="hyperparameter_tuning/sweep_config.yaml"
RUN_COUNT=50

echo "📊 W&B Configuration:"
echo "   Project: $WANDB_PROJECT"
echo "   Entity: $WANDB_ENTITY"
echo "   Config: $SWEEP_CONFIG"
echo "   Run count: $RUN_COUNT"
echo ""

# Change to hyperparameter_tuning directory
cd hyperparameter_tuning

echo "🔧 Starting hyperparameter optimization sweep..."
echo ""

# Run the sweep using our Python script
python run_sweep.py \
    --config sweep_config.yaml \
    --project $WANDB_PROJECT \
    --entity $WANDB_ENTITY \
    --count $RUN_COUNT

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Sweep completed successfully!"
    echo "📊 Check your W&B dashboard for results:"
    echo "   https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
else
    echo ""
    echo "❌ Sweep failed with exit code $?"
    exit 1
fi 