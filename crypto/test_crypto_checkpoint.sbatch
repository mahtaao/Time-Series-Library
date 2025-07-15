#!/bin/bash
#SBATCH --job-name=crypto_test
#SBATCH --account=def-bengioy
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/crypto_test_%j.out
#SBATCH --error=logs/crypto_test_%j.err
#SBATCH --mail-user=mahta.ramezanian@mila.quebec
#SBATCH --mail-type=ALL

# Load modules
module load StdEnv/2020
module load gcc/11.3.0
module load python/3.11.5 || module load python/3.11
module load cuda/11.8 || module load cuda/11.8.0

# Activate environment
source ~/Time-Series-Library/.venv/bin/activate

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# Create logs directory
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi

# CHANGE THESE PARAMETERS AS NEEDED
CHECKPOINT_PATH="./checkpoints/long_term_forecast_crypto_test_Transformer_custom_ftM_sl96_ll48_pl24_dm256_nh8_el2_dl1_df1024_fc1_ebtimeF_dtTrue_crypto_transformer_job46345106_0"
MODEL="Transformer"
BATCH_SIZE=32

# W&B settings (change these to your W&B account)
WANDB_PROJECT="crypto-forecasting"
WANDB_ENTITY="your-wandb-username"  # Change this to your W&B username
WANDB_RUN_NAME="test_${MODEL}_job${SLURM_JOB_ID}"

echo "Testing checkpoint: $CHECKPOINT_PATH"
echo "Model: $MODEL"
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run: $WANDB_RUN_NAME"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_PATH"
    echo "Available checkpoints:"
    ls -la ./checkpoints/
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH/checkpoint.pth" ]; then
    echo "ERROR: Checkpoint file not found: $CHECKPOINT_PATH/checkpoint.pth"
    exit 1
fi

echo "Checkpoint found! Starting testing..."

# Run testing
python crypto/test_crypto_checkpoint.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --model "$MODEL" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --batch_size $BATCH_SIZE \
    --save_plots

echo "Testing completed!" 