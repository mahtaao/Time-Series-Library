#!/bin/bash
#SBATCH --job-name=crypto_test
#SBATCH --account=def-bengioy
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/crypto_test_%j.out
#SBATCH --error=logs/crypto_test_%j.err
#SBATCH --mail-user=mahta.ramezanian@mila.quebec
#SBATCH --mail-type=ALL

# Load modules

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
CHECKPOINT_PATH="./checkpoints/iTransformer_job46505243"
MODEL="iTransformer"
BATCH_SIZE=32

echo "Testing checkpoint: $CHECKPOINT_PATH"
echo "Model: $MODEL"

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
    --batch_size $BATCH_SIZE \
    --save_plots

echo "Testing completed!" 