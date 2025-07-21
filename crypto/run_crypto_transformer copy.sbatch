#!/bin/bash
#SBATCH --job-name=crypto_transformer
#SBATCH --account=def-bengioy
#SBATCH --time=11:55:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/crypto_transformer_%j.out
#SBATCH --error=logs/crypto_transformer_%j.err
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
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"

# CHANGE THESE PARAMETERS AS NEEDED
MODEL="TimesNet"       # Model: Transformer, TimesNet, iTransformer, DLinear, etc.
# Note: For Mamba model, d_ff is capped at 256 due to selective_scan limitation
SEQ_LEN=96                    # Input sequence length
PRED_LEN=24                   # Prediction length
TRAIN_EPOCHS=100               # Training epochs
BATCH_SIZE=32                 # Batch size
LEARNING_RATE=0.0001          # Learning rate
D_MODEL=256                   # Model dimension
N_HEADS=8                     # Number of attention heads
E_LAYERS=2                    # Encoder layers
D_LAYERS=1                    # Decoder layers
D_FF=256                      # Feed-forward dimension (capped at 256 for Mamba)
PATIENCE=50                    # Early stopping patience

# W&B settings (change these to your W&B account)
WANDB_PROJECT="crypto-forecasting"
WANDB_ENTITY="mahta-milaquebec"  # Your W&B username
WANDB_RUN_NAME="train_${MODEL}_job${SLURM_JOB_ID}"

# Logging settings for detailed but less frequent logging
LOG_EVERY_N_EPOCHS=10          # Log detailed metrics every 10 epochs
LOG_PREDICTIONS=1              # Enable sample prediction logging
MAX_PRED_SAMPLES=5             # Maximum number of samples to log

# Create descriptive name for checkpoint directory
JOB_NAME="crypto_${MODEL,,}_job${SLURM_JOB_ID}"
echo "Running $MODEL on crypto data..."
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run: $WANDB_RUN_NAME"
echo "Logging every $LOG_EVERY_N_EPOCHS epochs"

# Run training
python crypto/run_crypto_transformer.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id crypto_test \
    --model $MODEL \
    --data custom \
    --seq_len $SEQ_LEN \
    --pred_len $PRED_LEN \
    --train_epochs $TRAIN_EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --e_layers $E_LAYERS \
    --d_layers $D_LAYERS \
    --d_ff $D_FF \
    --patience $PATIENCE \
    --enc_in 24 \
    --dec_in 24 \
    --inverse \
    --des $JOB_NAME \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --wandb_run_name $WANDB_RUN_NAME \
    --log_every_n_epochs $LOG_EVERY_N_EPOCHS \
    --log_predictions \
    --max_pred_samples $MAX_PRED_SAMPLES

echo "Training completed!" 