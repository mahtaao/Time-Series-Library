#!/bin/bash
#SBATCH --job-name=crypto_all_models
#SBATCH --account=def-bengioy
#SBATCH --time=8:55:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/all_models_%j.out
#SBATCH --error=logs/all_models_%j.err
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

# Models to test
MODELS=("Transformer" "TimesNet" "iTransformer" "DLinear" "PatchTST" "FEDformer" "Autoformer" "LightTS" "Informer")
# Note: Mamba requires d_ff <= 256 due to selective_scan limitation

# Parameters
SEQ_LEN=96
PRED_LEN=24
TRAIN_EPOCHS=50  # Increased for better comparison
BATCH_SIZE=32
LEARNING_RATE=0.0001
D_MODEL=256
N_HEADS=8
E_LAYERS=2
D_LAYERS=1
D_FF=1024  # Note: For Mamba model, this would be capped at 256
PATIENCE=15
LR_SCHEDULE="type2"  # Fixed learning rate schedule to prevent vanishing gradients

# W&B settings
WANDB_PROJECT="crypto-forecasting"
WANDB_ENTITY="mahta-milaquebec"

# Logging settings for detailed but less frequent logging
LOG_EVERY_N_EPOCHS=10          # Log detailed metrics every 10 epochs
LOG_PREDICTIONS=1              # Enable sample prediction logging
MAX_PRED_SAMPLES=3             # Maximum number of samples to log (reduced for multiple models)

echo "Running comparison of ${#MODELS[@]} models..."
echo "W&B Project: $WANDB_PROJECT"
echo "Logging every $LOG_EVERY_N_EPOCHS epochs"

# Run each model
for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Running $MODEL..."
    echo "=========================================="
    
    # Create unique run name for each model
    WANDB_RUN_NAME="train_${MODEL}_job${SLURM_JOB_ID}"
    JOB_NAME="crypto_${MODEL,,}_job${SLURM_JOB_ID}"
    
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
        --lradj $LR_SCHEDULE \
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
    
    echo "Completed $MODEL"
    echo ""
    
    # Clear GPU memory between models
    python -c "import torch; torch.cuda.empty_cache()"
done

echo "All models completed!" 