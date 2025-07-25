#!/bin/bash
#SBATCH --job-name=crypto_inference
#SBATCH --account=def-bengioy
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/crypto_inference_%j.out
#SBATCH --error=logs/crypto_inference_%j.err
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
# Extract model info from checkpoint name
CHECKPOINT_NAME="iTransformer_job46505243"

# Parse checkpoint name to extract model parameters
# Format: {MODEL}_job{JOB_ID}
echo "Parsing checkpoint name: $CHECKPOINT_NAME"

# Extract model name (before _job)
MODEL=$(echo $CHECKPOINT_NAME | sed -n 's/^\([^_]*\)_job.*/\1/p')
echo "Extracted model: $MODEL"

# Extract job ID (after job)
JOB_ID=$(echo $CHECKPOINT_NAME | sed -n 's/.*job\([0-9]*\).*/\1/p')
echo "Extracted job ID: $JOB_ID"

# For simplified checkpoint names, we use default parameters
# These should match the training configuration
SEQ_LEN=96
LABEL_LEN=48
PRED_LEN=24
D_MODEL=256
N_HEADS=8
E_LAYERS=2
D_LAYERS=1
D_FF=256
FACTOR=1
EMBED=timeF
DISTIL=True

# Set default values if parsing fails
SEQ_LEN=${SEQ_LEN:-96}
LABEL_LEN=${LABEL_LEN:-48}
PRED_LEN=${PRED_LEN:-24}
D_MODEL=${D_MODEL:-256}
N_HEADS=${N_HEADS:-8}
E_LAYERS=${E_LAYERS:-2}
D_LAYERS=${D_LAYERS:-1}
D_FF=${D_FF:-256}
FACTOR=${FACTOR:-1}
EMBED=${EMBED:-timeF}
DISTIL=${DISTIL:-True}

# Model-specific parameter validation
if [ "$MODEL" = "Mamba" ]; then
    echo "Mamba model detected - capping d_ff at 256 for CUDA compatibility"
    D_FF=256
fi

# Inference parameters
BATCH_SIZE=32
USE_WANDB=1
WANDB_PROJECT="crypto-forecasting"
WANDB_ENTITY="mahta-milaquebec"
WANDB_RUN_NAME="inference_${MODEL}_job${SLURM_JOB_ID}"

echo "Inference Configuration:"
echo "  Model: $MODEL"
echo "  Checkpoint: $CHECKPOINT_NAME"
echo "  Sequence Length: $SEQ_LEN"
echo "  Label Length: $LABEL_LEN"
echo "  Prediction Length: $PRED_LEN"
echo "  d_model: $D_MODEL"
echo "  n_heads: $N_HEADS"
echo "  e_layers: $E_LAYERS"
echo "  d_layers: $D_LAYERS"
echo "  d_ff: $D_FF"
echo "  factor: $FACTOR"
echo "  embed: $EMBED"
echo "  distil: $DISTIL"
echo "  W&B Project: $WANDB_PROJECT"
echo "  W&B Run: $WANDB_RUN_NAME"

# Verify checkpoint exists
CHECKPOINT_PATH="./checkpoints/${CHECKPOINT_NAME}"
if [ ! -f "${CHECKPOINT_PATH}/checkpoint.pth" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT_PATH}/checkpoint.pth"
    exit 1
fi

echo "Checkpoint found successfully!"

# Run inference
python crypto/test_crypto_checkpoint.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --model "$MODEL" \
    --batch_size $BATCH_SIZE \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \
    --wandb_entity $WANDB_ENTITY \
    --save_plots

echo "Inference completed!" 