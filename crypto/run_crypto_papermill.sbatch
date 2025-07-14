#!/bin/bash
#SBATCH --job-name=crypto_papermill
#SBATCH --account=def-bengioy
#SBATCH --time=0:50:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/crypto_papermill_%j.out
#SBATCH --error=logs/crypto_papermill_%j.err
#SBATCH --mail-user=mahta.ramezanian@mila.quebec
#SBATCH --mail-type=ALL

# Load required modules
module load StdEnv/2020
module load gcc/11.3.0
module load python/3.11
module load cuda/11.8.0

# Activate virtual environment
source ~/Time-Series-Library/.venv/bin/activate

# Install papermill if not already installed (skip if network issues)
pip install papermill --no-deps || echo "Warning: Could not install papermill, trying to continue..."

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# Create logs directory
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run notebook with papermill (allows parameter injection)
echo "Running notebook with papermill..."

# Create parameters file to avoid shell escaping issues
cat > papermill_params.json << 'EOF'
{
    "use_gpu": true,
    "batch_size": 64,
    "epochs": 10
}
EOF

papermill \
    crypto/DRW_transformer.ipynb \
    crypto/DRW_transformer_${SLURM_JOB_ID}.ipynb \
    --parameters_file papermill_params.json \
    --log-output \
    --progress-bar

# Clean up
rm -f papermill_params.json

echo "Papermill execution completed!" 