#!/bin/bash
#SBATCH --job-name=itransformer_eeg
#SBATCH --account=def-gdumas85
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/itransformer_eeg_%j.out
#SBATCH --error=logs/itransformer_eeg_%j.err
#SBATCH --mail-user=mahta.ramezanian@mila.quebec
#SBATCH --mail-type=ALL

# Load required modules
module load python/3.11
module load cuda/11.8

# Activate virtual environment if you have one
# source ~/venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create logs directory
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Set model parameters
model_name=iTransformer

# Run the training
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/mahta/projects/def-gdumas85/tmp/omneeg \
  --model_id eeg_forecast \
  --model $model_name \
  --data h5_eeg \
  --features M \
  --seq_len 500 \
  --label_len 250 \
  --pred_len 100 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1024 \
  --dec_in 1024 \
  --c_out 1024 \
  --des 'EEG_Forecast_Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --n_heads 8 \
  --batch_size 8 \
  --learning_rate 0.0001 \
  --train_epochs 50 \
  --patience 10 \
  --itr 1 \
  --dropout 0.1 \
  --embed timeF \
  --freq h \
  --activation gelu \
  --use_gpu True \
  --gpu 0 \
  --num_workers 4 \
  --checkpoints ./checkpoints/

echo "Training completed!" 