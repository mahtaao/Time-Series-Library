# Crypto Forecasting Cluster Jobs

This directory contains sbatch scripts to run your crypto forecasting notebook non-interactively on the cluster.

## Quick Start

### Option 1: Simple Python Script (Recommended)
```bash
./submit_crypto_jobs.sh simple
```

### Option 2: All Methods
```bash
./submit_crypto_jobs.sh all
```

## Available Job Types

### 1. Simple Python Script (`run_crypto_simple.sbatch`) ⭐ **Recommended**
- **What it does**: Runs the converted Python script directly
- **Pros**: Fast, reliable, no external dependencies, proven to work
- **Cons**: No notebook output
- **Usage**: `./submit_crypto_jobs.sh simple`

### 2. Papermill Execution (`run_crypto_papermill.sbatch`)
- **What it does**: Runs notebook with parameter injection
- **Pros**: Can modify parameters, preserves notebook structure
- **Cons**: Requires papermill installation, may have network issues
- **Usage**: `./submit_crypto_jobs.sh papermill`

## Job Configuration

### Resource Allocation
- **GPU**: 1 GPU per job
- **Memory**: 16-64GB depending on job type
- **CPU**: 8 cores
- **Time**: 2-8 hours depending on job type
- **Account**: def-bengioy (updated from def-gdumas85)

### Environment Setup
- Python 3.11
- CUDA 11.8.0 (fixed from 11.8)
- Virtual environment: `~/Time-Series-Library/.venv`

## Monitoring Jobs

### Check Job Status
```bash
squeue -u $USER
```

### View Logs
```bash
# View output logs
tail -f logs/crypto_simple_*.out

# View error logs  
tail -f logs/crypto_simple_*.err
```

### Cancel Jobs
```bash
scancel <job_id>
```

## Troubleshooting

### Common Issues

1. **CUDA Module Error**
   - **Problem**: `cuda/11.8` not found
   - **Solution**: Updated to `cuda/11.8.0` in all scripts

2. **Network Connectivity**
   - **Problem**: Can't install papermill
   - **Solution**: Use `simple` or `python` options instead

3. **Memory Issues**
   - **Problem**: Out of memory errors
   - **Solution**: Increase `--mem` in sbatch script

4. **Timeout Issues**
   - **Problem**: Jobs taking too long
   - **Solution**: Increase `--time` in sbatch script

### Debugging Steps

1. **Check module availability**:
   ```bash
   module spider cuda
   module spider python
   ```

2. **Test environment locally**:
   ```bash
   source ~/Time-Series-Library/.venv/bin/activate
   python crypto_forecast_script.py
   ```

3. **Check data availability**:
   ```bash
   ls -la data/
   ```

## Output Files

### Generated Files
- `submission_with_X174.csv` - Final predictions
- `logs/crypto_*_*.out` - Job output logs
- `logs/crypto_*_*.err` - Job error logs

### For Papermill Jobs
- `crypto/20250531_DRW_executed_*.ipynb` - Executed notebooks

## Customization

### Modify Resource Allocation
Edit any `.sbatch` file to change:
```bash
#SBATCH --time=2:00:00      # Job time limit
#SBATCH --mem=32G           # Memory allocation
#SBATCH --cpus-per-task=8   # CPU cores
#SBATCH --gres=gpu:1        # GPU count
```

### Change Account
```bash
#SBATCH --account=def-your-account
```

### Modify Email Notifications
```bash
#SBATCH --mail-user=your.email@domain.com
#SBATCH --mail-type=ALL
```

## Best Practices

1. **Start with Simple**: Use `simple` option first to test
2. **Monitor Resources**: Check job logs for memory/GPU usage
3. **Use Appropriate Time**: Estimate runtime and add buffer
4. **Keep Logs**: Save important outputs and logs
5. **Test Locally**: Verify script works before submitting

## Example Workflow

```bash
# 1. Test locally first
python crypto_forecast_script.py

# 2. Submit simple job
./submit_crypto_jobs.sh simple

# 3. Monitor progress
squeue -u $USER
tail -f logs/crypto_simple_*.out

# 4. Check results
ls -la submission_with_X174.csv
```

## Support

If you encounter issues:
1. Check the error logs in `logs/`
2. Verify your environment setup
3. Test with the `simple` option first
4. Contact cluster support if needed 