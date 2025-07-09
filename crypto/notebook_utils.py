import subprocess
import time
import os
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

def submit_cell_as_job(
    cell_code: str,
    job_name: str,
    output_file: str,
    max_wait_time: int = 7200,
    memory: str = "32G",
    time_limit: str = "00:59:00",
    gpu: bool = True,
    account: str = "def-bengioy"
) -> Optional[Dict[str, Any]]:
    """Submit notebook cell as cluster job and wait for results."""
    
    print(f"Submitting {job_name} job...")
    
    
    python_script = f'''#!/usr/bin/env python3
import sys
import os
import pickle
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

# Create results directory
results_dir = Path('crypto/results')
results_dir.mkdir(exist_ok=True)

try:
    # Execute the original cell code
{cell_code}
    
    # Try to save trained model if it exists
    if 'model' in locals() and hasattr(locals()['model'], 'state_dict'):
        model = locals()['model']
        torch.save(model.state_dict(), results_dir / 'model.pth')
        print(f"✅ Trained model saved to {{results_dir / 'model.pth'}}")
    
    # Try to save predictions if they exist
    if 'predictions' in locals():
        np.save(results_dir / 'predictions.npy', locals()['predictions'])
        print(f"✅ Predictions saved to {{results_dir / 'predictions.npy'}}")
    
    if 'targets' in locals():
        np.save(results_dir / 'targets.npy', locals()['targets'])
        print(f"✅ Targets saved to {{results_dir / 'targets.npy'}}")
    
    if 'final_test_preds' in locals():
        np.save(results_dir / 'final_test_preds.npy', locals()['final_test_preds'])
        print(f"✅ Final test predictions saved to {{results_dir / 'final_test_preds.npy'}}")
    
    if 'benchmark_results' in locals():
        import pandas as pd
        benchmark_df = pd.DataFrame(locals()['benchmark_results'])
        benchmark_df.to_csv(results_dir / 'benchmark_results.csv', index=False)
        print(f"✅ Benchmark results saved to {{results_dir / 'benchmark_results.csv'}}")
    
    if 'weighted_ensemble_score' in locals():
        import json
        with open(results_dir / 'ensemble_score.json', 'w') as f:
            json.dump({{'weighted_ensemble_score': locals()['weighted_ensemble_score']}}, f)
        print(f"✅ Ensemble score saved to {{results_dir / 'ensemble_score.json'}}")
    
except Exception as e:
    print(f"❌ Error: {{e}}")

results = {{
    'status': 'completed',
    'job_name': '{job_name}',
}}

output_dir = os.path.dirname('{output_file}')
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
with open('{output_file}', 'wb') as f:
    pickle.dump(results, f)
'''
    
    gpu_flag = "#SBATCH --gres=gpu:1" if gpu else ""
    
    sbatch_script = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --output={job_name}_%j.out
#SBATCH --error={job_name}_%j.err
#SBATCH --time={time_limit}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task=4
{gpu_flag}

module load StdEnv/2020
module load gcc/11.3.0
module load cuda/11.8.0
module load python/3.11

# Activate virtual environment (correct path)
source $HOME/Time-Series-Library/.venv/bin/activate
cd $SLURM_SUBMIT_DIR

cat > run_{job_name}_temp.py << 'EOF'
{python_script}
EOF

python run_{job_name}_temp.py
rm -f run_{job_name}_temp.py
'''
    
    temp_sbatch_file = f'temp_{job_name}.sbatch'
    with open(temp_sbatch_file, 'w') as f:
        f.write(sbatch_script)
    
    try:
        # Submit with explicit account parameter
        result = subprocess.run(['sbatch', '--account', account, temp_sbatch_file], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"Job submitted: {job_id}")
            
            start_time = time.time()
            while time.time() - start_time < max_wait_time:
                result = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True)
                if result.returncode != 0 or job_id not in result.stdout:
                    break
                time.sleep(30)
            
            result = subprocess.run(['sacct', '-j', job_id, '--format=JobID,State,ExitCode'], 
                                   capture_output=True, text=True)
            
            if result.returncode == 0 and 'COMPLETED' in result.stdout:
                if os.path.exists(output_file):
                    with open(output_file, 'rb') as f:
                        return pickle.load(f)
            return None
        else:
            print(f"Failed to submit job: {result.stderr}")
            return None
            
    finally:
        if os.path.exists(temp_sbatch_file):
            os.remove(temp_sbatch_file) 