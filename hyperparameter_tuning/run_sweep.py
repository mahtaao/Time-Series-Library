#!/usr/bin/env python3
"""
W&B Sweep Runner for Crypto Forecasting
Initialize and run hyperparameter optimization sweeps.
Based on https://docs.wandb.ai/guides/sweeps/
"""

import wandb
import argparse
import os
import sys
from pathlib import Path

def initialize_sweep(config_path, project, entity):
    """Initialize a new sweep"""
    
    print(f"🔧 Initializing sweep with config: {config_path}")
    print(f"📊 Project: {project}")
    print(f"👤 Entity: {entity}")
    
    try:
        sweep_id = wandb.sweep(
            sweep=config_path,
            project=project,
            entity=entity
        )
        
        print(f"✅ Sweep initialized successfully!")
        print(f"🆔 Sweep ID: {sweep_id}")
        print(f"🔗 Sweep URL: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
        
        return sweep_id
        
    except Exception as e:
        print(f"❌ Failed to initialize sweep: {e}")
        return None

def run_sweep_agent(sweep_id, project, entity, count=50):
    """Run the sweep agent"""
    
    print(f"🚀 Starting sweep agent for sweep: {sweep_id}")
    print(f"📊 Project: {project}")
    print(f"👤 Entity: {entity}")
    print(f"🔄 Run count: {count}")
    
    try:
        wandb.agent(
            f"{entity}/{project}/{sweep_id}",
            function=None,  # Uses train.py as the program
            count=count,
            project=project,
            entity=entity
        )
        
        print("✅ Sweep agent completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Sweep agent failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run W&B sweep for crypto forecasting')
    parser.add_argument('--config', type=str, default='sweep_config.yaml', 
                       help='Path to sweep configuration file')
    parser.add_argument('--project', type=str, default='crypto-forecasting',
                       help='W&B project name')
    parser.add_argument('--entity', type=str, default='mahta-milaquebec',
                       help='W&B entity name')
    parser.add_argument('--count', type=int, default=50,
                       help='Number of runs to execute')
    parser.add_argument('--sweep_id', type=str, default=None,
                       help='Existing sweep ID to run (skip initialization)')
    parser.add_argument('--init_only', action='store_true',
                       help='Only initialize sweep, don\'t run agent')
    parser.add_argument('--run_only', action='store_true',
                       help='Only run agent, don\'t initialize sweep')
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    sweep_id = args.sweep_id
    
    # Initialize sweep if needed
    if not args.run_only:
        sweep_id = initialize_sweep(config_path, args.project, args.entity)
        if not sweep_id:
            sys.exit(1)
    
    # Run sweep agent if needed
    if not args.init_only:
        if not sweep_id:
            print("❌ No sweep ID provided for running agent")
            sys.exit(1)
        
        success = run_sweep_agent(sweep_id, args.project, args.entity, args.count)
        if not success:
            sys.exit(1)
    
    print("🎉 Sweep operation completed!")

if __name__ == "__main__":
    main() 