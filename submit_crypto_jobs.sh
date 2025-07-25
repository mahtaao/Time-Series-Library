#!/bin/bash

# Helper script to submit crypto forecasting jobs
# Usage: ./submit_crypto_jobs.sh [option]

echo "🚀 Crypto Forecasting Job Submission Helper"
echo "=========================================="

# Make scripts executable
chmod +x run_crypto_simple.sbatch
chmod +x run_crypto_papermill.sbatch

case "${1:-simple}" in
    "simple")
        echo "🚀 Submitting simple Python script job..."
        sbatch run_crypto_simple.sbatch
        ;;
    "papermill")
        echo "📊 Submitting papermill job..."
        sbatch run_crypto_papermill.sbatch
        ;;
    "all")
        echo "🔥 Submitting both job types..."
        echo "1. Simple Python script..."
        sbatch run_crypto_simple.sbatch
        echo "2. Papermill execution..."
        sbatch run_crypto_papermill.sbatch
        ;;
    *)
        echo "❌ Unknown option: $1"
        echo "Available options:"
        echo "  simple     - Run simple Python script (recommended, default)"
        echo "  papermill  - Run with papermill (parameterized)"
        echo "  all        - Submit both job types"
        echo ""
        echo "Usage: ./submit_crypto_jobs.sh [option]"
        exit 1
        ;;
esac

echo "✅ Job(s) submitted successfully!"
echo "📋 Check job status with: squeue -u $USER"
echo "📊 Monitor logs in: logs/ directory" 