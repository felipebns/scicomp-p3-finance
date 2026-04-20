#!/bin/bash
#SBATCH --job-name=finance-pipeline
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd ~/scicomp-p3-finance

mkdir -p logs

module purge
module load python/3.11.7

source venv/bin/activate

# Prevent conflicts with internal parallelism
# These environment variables tell libraries (OpenMP, MKL, OpenBLAS) to use single thread
# This is CRITICAL - without these, SLURM CPU allocation conflicts with library parallelism
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Starting finance pipeline job: $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Starting at: $(date)"

python main.py

echo "Job completed at: $(date)"
