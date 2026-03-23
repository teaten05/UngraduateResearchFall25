#!/bin/bash
#SBATCH --partition=standard-s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00        # Increased time
#SBATCH --mem=256G
#SBATCH --job-name=drone_sims
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log

# Set threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# CRITICAL: Limit Python parallel workers
export PYTHON_MAX_WORKERS=8  # Only 8 parallel simulations instead of 32

cd /users/tpsloan/mcmc_simulations/methodology

python main_pipeline.py --full

echo "Job completed successfully!"