#!/bin/bash
#SBATCH --job-name=agreement
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/agreement_%A_%a.out
#SBATCH --error=logs/agreement_%A_%a.err

mkdir -p logs agreement_results

gammas=(0.01 0.02 0.05 0.10)
gamma=${gammas[$SLURM_ARRAY_TASK_ID]}

module load Julia/1.11.6-linux-x86_64

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "agreement task $SLURM_ARRAY_TASK_ID: gamma=$gamma on $(hostname)"
julia agreement_multigamma.jl $gamma
