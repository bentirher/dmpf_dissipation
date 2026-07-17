#!/bin/bash
#SBATCH --job-name=direct_norm
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/direct_norm_%j.out
#SBATCH --error=logs/direct_norm_%j.err

mkdir -p logs

module load Julia/1.11.6-linux-x86_64

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "direct_norm_diag on $(hostname)"
julia direct_norm_diag.jl
