#!/bin/bash
#SBATCH --job-name=gram_conv
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/gram_conv_%j.out
#SBATCH --error=logs/gram_conv_%j.err

mkdir -p logs

module load Julia/1.11.6-linux-x86_64

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "gram_convergence on $(hostname)"
julia gram_convergence.jl
