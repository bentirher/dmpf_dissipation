#!/bin/bash
#SBATCH --job-name=kref_conv
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/kref_conv_%j.out
#SBATCH --error=logs/kref_conv_%j.err

mkdir -p logs kref_results

module load Julia/1.11.6-linux-x86_64

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "kref_convergence on $(hostname)"
julia kref_convergence.jl
