#!/bin/bash
#SBATCH --job-name=sinv_coeffs
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/sinv_%j.out
#SBATCH --error=logs/sinv_%j.err

mkdir -p logs

module load Julia/1.11.6-linux-x86_64

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "sinv_vs_sdag_coeffs on $(hostname)"
julia sinv_vs_sdag_coeffs.jl
