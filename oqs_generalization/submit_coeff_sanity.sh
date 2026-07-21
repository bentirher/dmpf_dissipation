#!/bin/bash
#SBATCH --job-name=coeff_sanity
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/coeff_sanity_%j.out
#SBATCH --error=logs/coeff_sanity_%j.err

mkdir -p logs

module load Julia/1.11.6-linux-x86_64

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "coeff_sanity on $(hostname)"
julia coeff_sanity.jl
