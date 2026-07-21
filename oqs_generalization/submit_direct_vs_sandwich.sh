#!/bin/bash
#SBATCH --job-name=dir_vs_sand
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dir_vs_sand_%j.out
#SBATCH --error=logs/dir_vs_sand_%j.err

mkdir -p logs

module load Julia/1.11.6-linux-x86_64

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "direct_vs_sandwich on $(hostname)"
julia direct_vs_sandwich.jl
