#!/bin/bash
#SBATCH --job-name=moc_compare
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/moc_compare_%j.out
#SBATCH --error=logs/moc_compare_%j.err

mkdir -p logs moc_results

module load Julia/1.11.6-linux-x86_64

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "moc_compare (closed system) on $(hostname)"
julia closed_moc_vs_plain.jl
