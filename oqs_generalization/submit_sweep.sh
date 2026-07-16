#!/bin/bash
#SBATCH --job-name=mpf_sweep
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err

mkdir -p logs sweep_results

module load Julia/1.11.6-linux-x86_64   # use whatever `module spider julia` gave you
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "mpf_sweep (de-duplicated, single job) on $(hostname)"
julia run_sweep_case_v2.jl