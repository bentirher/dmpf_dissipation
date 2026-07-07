#!/bin/bash
#SBATCH --job-name=mpf_sweep
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

mkdir -p logs sweep_results

maxdims=(16 32 64 128 256)
md=${maxdims[$SLURM_ARRAY_TASK_ID]}

module load Julia/1.11.6-linux-x86_64   # use whatever `module spider julia` gave you
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Task $SLURM_ARRAY_TASK_ID: maxdim=$md on $(hostname)"
julia run_sweep_case.jl $md