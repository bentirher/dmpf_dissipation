#!/bin/bash
#SBATCH --job-name=dsum
#SBATCH --qos=regular
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/dsum_%j.out
#SBATCH --error=logs/dsum_%j.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "directsum_test on $(hostname)"; echo "start: $(date)"
julia directsum_test.jl
echo "end: $(date)"
