#!/bin/bash
# Fast regression test. Run after ANY edit to trotter_error_gram.jl.
# n=3, ceiling 16, untruncated. Minutes. Exits nonzero on failure.
#SBATCH --job-name=sanity
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sanity_%j.out
#SBATCH --error=logs/sanity_%j.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64
# BOTH are required: JULIA_NUM_THREADS alone does NOT control BLAS, and unpinned
# OpenBLAS spawns ~64 threads onto the allocation (68 s build -> >10 min hang).
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export N_QUBITS=3 GAMMA=0.05 TVAL=3.0 K0=48 ORDER=2
echo "sanity_check on $(hostname)"; echo "start: $(date)"
julia sanity_check.jl
echo "end: $(date)"
