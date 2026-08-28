#!/bin/bash
#SBATCH --job-name=oracleconv
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/oracle_%j.out
#SBATCH --error=logs/oracle_%j.err

# Settles the n=6 ground truth. Cheap: this is an MPS calculation (ceiling 64),
# not an MPO one (ceiling 4096) -- 16 short runs, well inside 4 h.
# Run this BEFORE re-running the n=6 sweep: every dc in job 6995914 is measured
# against an oracle that drifted by 4.7e-4, which is 2.1% of the signal.

mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export N_QUBITS=6 GAMMA=0.05 TVAL=3.0 K0=48 K_REF=48 ORDER=2

echo "oracle_convergence on $(hostname)"; echo "start: $(date)"
julia oracle_convergence.jl
echo "end: $(date)"
