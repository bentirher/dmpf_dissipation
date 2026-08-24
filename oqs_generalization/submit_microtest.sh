#!/bin/bash
# =============================================================================
# submit_microtest.sh -- isolate B + delta == A
#
# n=3, untruncated, minutes. Run this BEFORE any further sweeps: if the
# recursion's founding identity does not hold numerically, every number
# downstream of it is suspect.
# =============================================================================
#SBATCH --job-name=microtest
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/micro_%j.out
#SBATCH --error=logs/micro_%j.err

mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export N_QUBITS=3 GAMMA=0.05 TVAL=3.0 K0=48 ORDER=2

echo "delta_microtest on $(hostname)"
echo "start: $(date)"
julia delta_microtest.jl
echo "end: $(date)"
