#!/bin/bash
# =============================================================================
# submit_identity.sh -- localise the T4 identity failure
#
# n=3, ceiling 16, everything untruncated. Cheap: minutes, not hours.
# Task 0: the (8,8) diagonal pair. Task 1: the (3,8) off-diagonal pair, which
# showed the larger residual (8.9e-2) and has the badly conditioned k=3 defect.
# =============================================================================
#SBATCH --job-name=ident
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/ident_%A_%a.out
#SBATCH --error=logs/ident_%A_%a.err

mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

case $SLURM_ARRAY_TASK_ID in
  0) export KI=8 KJ=8 ;;
  1) export KI=3 KJ=8 ;;
esac

export N_QUBITS=3 GAMMA=0.05 TVAL=3.0 K0=48 ORDER=2

echo "identity_localise KI=$KI KJ=$KJ on $(hostname)"
echo "start: $(date)"
julia identity_localise.jl
echo "end: $(date)"
