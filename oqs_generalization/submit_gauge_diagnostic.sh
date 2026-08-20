#!/bin/bash
# =============================================================================
# submit_gauge_diagnostic.sh   --   EXPERIMENT 1, run this FIRST
#
# Job array over (n, maxdim_lo, maxdim_hi):
#   task 0 : n=4, 32 vs  64   (cheap -- read this one first)
#   task 1 : n=5, 64 vs 256
#   task 2 : n=5, 128 vs 256  (the study size; the expensive one)
#
# Uses ONLY existing code. No new algorithm is tested here -- this measures
# whether the truncation error in (M, L, P) lands in the harmless gauge
# directions or in the ones that set the coefficients.
#
#   sbatch submit_gauge_diagnostic.sh
#   tail -f logs/gauge_*_0.out
# =============================================================================
#SBATCH --job-name=gauge_diag
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0
#SBATCH --output=logs/gauge_%A_%a.out
#SBATCH --error=logs/gauge_%A_%a.err

mkdir -p logs

module load Julia/1.11.6-linux-x86_64

# BOTH of these are required. JULIA_NUM_THREADS alone does NOT control BLAS,
# and unpinned OpenBLAS spawns ~64 threads onto the allocation -- this turned a
# 68 s build into a >10 min hang (findings_thematic.md section 5).
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

case $SLURM_ARRAY_TASK_ID in
  0) export N_QUBITS=4 MD_LO=64  MD_HI=256  ;;
  1) export N_QUBITS=5 MD_LO=64  MD_HI=256 ;;
  2) export N_QUBITS=5 MD_LO=128 MD_HI=256 ;;
esac

export GAMMA=0.05
export TVAL=3.0
export K_REF=40
export ORDER=2

echo "gauge_diagnostic  n=$N_QUBITS  md=$MD_LO vs $MD_HI  on $(hostname)"
echo "start: $(date)"

julia gauge_diagnostic.jl

echo "end: $(date)"
