#!/bin/bash
# Locates the classically expensive corner of (n, gamma, t) for the vectorized
# MPS/TEBD route, to motivate the parameters of the ancilla-based quantum
# circuit experiment. Job array over gamma; one CSV per task.
#
# TMAX is set per gamma to roughly 2/gamma, so the operator-entanglement barrier
# peak (near t ~ 1/gamma) sits inside the window instead of just past its edge.
# Smaller gamma therefore means a longer, more expensive run -- that is the
# physics, not a scheduling accident, and the walltimes below reflect it.
#SBATCH --job-name=entgrow
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/entgrow_%A_%a.out
#SBATCH --error=logs/entgrow_%A_%a.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

case $SLURM_ARRAY_TASK_ID in
  0) export GAMMA=0.01 TMAX=200.0 NT=50 MAXDIM=1024 ;;  # tallest barrier, most expensive
  1) export GAMMA=0.05 TMAX=40.0  NT=40 MAXDIM=512  ;;  # the project's current gamma
  2) export GAMMA=0.10 TMAX=20.0  NT=40 MAXDIM=512  ;;
  3) export GAMMA=0.20 TMAX=10.0  NT=40 MAXDIM=256  ;;  # barrier already suppressed
esac

export N_LIST=8,12,16,20
export DT=0.02 ORDER=2 CUTOFF=1e-12 JCOUP=0.5 DISORDER=false

echo "entanglement_growth GAMMA=$GAMMA N_LIST=$N_LIST on $(hostname)"
echo "start: $(date)"
julia entanglement_growth_study.jl
echo "end: $(date)"
