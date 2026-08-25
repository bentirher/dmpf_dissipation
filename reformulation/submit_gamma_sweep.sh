#!/bin/bash
# THE RECOMMENDED NEXT EXPERIMENT. Job array over gamma; one CSV per task.
# Answers "does the N-route advantage survive stronger dissipation?" -- the
# single most important open question, and the direct successor to slide 22.
#SBATCH --job-name=gsweep
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/gsweep_%A_%a.out
#SBATCH --error=logs/gsweep_%A_%a.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
case $SLURM_ARRAY_TASK_ID in
  0) export GAMMA=0.01 ;;
  1) export GAMMA=0.05 ;;   # reproduces the existing result -- keep as control
  2) export GAMMA=0.10 ;;
  3) export GAMMA=0.20 ;;
esac
export N_QUBITS=4 TVAL=3.0 K0=48 K_REF=48 ORDER=2
echo "gamma_sweep GAMMA=$GAMMA on $(hostname)"; echo "start: $(date)"
julia gamma_sweep.jl
echo "end: $(date)"
