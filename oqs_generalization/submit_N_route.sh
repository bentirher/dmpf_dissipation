#!/bin/bash
# =============================================================================
# submit_N_route.sh   --   EXPERIMENTS 2 and 3
#
# Run AFTER submit_gauge_diagnostic.sh has reported.
#
# Task 0 is deliberately tiny and short: a 10-minute `test`-QoS-sized run at
# n=4 with STEPS=1 only. Read its output before committing to the rest.
# Repeatedly doubling --time and hoping has wasted days before; a short
# instrumented run first gives the answer (findings_thematic.md section 5).
#
#   task 0 : n=4, STEPS=1    -- the norm hierarchy / bond dimension diagnostic
#   task 1 : n=4, STEPS=123  -- full pipeline at a size where the oracle is cheap
#   task 2 : n=5, STEPS=123  -- the study size
#
#   sbatch submit_N_route.sh
#   tail -f logs/nroute_*_0.out
#   # after: sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode ; seff <jobid>
# =============================================================================
#SBATCH --job-name=n_route
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0
#SBATCH --output=logs/nroute_%A_%a.out
#SBATCH --error=logs/nroute_%A_%a.err

mkdir -p logs

module load Julia/1.11.6-linux-x86_64

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

case $SLURM_ARRAY_TASK_ID in
  0) export N_QUBITS=4 MAXDIM=256  MAXDIM_G=256 STEPS=1   ;;
  1) export N_QUBITS=4 MAXDIM=64  MAXDIM_G=32 STEPS=123 ;;
  2) export N_QUBITS=5 MAXDIM=128 MAXDIM_G=48 STEPS=123 ;;
esac

export GAMMA=0.05
export TVAL=3.0
export K0=48                 # multiple of both k_j = 3 and 8
export ORDER=2
export K_REF_ORACLE=200

echo "run_N_route  n=$N_QUBITS  maxdim=$MAXDIM  maxdim_G=$MAXDIM_G  steps=$STEPS  on $(hostname)"
echo "start: $(date)"

julia run_N_route.jl

echo "end: $(date)"
