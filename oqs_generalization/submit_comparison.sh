#!/bin/bash
# =============================================================================
# submit_comparison.sh
#
# Head-to-head M-route vs N-route at n=4, scored against the EXACT point
# (maxdim = 256 = 16^min(2,2), no truncation). Produces the four CSVs behind
# the replacement figures for deck slides 23 and 26.
#
# n=4 only: this is the largest size where ground truth is free. Do NOT run
# this at n=5 -- there maxdim=256 is the ceiling too, but M(256) costs 426-988 s
# per call and the sweep would take many hours for no extra insight.
#
#   sbatch submit_comparison.sh
#   tail -f logs/cmp_*.out
# =============================================================================
#SBATCH --job-name=cmp_MN
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cmp_%j.out
#SBATCH --error=logs/cmp_%j.err

mkdir -p logs

module load Julia/1.11.6-linux-x86_64

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export N_QUBITS=4
export GAMMA=0.05
export TVAL=3.0
export K0=48
export K_REF=48          # matched to K0 so both routes target the SAME reference
export ORDER=2

echo "comparison_study on $(hostname)"
echo "start: $(date)"

julia comparison_study.jl

echo "end: $(date)"
