#!/bin/bash
# M-route vs N-route at n=4, scored against the exact (untruncated) point.
# Produces cmp_coefficients.csv and cmp_errors.csv. ~1-2 h.
# n=4 ONLY: at n=5 the ceiling is also 256 but M(256) costs 426-988 s per call.
#SBATCH --job-name=cmp_MN
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/cmp_%j.out
#SBATCH --error=logs/cmp_%j.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export N_QUBITS=4 GAMMA=0.05 TVAL=3.0 K0=48 K_REF=48 ORDER=2
# K_REF MUST equal K0 -- otherwise the routes target different references and
# the comparison mixes truncation with a systematic offset.
echo "comparison_study on $(hostname)"; echo "start: $(date)"
julia comparison_study.jl
echo "end: $(date)"
