#!/bin/bash
# Spectra of F, G, Xi, Psi, Phi + discarded-weight-vs-error.
# Run AFTER submit_comparison.sh (reads cmp_coefficients.csv). ~15 min.
#SBATCH --job-name=spectra
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/spectra_%j.out
#SBATCH --error=logs/spectra_%j.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export N_QUBITS=4 GAMMA=0.05 TVAL=3.0 K0=48 ORDER=2 KDIAG=8
echo "spectra_study on $(hostname)"; echo "start: $(date)"
julia spectra_study.jl
echo "end: $(date)"
