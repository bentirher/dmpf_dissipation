#!/bin/bash
# =============================================================================
# submit_spectra.sh
#
# Runs ONLY the spectra + weight-vs-error section. The maxdim sweep from job
# 6872636 already completed and wrote cmp_coefficients.csv and cmp_errors.csv;
# this reads the dc values from that file rather than recomputing them.
#
# Must be launched from the same directory as the earlier sweep, since Julia
# writes and reads CSVs relative to the WORKING directory:
#   /scratch/bentirher/bond_dim_study/oqs_generalization
#
#   sbatch submit_spectra.sh
# =============================================================================
#SBATCH --job-name=spectra
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/spectra_%j.out
#SBATCH --error=logs/spectra_%j.err

mkdir -p logs

module load Julia/1.11.6-linux-x86_64

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export N_QUBITS=4
export GAMMA=0.05
export TVAL=3.0
export K0=48
export ORDER=2
export KDIAG=8

echo "spectra_study on $(hostname)"
echo "start: $(date)"

julia spectra_study.jl

echo "end: $(date)"
