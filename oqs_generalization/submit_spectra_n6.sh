#!/bin/bash
#SBATCH --job-name=spec_n6
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=logs/specn6_%j.out
#SBATCH --error=logs/specn6_%j.err

# THE PLATEAU TEST, and the measurement that has been outstanding since day one:
# does the effective rank of Phi GROW WITH n?
#
# At n=4, Phi needed 193 of a possible 256 singular values to retain 1-1e-3 of
# its Schmidt weight -- 75% of the ceiling, which is uninterpretable because the
# ceiling is tiny. n=6 has a ceiling of 4096. If truncated-Phi at n=6 already
# needs >>193, the rank is growing with n and there is no low-rank structure to
# exploit at any size. If it stays near 193, there is.
#
# This is also the direct test of the n=6 error plateau (dc stuck at 2-9e-3 over
# a 4x range in chi): if Phi's spectrum has a long flat tail, the discarded
# weight barely falls as chi rises, and a plateau in dc is exactly what you would
# see. The weight-vs-error table at the end of the run measures precisely that.
#
# MAXCHI=256 -- v1 hard-coded the ceiling (4096 at n=6): days, and OOM.
# Consequence: the spectra are those of the TRUNCATED Phi, so effective ranks are
# LOWER BOUNDS. That is sufficient for the question being asked.

mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export N_QUBITS=6 GAMMA=0.05 TVAL=3.0 K0=48 ORDER=2 KDIAG=8
export MAXCHI=256 TAG=n6

echo "spectra_study n=6 on $(hostname)"; echo "start: $(date)"
julia spectra_study_v2.jl || { echo "FAILED"; exit 1; }
echo "end: $(date)"
echo
echo "READ: eff_rank of Phi at 1-1e-3, against 193 at n=4 (ceiling 256)."
echo "  ~200      -> rank is BOUNDED. Low-rank structure exists. Good."
echo "  >>200     -> rank grows with n. No advantage at any size. Decisive."
echo "Also: does Phi's discarded weight fall as maxdim rises, or plateau?"
