#!/bin/bash
# FAMILY SCAN -- run this FIRST, before step2. Minutes, not hours.
#
# Step 1 showed err_proj = 5.0e-3 (the DMPF error with PERFECT coefficients)
# while the classical route reached 1.5e-3 at chi=128. err_proj is a floor no
# bond dimension can cross, so that comparison was lost before it started.
#
# The floor was not fundamental. ks=[3,8] is two candidates (so dc has exactly
# ONE available direction, making the L-MPF Appendix D argument vacuous), on a
# :project formula that Step 0 measured at p_eff = 1.00, with k=3 meaning
# dt = 1.0 and a ~43% state error. This scan scores several families directly
# instead of guessing a replacement.
#
# Output: family_scan.csv + a printed KS= recommendation for submit_step2.sh.
#SBATCH --job-name=family_scan
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/family_%j.out
#SBATCH --error=logs/family_%j.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export N_QUBITS=6 GAMMA=0.05 TVAL=3.0
export K0=48 ORDER_REF=4 SPLITTING_REF=strang
export ORDER=2 SPLITTING=strang     # candidates now on the SYMMETRIC formula
export EVO_MODE=gates CUTOFF=1e-32

# K0=48 is divisible by lcm of every family below, so all of them stay
# cross-checkable against the MOC/N-route (which needs k0 % k_j == 0).
export FAMILIES="3,8;4,12;4,8,16;4,6,8,12;6,8,12,16;2,4,8,16;4,6,8,12,16,24"

echo "family_scan on $(hostname)"; echo "start: $(date)"
julia family_scan.jl
echo "end: $(date)"

# Choose on err_proj, subject to r>=3 and cond(N) under ~1e4. Then confirm at
# n=8 (N_QUBITS=8 sbatch submit_family_scan.sh) before committing to step2 --
# err_proj should be nearly n-independent, and if it isn't, that is itself
# important to know.
