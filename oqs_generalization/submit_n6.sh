#!/bin/bash
#SBATCH --job-name=n6
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --output=logs/n6_%j.out
#SBATCH --error=logs/n6_%j.err

# n=6, everything else IDENTICAL to the n=4 runs. Uses comparison_study_v3.jl.
#
# WHY v3 AND NOT v2. Two places in v2 assume the MPO ceiling is reachable:
#   1. line 134  ex_lag = m_route(chi)  -- at n=6, chi = 16^3 = 4096. That single
#      line would launch a maxdim=4096 M-route run: days of wall time, and it
#      would OOM long before finishing.
#   2. spectra_study.jl line 40 does the same thing (chi = theoretical_max_bond_dim)
#      and builds F and Phi at maxdim=4096. It is therefore NOT run here.
# v3 caps the MPO grid at MAXCHI (default 256) and anchors the v1 reference to
# the top of the grid instead of the ceiling.
#
# THE CEILINGS DIVERGE AT n=6, which is the entire point of this run:
#     MPS ceiling (oracle)  4^3 =   64   <- still exact, still cheap
#     MPO ceiling (routes) 16^3 = 4096   <- unreachable
# So unlike n=4, there is NO untruncated MPO point. The oracle is the only exact
# object in the run. That is precisely the regime the method is supposed to be
# for, and the reason n=4 could never settle anything.
#
# MEMORY 180G, not 64G: exact_delta carries chi(A)+chi(B), and at n=6 there are
# 6 Liouville sites rather than 4. If it OOMs, drop to MAXCHI=128 rather than
# cutting memory further.
#
# TIME. n=4 at maxdim=256 cost ~940 s for the N-route. Cost is roughly linear in
# the number of sites at fixed chi, so expect ~1.5x, i.e. ~40 min for the top
# point and ~1.5 h for the sweep. 16 h is deliberate margin: the chi-scaling
# exponent was fitted at n=4 (t_N ~ chi^1.88) and may be worse here.
#
# ONE CHANGE AT A TIME: K0 stays at 48 so this is directly comparable to the n=4
# data. The k0=192 test is a SEPARATE job at n=4 -- see submit_k0.sh.

mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export N_QUBITS=6 GAMMA=0.05 TVAL=3.0 K0=48 K_REF=48 ORDER=2
export MAXCHI=256 GSCAN=0 TAG=n6

echo "n=6 comparison on $(hostname)"
echo "start: $(date)"

julia comparison_study_v3.jl || { echo "FAILED"; exit 1; }

echo "end: $(date)"
echo
echo "wrote: cmp_coefficients_n6.csv  cmp_errors_n6.csv"
echo
echo "THE THREE NUMBERS THAT MATTER:"
echo "  1. 'drift' in the GROUND TRUTH block. The oracle must still saturate"
echo "     (maxdim 64 vs 128). If it does not, nothing else in the run is usable."
echo "  2. 'signal'. At n=4 it was 1.4287e-02. If it GROWS with n, the free"
echo "     formula degrades and the method has more to offer. If it SHRINKS,"
echo "     the value proposition of DMPF itself weakens -- that is the open risk"
echo "     flagged back in check 0."
echo "  3. Whether ANY maxdim gets dc below the signal. At n=4 the N-route"
echo "     managed it from 32 upward; here every point is truncated."
