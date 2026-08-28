#!/bin/bash
#SBATCH --job-name=n6
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=20:00:00
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
# TIME. Measured at n=6 in job 6995914: t_N = 256, 752, 2178, 7091, 20617 s for
# chi = 48, 64, 96, 128, 192, plus t_M = 108, 255, 615, 1859, 5029. Summing the
# GRID below gives 12.2 h including the v1-reference m_route(192) run. Hence
# 20 h, not 12: the n=4 repeat showed ~40% node-to-node variation, which would
# put this at ~17 h on a slow node. (My earlier 16 h estimate assumed
# t_N ~ chi^1.88 from n=4; the real n=6 exponent is 3.08, which is why chi=256
# overran.)
#
# ONE CHANGE AT A TIME: K0 stays at 48 so this is directly comparable to the n=4
# data. The k0=192 test is a SEPARATE job at n=4 -- see submit_k0.sh.

mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export N_QUBITS=6 GAMMA=0.05 TVAL=3.0 K0=48 K_REF=48 ORDER=2
export GSCAN=0 TAG=n6

# GRID overrides the default [16,32,48,64,96,128,192,256] entirely.
#   chi=16 and 32 gave dc = 2.9e-1 and 3.5e-2 against a signal of 2.3e-2, i.e.
#     both lose to the free formula. No information, ~2 min. Dropped.
#   chi=256 alone needs ~13.6 h by the fitted t_N ~ chi^3.08. That is what killed
#     job 6995914 at the 16 h limit. Dropped.
#   chi=48 is where chi_break sat, so the grid is centred there.
export GRID=48,64,96,128,192

# CUTOFF_ORACLE: the oracle drifted 4.7e-4 between maxdim 64 and 128 in job
# 6995914, which cannot be a bond-dimension effect (MPS ceiling is 4^3 = 64).
# SET THIS FROM THE oracle_convergence.jl TABLE before trusting the run. 0.0
# disables cutoff-based truncation for the oracle and leaves maxdim in charge;
# the oracle is cheap enough to afford it. The routes keep CUTOFF=1e-14.
export CUTOFF_ORACLE=0.0

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
