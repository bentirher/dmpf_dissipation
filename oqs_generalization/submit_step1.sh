#!/bin/bash
# STEP 1 -- the decisive measurement. RUN ONLY AFTER STEP 0 HAS PASSED.
#
# Does the DMPF observable, with coefficients computed at bond dimension chi,
# beat a direct classical simulation at the SAME chi? This is the L-MPF Fig. 4
# comparison (arXiv:2609.05024) transplanted from noise mitigation to Trotter
# error, and it is the claim that replaces the asymptotic-efficiency claim.
#
# n=8 is the default because it is the smallest size where the comparison is
# meaningful: the Liouville MPS ceiling is 4^4 = 256, so there is a real range
# of chi over which the state is genuinely truncated. At n=4 the ceiling is 16
# and there is almost nothing to measure. Note this is the same reason the MPO
# route cannot be used here at all: its ceiling at n=8 is 16^4 = 65536.
#
# Output: step1_observables.csv, step1_coefficients.csv
#SBATCH --job-name=step1_cost
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/step1_%j.out
#SBATCH --error=logs/step1_%j.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export N_QUBITS=8 GAMMA=0.05 TVAL=3.0 ORDER=2
export KS=3,8

# ---- SET THESE THREE FROM THE STEP 0 RECOMMENDATION -----------------------
# Step 0 prints them in the form:  export K0=... ORDER_REF=... SPLITTING_REF=...
export K0=24
export ORDER_REF=4
export SPLITTING_REF=strang
# Candidates stay on the existing composition; only the reference moves.
export SPLITTING=project
#
# Do NOT set SPLITTING_REF=project. It is first order once gamma > 0 (see
# symmetric_splitting.jl), so the reference will not be converged and every
# 'error' in the output becomes a distance to a wrong state. The script warns
# if you do it anyway.
# ---------------------------------------------------------------------------

export CHI_LIST=4,8,16,32,64,128,256   # capped internally at 4^(n/2)
# ---- cutoff -----------------------------------------------------------------
# ITensors' cutoff bounds the SQUARED discarded weight, so cutoff=1e-16 permits
# a state error of sqrt(1e-16) = 1e-8 PER TRUNCATION. That accumulated linearly
# in k0 and set the ~1e-5 floor in the previous run (strang:4 |1-Tr| doubling
# with every doubling of k0, reaching 1.06e-5 at k0=768). Leave this at
# round-off; raise it only to reproduce that floor deliberately.
export CUTOFF=1e-32

export EVO_MODE=gates                  # apply gate lists directly; do NOT use
                                       # :mpo for strang:4, whose 25 layers get
                                       # truncated at MPO_MAXDIM
export MPO_MAXDIM=512                  # only used when EVO_MODE=mpo

echo "step1_matched_cost on $(hostname)"; echo "start: $(date)"
julia step1_matched_cost.jl
echo "end: $(date)"

# ---------------------------------------------------------------------------
# SUGGESTED FOLLOW-UPS, in order of value:
#
#  1. gamma. The single most important robustness axis, and the one that broke
#     the previous result (Step 15: agreement held only at gamma=0.05). Run the
#     same script at GAMMA in {0.01, 0.05, 0.10, 0.20} as a job array. If the
#     advantage or the correlation corr_ref_kj degrades with gamma, the honest
#     claim narrows to the weak-dissipation regime -- which is still the
#     physically relevant one for device noise, but must be stated.
#
#  2. t. Truncation-error correlation should decay with evolution time. Run
#     TVAL in {1.5, 3.0, 6.0} to find where.
#
#  3. n. NOT for the comparison itself (which needs an exact reference, so
#     n <= 10), but to confirm the advantage is not shrinking with system size.
#     n=10 has ceiling 1024 and will need more memory and wall time.
#
#  4. the candidate family (Step 4 of the plan). ks=3,8 gives cond(N) ~ 800 and
#     a near rank-1 N. Try KS=3,6,12 and KS=2,4,8,16. Also worth testing the
#     ergodic-amplification analogue: candidates that differ in gamma rather
#     than in k, whose error vectors are genuinely non-parallel and would break
#     the rank-1 degeneracy that k-variation alone cannot.
# ---------------------------------------------------------------------------
