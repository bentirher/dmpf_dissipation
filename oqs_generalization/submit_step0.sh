#!/bin/bash
# STEP 0 -- converge the k0 reference. RUN THIS FIRST, BEFORE EVERYTHING ELSE.
#
# Fixes README open issue 3 ("the reference is not converged: k0 48->96 shifts
# E_k8 by +6.3%"). Every 'exact' number downstream is scored against this state,
# so until it is converged, 'error' means 'distance to a slightly wrong answer'.
#
# Pure state route: no MPOs beyond the step channel, no MOC. At n=6 the Liouville
# MPS ceiling is 64 and every run below is UNTRUNCATED, so this is minutes-to-an-
# hour, not the multi-hour MOC runs.
#
# Output: step0_reference.csv, plus a printed (ORDER_REF, K0) recommendation
# that you feed straight into submit_step1.sh.
#SBATCH --job-name=step0_ref
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/step0_%j.out
#SBATCH --error=logs/step0_%j.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64
# BOTH are required: JULIA_NUM_THREADS alone does NOT control BLAS, and unpinned
# OpenBLAS spawns ~64 threads onto the allocation.
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export N_QUBITS=6 GAMMA=0.05 TVAL=3.0 ORDER=2
export KS=3,8
export ORDERS_REF=2,4      # tests whether an order-4 reference converges at a
                           # much smaller k0 -- if it does, every Step 1 run
                           # gets cheaper by that factor, for free
export N_LADDER=6          # k0 in {24,48,96,192,384,768} for ks=3,8
export REL_TOL=1e-3

echo "step0_reference_convergence on $(hostname)"; echo "start: $(date)"
julia step0_reference_convergence.jl
echo "end: $(date)"

# ---------------------------------------------------------------------------
# AFTER IT PASSES: rerun at the n you will actually use in Step 1, to confirm
# the recommended k0 does not drift with system size. Trotter convergence is
# essentially n-independent for a local model, so this is a check, not a
# re-derivation, and it is worth the 1-2 h:
#
#   N_QUBITS=8 N_LADDER=5 sbatch submit_step0.sh
#
# If NO k0 on the ladder passes, raise N_LADDER and rerun. Do not proceed to
# Step 1 until one does: the whole point of Step 1 is a factor-of-a-few
# comparison, and a 6% systematic under the reference can manufacture or
# destroy it outright.
# ---------------------------------------------------------------------------
