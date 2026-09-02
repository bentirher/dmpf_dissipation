#!/bin/bash
# Round 2. Three job types, because round 1 spent 12h per task chasing S_op
# (converged at maxdim 64) and chi_req (not converged at 256) on the same grid,
# and the only ladder that finished was gamma=0.2 -- the easy corner.
#
#   sbatch --array=0    submit_entanglement_growth.sh   # 0: Trotter check, run FIRST
#   sbatch --array=1-3  submit_entanglement_growth.sh   # 1-3: scaling grid
#   sbatch --array=4-6  submit_entanglement_growth.sh   # 4-6: chi_req ladders
#
# COST NOTE, since this is what killed round 1. TEBD cost goes as
# (steps) x n x chi^3. Round 1's gamma=0.2 n=20 run (500 steps, chi=256) took
# ~3.5 h, i.e. ~0.7 s per two-site SVD. Extrapolating: chi=512 is 8x that per
# SVD, and chi=2048 is 512x -- roughly 1800 h for a full n=24 sweep. Converging
# chi_req at n=20, gamma=0.05 is NOT affordable here, and that is itself the
# result: the honest deliverable is a rigorous lower bound, "chi > X", produced
# by the ladder jobs. The scaling jobs stay cheap on purpose.
#SBATCH --job-name=entgrow
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/entgrow_%A_%a.out
#SBATCH --error=logs/entgrow_%A_%a.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Shared: dt=0.05 replaces round 1's dt=0.02 (task 0 verifies this is safe;
# do not trust the scaling runs until it has). TMAX now scales with n, not with
# 1/gamma -- the barrier peaks at t ~ n/2, so TMAX_FACTOR=1.5 brackets it.
export DT=0.05 ORDER=2 CUTOFF=1e-12 JCOUP=0.5 DISORDER=false
export TMAX_FACTOR=1.5 NT=30

case $SLURM_ARRAY_TASK_ID in
  # --- 0: Trotter step check. Cheap, and everything else depends on it. ------
  0) export MODE=dt      N_LIST=12 GAMMA_LIST=0.05 MAXDIM=256
     export OUTDIR=r2_dtcheck TAG=dt ;;

  # --- 1-3: scaling grid, one gamma per task, n ladder inside ---------------
  # maxdim=256 is deliberate: S_op converged at 64 in round 1, so this is
  # ample for the physics result, and it keeps n=24 reachable. chi_req from
  # these runs is a lower bound only -- the `saturated` column says so.
  1) export MODE=scaling N_LIST=8,12,16,20,24 GAMMA_LIST=0.02 MAXDIM=256
     export OUTDIR=r2_scaling_g0p02 TAG=g0p02 ;;
  2) export MODE=scaling N_LIST=8,12,16,20,24 GAMMA_LIST=0.05 MAXDIM=256
     export OUTDIR=r2_scaling_g0p05 TAG=g0p05 ;;
  3) export MODE=scaling N_LIST=8,12,16,20,24 GAMMA_LIST=0.10 MAXDIM=256
     export OUTDIR=r2_scaling_g0p10 TAG=g0p10 ;;

  # --- 4-6: chi_req ladders at the interesting points -----------------------
  # n=8 is EXACT: the MPDO ceiling is 4^4 = 256, so maxdim 256 truncates
  # nothing at all. That run is the only rigorously converged chi_req in the
  # whole study and anchors the rest.
  4) export MODE=ladder  N_LIST=8  GAMMA_LIST=0.05 MAXDIM=256
     export OUTDIR=r2_ladder_n8  TAG=n8 ;;
  5) export MODE=ladder  N_LIST=12 GAMMA_LIST=0.05 MAXDIM=512
     export OUTDIR=r2_ladder_n12 TAG=n12 ;;
  6) export MODE=ladder  N_LIST=16 GAMMA_LIST=0.05 MAXDIM=512
     export OUTDIR=r2_ladder_n16 TAG=n16 ;;
esac

mkdir -p "$OUTDIR"
echo "task=$SLURM_ARRAY_TASK_ID MODE=$MODE N_LIST=$N_LIST GAMMA_LIST=$GAMMA_LIST MAXDIM=$MAXDIM"
echo "OUTDIR=$OUTDIR host=$(hostname) cwd=$(pwd)"
echo "start: $(date)"

for f in vectorized_evolution.jl closed_evolution.jl entanglement_growth_study.jl F_diagnostics.jl; do
  [ -f "$f" ] || { echo "FATAL: $f not found in $(pwd)" >&2; exit 2; }
done

julia entanglement_growth_study.jl
JULIA_STATUS=$?
echo "end: $(date)"

echo "--- CSVs written ---"
ls -la "$OUTDIR"
NCSV=$(find "$OUTDIR" -name '*.csv' | wc -l)
echo "csv count: $NCSV"

# Exit on Julia's status, NOT on ls's. In round 1 the last command was a
# successful `ls` on an empty directory, so SLURM reported COMPLETED for jobs
# that had died on an UndefVarError at load time.
if [ "$JULIA_STATUS" -ne 0 ]; then
  echo "FATAL: julia exited $JULIA_STATUS -- see the .err log" >&2
  exit "$JULIA_STATUS"
fi
# manifest.csv is written before any physics, so 1 CSV means Julia loaded and
# then died during the first evolution.
if [ "$NCSV" -le 1 ]; then
  echo "FATAL: julia exited 0 but produced no data beyond the manifest" >&2
  exit 3
fi
exit 0
