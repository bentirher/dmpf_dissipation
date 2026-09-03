#!/bin/bash
# Round 2, revised after the task-0 Trotter check.
#
#   sbatch --array=0   ...   # Trotter ORDER check at n=8 (EXACT). Already run.
#   sbatch --array=1-3 ...   # scaling grid, one gamma per task  <- the deliverable
#   sbatch --array=4-6 ...   # chi_req ladders
#
# GOAL: pick (n, gamma, t) for the hardware runs. Tasks 1-3 give S_op(n,gamma,t)
# and the area-law onset n_sat(gamma), which is what decides whether a given
# hardware point is classically reachable. Tasks 4-6 bound the bond dimension a
# classical competitor would actually need at those points.
#
# WHY TASK 0 CHANGED. The n=12 dt check showed the S_op error falling by only
# ~2x per halving of dt at t=6 and t=9 (3.3x at t=3), i.e. FIRST-order
# behaviour, not the second order the scheme advertises. Two candidate causes:
#   (a) get_open_step_gates(...;order=2) composes odd(dt/2), even(dt), diss(dt),
#       odd(dt/2). The even and dissipator layers are multiplied as a bare
#       first-order product, leaving an uncancelled [C,B]dt^2 per step. With
#       gamma=0 this collapses to plain Strang, which is why it never showed up
#       in closed-system tests.
#   (b) that run had maxdim=256 binding at n=12, gamma=0.05, so truncation error
#       was mixed into the dt error.
# Task 0 now separates them: at n=8 the MPDO ceiling is 4^4 = 256, so MAXDIM=256
# truncates NOTHING and the measured exponent is unambiguous. It also runs the
# corrected symmetric splitting (SPLITTING=strang) side by side at identical dt.
#
# COST. TEBD goes as (steps) x n x chi^3. Measured on this cluster: ~0.24 s per
# two-site SVD at chi=256 (from the 66-min task-0 run: 765 steps, n=12). The
# scaling tasks at MAXDIM=256 come to roughly 5.5 h each; the ladders were
# retuned down from ~19-26 h (over the limit) by capping the top rung at 512 and
# dropping the 1.75*t_peak sample point.
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

# DT=0.05 kept for the production runs. The measured error at dt=0.05 relative
# to dt=0.02 is 0.02% / 0.16% / 0.50% of S_op at t = 3 / 6 / 9. The absolute
# threshold of 1e-3 bits quoted earlier was the wrong yardstick for a quantity
# of order 5 bits; sub-1% is what the peak-height and n-scaling results need.
# Revisit if task 0 shows the scheme is first order AND you switch to :strang.
export DT=0.05 ORDER=2 CUTOFF=1e-12 JCOUP=0.5 DISORDER=false
export TMAX_FACTOR=1.5 NT=30

# SPLITTING=strang, decided by the n=8 exact order check (task 0, first run).
# Effective order from successive differences, measured with truncation
# identically zero:
#     t          2     4     6    10
#   :project    1.6   1.6   2.3   sign flip
#   :strang     2.0   2.0   2.0   2.1
# At t=4, :strang at dt=0.05 is more accurate than :project at dt=0.025 while
# taking a quarter of the steps. :strang costs ~33% more per step (one extra
# even layer), so it is a net ~2.6x at fixed accuracy, and it has a defensible
# error bar because its order is actually the order it claims.
export SPLITTING=strang

# ---------------------------------------------------------------------------
# DT AND THE HARDWARE RUN -- read this before quoting these numbers as the
# classical baseline for a circuit experiment.
#
# DT=0.05 was chosen so the CLASSICAL integrator is converged: it approximates
# the continuous-time Lindblad dynamics to <0.2% in S_op. That is the right
# choice for "how entangled does the master equation get".
#
# It is NOT the right choice for "can a classical machine reproduce my circuit".
# The hardware will run a Trotterized circuit at whatever step its gate budget
# allows -- likely far coarser than 0.05. A coarse Trotterization is a DIFFERENT
# map from the master equation, with its own entanglement growth, and it is that
# map a classical competitor would have to match. Once the hardware Trotter step
# is fixed, rerun tasks 1-3 with DT set to it and TMAX_FACTOR chosen so
# TMAX = (number of Trotter steps) x DT.
# ---------------------------------------------------------------------------

case $SLURM_ARRAY_TASK_ID in
  # --- 0: Trotter ORDER at n=8, exact (ceiling 4^4 = 256). Cheap, decisive. --
  0) export MODE=order   N_LIST=8 GAMMA_LIST=0.05 MAXDIM=256 ORDER=2
     export OUTDIR=r2_order TAG=ord ;;


  # --- 1-3: scaling grid. MAXDIM=256 is ample: S_op was converged at 64 in
  # round 1. chi_req from these is a lower bound only; the `saturated` column
  # marks every row where maxdim binds.
  1) export MODE=scaling N_LIST=8,12,16,20,24 GAMMA_LIST=0.02 MAXDIM=256
     export OUTDIR=r2_scaling_g0p02 TAG=g0p02 ;;
  2) export MODE=scaling N_LIST=8,12,16,20,24 GAMMA_LIST=0.05 MAXDIM=256
     export OUTDIR=r2_scaling_g0p05 TAG=g0p05 ;;
  3) export MODE=scaling N_LIST=8,12,16,20,24 GAMMA_LIST=0.10 MAXDIM=256
     export OUTDIR=r2_scaling_g0p10 TAG=g0p10 ;;

  # --- 4-6: chi_req ladders. Task 4 is the anchor: at n=8 the ladder tops out
  # at the exact ceiling, so its chi_req is the one rigorously converged number
  # in the whole study. Tasks 5-6 give honest lower bounds.
  4) export MODE=ladder  N_LIST=8  GAMMA_LIST=0.05 MAXDIM=256
     export OUTDIR=r2_ladder_n8  TAG=n8 ;;
  5) export MODE=ladder  N_LIST=12 GAMMA_LIST=0.05 MAXDIM=256
     export OUTDIR=r2_ladder_n12 TAG=n12 ;;
  6) export MODE=ladder  N_LIST=16 GAMMA_LIST=0.05 MAXDIM=256
     export OUTDIR=r2_ladder_n16 TAG=n16 ;;
esac

mkdir -p "$OUTDIR"
echo "task=$SLURM_ARRAY_TASK_ID MODE=$MODE N_LIST=$N_LIST GAMMA_LIST=$GAMMA_LIST MAXDIM=$MAXDIM SPLITTING=$SPLITTING"
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

# Exit on Julia's status, NOT on ls's.
if [ "$JULIA_STATUS" -ne 0 ]; then
  echo "FATAL: julia exited $JULIA_STATUS -- see the .err log" >&2
  exit "$JULIA_STATUS"
fi
# manifest.csv is written before any physics, so <=1 CSV means Julia loaded and
# then died during the first evolution.
if [ "$NCSV" -le 1 ]; then
  echo "FATAL: julia exited 0 but produced no data beyond the manifest" >&2
  exit 3
fi
exit 0
