#!/bin/bash
# Trajectory (dynamic-circuit) half of the classical-hardness study.
#
#   sbatch --array=0   submit_trajectory_study.sh   # n=8 VALIDATION. Run first.
#   sbatch --array=1-3 submit_trajectory_study.sh   # the operating line
#   sbatch --array=4-5 submit_trajectory_study.sh   # where do trajectories break?
#
# WHAT THIS DECIDES. The MPDO study showed the vectorised route needs
# chi ~ 1e7 at the n=24 operating point. That bounds one classical method. This
# measures the other, and prints the head-to-head:
#
#       advantage  =  chi_MPDO^3  /  ( N_traj * chi_traj^3 )
#
# If the advantage exceeds 1 -- which is the likely outcome, since the no-jump
# part of amplitude damping is a product operator and barely touches
# entanglement -- then "classically hard" is NOT established at these parameters
# and the operating line has to move out to where both methods fail. Tasks 4-5
# are there to find that point.
#SBATCH --job-name=mctraj
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mctraj_%A_%a.out
#SBATCH --error=logs/mctraj_%A_%a.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64

# THREADING IS DIFFERENT HERE, on purpose.
# The MPDO runs were one big linear-algebra problem, so BLAS got all the cores.
# Trajectories are embarrassingly parallel and each one is small, so the cores
# go to Julia and BLAS is pinned to 1. Leaving OPENBLAS_NUM_THREADS at 16 while
# Julia also spawns 16 threads oversubscribes the node by 256x and is slower
# than running serial.
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Shared. dt and the Trotter layer order match the :strang MPDO runs exactly,
# so trajectory and MPDO numbers differ only in the unravelling.
# TMAX_FACTOR=0.6 brackets the barrier peak at t = 0.45n with margin.
export DT=0.05 CUTOFF=1e-10 JCOUP=0.5 TMAX_FACTOR=0.6 NT=20 SEED0=1000
export TARGET_SEM=0.01

# NTRAJ=200 is deliberate and is NOT the N used in the cost formula. The chi
# statistics (mean, p95) converge in ~100 trajectories; the number needed for
# the <Z> error bar is EXTRAPOLATED from the measured per-trajectory variance by
# Ntraj_for(), so there is no reason to actually run 1e4 of them.

case $SLURM_ARRAY_TASK_ID in
  # --- 0: VALIDATION. n=8, gamma=0.05 matches open_n8_g0p050_*.csv from the
  # MPDO sweep, which is EXACT there (MPDO ceiling 4^4 = 256 = maxdim). If
  # <Z_mid>(t) agrees with that curve inside the SEM, the dynamic circuit and
  # the master equation are the same physics. Nothing downstream is worth
  # running until this passes.
  0) export N=8 GAMMA=0.05 NTRAJ=400 MAXDIM=256 NT=15 TMAX_FACTOR=1.5
     export CHI_MPDO=256 OUTDIR=traj_validate_n8 TAG=val ;;

  # --- 1-3: the operating line, gamma* = 1.4/n, t* = 0.45n.
  # CHI_MPDO values are from Table 5 of the report (extrapolated along the same
  # line); pass measured ones instead once the maxdim ladders have run.
  1) export N=16 GAMMA=0.0875 NTRAJ=200 MAXDIM=512
     export CHI_MPDO=524288      OUTDIR=traj_n16 TAG=n16 ;;   # 2^19
  2) export N=20 GAMMA=0.0700 NTRAJ=200 MAXDIM=512
     export CHI_MPDO=8388608     OUTDIR=traj_n20 TAG=n20 ;;   # 2^23
  3) export N=24 GAMMA=0.0583 NTRAJ=200 MAXDIM=512
     export CHI_MPDO=134217728   OUTDIR=traj_n24 TAG=n24 ;;   # 2^27

  # --- 4-5: push out along the line until the TRAJECTORY method breaks too.
  # Rough expectation: trajectory entropy tracks the closed-system value,
  # S_vN ~ 0.7 t, so chi_traj becomes infeasible around S_vN ~ 30 bits, i.e.
  # t ~ 43, i.e. n ~ 95. These two bracket the approach to that.
  # MAXDIM=1024 here: expect saturation, and read chi_traj as a lower bound.
  4) export N=32 GAMMA=0.0438 NTRAJ=100 MAXDIM=1024
     export CHI_MPDO=1.7e10      OUTDIR=traj_n32 TAG=n32 ;;
  5) export N=40 GAMMA=0.0350 NTRAJ=64  MAXDIM=1024
     export CHI_MPDO=1.4e13      OUTDIR=traj_n40 TAG=n40 ;;
esac

mkdir -p "$OUTDIR"
echo "task=$SLURM_ARRAY_TASK_ID N=$N GAMMA=$GAMMA NTRAJ=$NTRAJ MAXDIM=$MAXDIM"
echo "OUTDIR=$OUTDIR threads=$JULIA_NUM_THREADS host=$(hostname) cwd=$(pwd)"
echo "start: $(date)"

for f in trajectory_evolution.jl trajectory_study.jl; do
  [ -f "$f" ] || { echo "FATAL: $f not found in $(pwd)" >&2; exit 2; }
done

julia --threads="$JULIA_NUM_THREADS" trajectory_study.jl
JULIA_STATUS=$?
echo "end: $(date)"

echo "--- CSVs written ---"
ls -la "$OUTDIR"
NCSV=$(find "$OUTDIR" -name '*.csv' | wc -l)
echo "csv count: $NCSV"

# Exit on Julia's status, NOT on ls's. An earlier round of this study reported
# COMPLETED for jobs that had died at load time, because the last command in the
# script was a successful `ls` on an empty directory.
if [ "$JULIA_STATUS" -ne 0 ]; then
  echo "FATAL: julia exited $JULIA_STATUS -- see the .err log" >&2
  exit "$JULIA_STATUS"
fi
# manifest.csv is written before any physics, so <=1 CSV means Julia loaded and
# then died during the first trajectory.
if [ "$NCSV" -le 1 ]; then
  echo "FATAL: julia exited 0 but produced no data beyond the manifest" >&2
  exit 3
fi
exit 0
