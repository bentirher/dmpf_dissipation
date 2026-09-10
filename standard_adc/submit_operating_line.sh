#!/bin/bash
# FULL RERUN of the operating line with the patched cost metrics.
#
#   sbatch --array=0-5 submit_operating_line.sh
#
# All six sizes run in PARALLEL as an array, so the wall clock is set by the
# slowest task (n=24, ~5 h), not by the ~10 h total. That is why a full rerun is
# worth it rather than patching only the point that needed it.
#
# WHAT THE RERUN BUYS (the first two are the real reasons):
#
# 1. THE COST FORMULA WAS WRONG, not just imprecise. An ensemble costs
#    sum_k chi_k^3 = N*<chi^3>, but the code compared against N*<chi>^3. For the
#    measured spread (lognormal sigma ~ 0.60, CV ~ 0.66) those differ by
#    exp(3 sigma^2) ~ 3x, which moves the crossover about 1.7 qubits EARLIER.
#    The runs now record <chi^3> directly instead of it being inferred.
#
# 2. NO ERROR BAR ON chi. The n=24 point's 16.5% uncertainty had to be
#    reverse-engineered from a lognormal fit to (mean, p95). chi_std and chi_sem
#    are now recorded, so the fit residuals can be weighted properly and
#    "does n=24 sit on the line?" becomes arithmetic rather than inference.
#
# 3. MEASURED WALL TIME per trajectory. <chi^3> is a third moment and is noisy
#    (relative error ~125% at Ntraj=16, ~31% at Ntraj=256), so it cannot be
#    trusted at small Ntraj. Elapsed time per trajectory measures the same cost
#    with no moment problem, and gives an independent extrapolation.
#
# NTRAJ is raised wherever the wave structure allows it. On 16 threads the wall
# clock is ceil(NTRAJ/16) waves, so 16, 32, 128, 256 are the only sensible
# values -- 48 costs the same as 64 did.
#SBATCH --job-name=opline
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=11:00:00
#SBATCH --output=logs/opline_%A_%a.out
#SBATCH --error=logs/opline_%A_%a.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64

# Many small independent problems: cores to Julia, BLAS pinned to 1.
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Identical to the runs being reproduced, so the new numbers are comparable
# with the old ones rather than merely similar.
export DT=0.05 CUTOFF=1e-10 JCOUP=0.5 SEED0=1000 TARGET_SEM=0.01
export TMAX_FACTOR=0.6 NT=20

# gamma* = 1.4/n exactly; CHI_MPDO from Table 5 of the report.
# Estimated wall clock from measured per-wave cost (0.62/1.20/2.33/5.25/41.8 min
# at n=10..20 for a wave of 16), times ceil(NTRAJ/16) waves.
case $SLURM_ARRAY_TASK_ID in
  0) export N=10 GAMMA=0.140000 NTRAJ=256 MAXDIM=512  CHI_MPDO=1.0e3
     export OUTDIR=op_n10 TAG=n10 ;;                       # ~10 min
  1) export N=12 GAMMA=0.116667 NTRAJ=256 MAXDIM=512  CHI_MPDO=6.5e3
     export OUTDIR=op_n12 TAG=n12 ;;                       # ~20 min
  2) export N=14 GAMMA=0.100000 NTRAJ=256 MAXDIM=1024 CHI_MPDO=4.2e4
     export OUTDIR=op_n14 TAG=n14 ;;                       # ~40 min
  3) export N=16 GAMMA=0.087500 NTRAJ=256 MAXDIM=1024 CHI_MPDO=5.24288e5
     export OUTDIR=op_n16 TAG=n16 ;;                       # ~1.5 h (was 128)
  4) export N=20 GAMMA=0.070000 NTRAJ=128 MAXDIM=1024 CHI_MPDO=8.388608e6
     export OUTDIR=op_n20 TAG=n20 ;;                       # ~5.6 h (was 64)
  # n=24: NTRAJ=32 is two waves at ~4.6 h each with TMAX_FACTOR=0.5, NT=12.
  # At TMAX_FACTOR=0.6 it would be ~11 h and would risk the wall, so this one
  # keeps the shorter window. The chi peak sits at t/t* = 0.93, well inside it.
  5) export N=24 GAMMA=0.058333 NTRAJ=32  MAXDIM=1024 CHI_MPDO=1.34217728e8
     export TMAX_FACTOR=0.5 NT=12
     export OUTDIR=op_n24 TAG=n24 ;;                       # ~9.2 h
esac

mkdir -p "$OUTDIR"
echo "task=$SLURM_ARRAY_TASK_ID N=$N GAMMA=$GAMMA NTRAJ=$NTRAJ MAXDIM=$MAXDIM TMAX_FACTOR=$TMAX_FACTOR"
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

# Exit on Julia's status, NOT on ls's.
if [ "$JULIA_STATUS" -ne 0 ]; then
  echo "FATAL: julia exited $JULIA_STATUS -- see the .err log" >&2
  exit "$JULIA_STATUS"
fi
if [ "$NCSV" -le 1 ]; then
  echo "FATAL: julia exited 0 but produced no data beyond the manifest" >&2
  exit 3
fi
exit 0
