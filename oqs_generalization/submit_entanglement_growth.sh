#!/bin/bash
# Locates the classically expensive corner of (n, gamma, t) for the vectorized
# MPS/TEBD route, to motivate the parameters of the ancilla-based quantum
# circuit experiment. Job array over gamma; one CSV per task.
#
# TMAX is set per gamma to roughly 2/gamma, so the operator-entanglement barrier
# peak (near t ~ 1/gamma) sits inside the window instead of just past its edge.
# Smaller gamma therefore means a longer, more expensive run -- that is the
# physics, not a scheduling accident, and the walltimes below reflect it.
#SBATCH --job-name=entgrow
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/entgrow_%A_%a.out
#SBATCH --error=logs/entgrow_%A_%a.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

case $SLURM_ARRAY_TASK_ID in
  0) export GAMMA=0.01 TMAX=200.0 NT=50 MAXDIM=1024 ;;  # tallest barrier, most expensive
  1) export GAMMA=0.05 TMAX=40.0  NT=40 MAXDIM=512  ;;  # the project's current gamma
  2) export GAMMA=0.10 TMAX=20.0  NT=40 MAXDIM=512  ;;
  3) export GAMMA=0.20 TMAX=10.0  NT=40 MAXDIM=256  ;;  # barrier already suppressed
esac

export N_LIST=8,12,16,20
export DT=0.02 ORDER=2 CUTOFF=1e-12 JCOUP=0.5 DISORDER=false

# One directory per gamma. Every CSV row carries its own n/gamma/maxdim/cutoff/dt,
# so the combined files across tasks can be concatenated afterwards with
#   head -1 results_g0p050/entanglement_growth_*.csv > all_timeseries.csv
#   tail -q -n +2 results_g*/entanglement_growth_g*.csv >> all_timeseries.csv
export OUTDIR=results_g$(echo $GAMMA | tr '.' 'p')
mkdir -p "$OUTDIR"

echo "entanglement_growth GAMMA=$GAMMA N_LIST=$N_LIST OUTDIR=$OUTDIR on $(hostname)"
echo "start: $(date)"
echo "cwd: $(pwd)"
# Sanity: the new files must sit next to the project's F_diagnostics.jl, because
# Julia resolves `include` relative to the including file's directory.
for f in vectorized_evolution.jl entanglement_growth_study.jl F_diagnostics.jl; do
  [ -f "$f" ] || { echo "FATAL: $f not found in $(pwd)" >&2; exit 2; }
done

julia entanglement_growth_study.jl
JULIA_STATUS=$?
echo "end: $(date)"

echo "--- CSVs written ---"
ls -la "$OUTDIR"
NCSV=$(find "$OUTDIR" -name '*.csv' | wc -l)
echo "csv count: $NCSV"

# Exit on Julia's status, NOT on ls's. Without this the job is reported
# COMPLETED even when Julia died on its first line, because the last command in
# the script was a successful `ls` on an empty directory.
if [ "$JULIA_STATUS" -ne 0 ]; then
  echo "FATAL: julia exited $JULIA_STATUS -- see the .err log" >&2
  exit "$JULIA_STATUS"
fi
if [ "$NCSV" -eq 0 ]; then
  echo "FATAL: julia exited 0 but wrote no CSVs" >&2
  exit 3
fi
exit 0
