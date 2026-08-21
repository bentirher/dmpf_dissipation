#!/bin/bash
# =============================================================================
# submit_systematic.sh   --   PRIORITY 1
#
# Isolates the dc = 9.36e-4 systematic that survives at maxdim=256 where
# nothing is truncated. Six independent tests; the first that fails is the
# cause. T1-T4 are cheap; T5/T6 run the full N at maxdim=256 five times, which
# is the bulk of the runtime.
#
#   sbatch submit_systematic.sh
#   tail -f logs/sysdiag_*.out
# =============================================================================
#SBATCH --job-name=sysdiag
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/sysdiag_%j.out
#SBATCH --error=logs/sysdiag_%j.err

mkdir -p logs

module load Julia/1.11.6-linux-x86_64

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

export N_QUBITS=4
export GAMMA=0.05
export TVAL=3.0
export K0=48
export ORDER=2
export NSMALL=3      # size for the T4 identity check (ceiling 16, exact, fast)

echo "systematic_diagnostic on $(hostname)"
echo "start: $(date)"

julia systematic_diagnostic.jl

echo "end: $(date)"
