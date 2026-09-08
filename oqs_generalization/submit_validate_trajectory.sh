#!/bin/bash
# Localises the -1.3 sigma discrepancy in the n=8 trajectory validation.
#
#   sbatch submit_validate_trajectory.sh
#
# Short and cheap -- it can also just be run interactively on a login node
# (`julia --threads=8 validate_trajectory.jl`). It exists as a batch job only so
# it can use a full node's threads for Test B.
#
# WHY THIS RATHER THAN MORE TRAJECTORIES. Two independent 400-trajectory runs
# agreed with the exact MPDO curve at -0.92 and -0.97 sigma, with 27 of 30
# deviations negative. Suggestive, not significant. Brute-forcing it to 3 sigma
# needs ~1e4 trajectories (~12 h) and would only tell you WHETHER, not WHERE.
# These two tests each have zero systematic error of their own and between them
# cover every moving part:
#
#   TEST A  gamma = 0  -> the trajectory is DETERMINISTIC (every reset outcome
#           is 0 with probability 1). Two seeds must agree bit for bit, and both
#           must match the closed-system MPS. Zero statistics. Covers the
#           Hamiltonian layers, the q-a-q-a interleaving, the non-adjacent
#           two-site gates that ITensor implements by swapping through the
#           ancilla, and the Strang ordering.
#
#   TEST B  J = 0      -> the qubits decouple into independent single-qubit
#           amplitude-damping processes with the analytic answer
#           <Z_j>(t) = 1 - 2 exp(-gamma t). Covers the Born sampling, the reset
#           and the channel angle. Cheap (chi = 1) and sharp (equivalent sites
#           are pooled, so N_eff = 4 * NREP), and it is run at two step sizes:
#           with J = 0 there is no Trotter error, so dt=0.5 and dt=0.05 must
#           agree exactly. A bias present at dt=0.05 but not dt=0.5 accumulates
#           PER STEP; one that is equal at both is per jump.
#
# READING THE RESULT
#   A pass, B pass -> the -1.3 sigma was a fluctuation; run --array=1-3 of
#                     submit_trajectory_study.sh and stop worrying.
#   A pass, B fail -> the bias is in reset_ancilla! or the sampling.
#   A fail         -> the bias is in the gate layers or the layout, and the
#                     stochastic agreement was masking it.
#SBATCH --job-name=trajval
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/trajval_%j.out
#SBATCH --error=logs/trajval_%j.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64

# Same threading rationale as the trajectory production runs: many small
# independent problems, so the cores go to Julia and BLAS is pinned to 1.
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Trajectories for Test B. 6000 pools to N_eff = 24000 per group, i.e.
# SEM ~ 0.006 -- enough to resolve the ~0.02 discrepancy at >3 sigma.
# Test A uses none (it is deterministic).
export NREP=${NREP:-6000}

echo "host=$(hostname) cwd=$(pwd) threads=$JULIA_NUM_THREADS NREP=$NREP"
echo "start: $(date)"

[ -f trajectory_evolution.jl ] || { echo "FATAL: trajectory_evolution.jl not found in $(pwd)" >&2; exit 2; }
[ -f validate_trajectory.jl ]  || { echo "FATAL: validate_trajectory.jl not found in $(pwd)"  >&2; exit 2; }
# Optional: the second half of Test A needs the MPDO study's closed-system
# reference. Copy vectorized_evolution.jl, closed_evolution.jl and the project
# chain (F_diagnostics.jl and what it includes) into this directory to enable
# it. The script detects them and degrades gracefully if they are absent.
if [ -f closed_evolution.jl ] && [ -f vectorized_evolution.jl ] && [ -f F_diagnostics.jl ]; then
  echo "closed-system reference present: full Test A"
else
  echo "NOTE: closed reference absent -- Test A runs its seed-independence half only."
  echo "      To enable the rest, copy vectorized_evolution.jl, closed_evolution.jl"
  echo "      and the project include chain (F_diagnostics.jl -> ... -> liouville_space.jl)"
  echo "      into $(pwd)."
fi

julia --threads="$JULIA_NUM_THREADS" validate_trajectory.jl
JULIA_STATUS=$?
echo "end: $(date)"

# Exit on Julia's status. (An earlier round of this study reported COMPLETED for
# jobs that had died at load time, because the last command was a successful ls.)
if [ "$JULIA_STATUS" -ne 0 ]; then
  echo "FATAL: julia exited $JULIA_STATUS -- see the .err log" >&2
  exit "$JULIA_STATUS"
fi
exit 0
