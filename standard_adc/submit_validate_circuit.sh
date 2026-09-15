#!/bin/bash
# STEP 1 of the hardware-aware study: does the Julia simulator run YOUR circuit?
#
#   sbatch submit_validate_circuit.sh
#
# Cheap (n=4) -- it can equally be run on a login node with
#   julia --threads=8 validate_circuit_native.jl
# It exists as a job only so Check B can use a full node for its 2000
# trajectories.
#
# WHY THIS STEP EXISTS. Four things in hardware_aware_circuit.py differ from the
# code the earlier hardness study was built on, and each one silently changes
# the answer:
#   1. Your rxx angle is 2*J*t/k; the old code used J*dt, i.e. HALF. Every time
#      quoted in the earlier report corresponds to t/2 of yours.
#   2. Your circuit is first-order Trotter (odd, even, damp); the old code used
#      a symmetric Strang splitting. Different circuits, different entanglement
#      per step.
#   3. Your initial state is a single excitation on q0; the old runs used Neel,
#      which is far more entangling.
#   4. Your ancilla shuttling is exact and channel-equivalent, so the SWAPs are
#      omitted here -- they cost hardware depth, not entanglement.
# circuit_native.jl is parameterised by the GATE ANGLE theta and the per-step
# jump probability p, never by (J, dt), so mismatch 1 cannot recur.
#
# WHAT TO DO WITH THE OUTPUT
#   A: validate_mpdo_population.csv holds pop(q0) vs t for k = 2, 8, 15 --
#      overlay it on `all_trotter` in ha_circuit_study.ipynb. At n=4 the MPDO is
#      EXACT (Liouville ceiling 4^2 = 16, far below maxdim), so any visible
#      disagreement is a circuit mismatch, not a truncation artefact.
#   B: trajectories vs MPDO on the same circuit, in sigma.
#   C: closed circuit -- trajectories must be deterministic and match exactly.
#SBATCH --job-name=cktval
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cktval_%j.out
#SBATCH --error=logs/cktval_%j.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NTRAJ=${NTRAJ:-2000}

echo "host=$(hostname) cwd=$(pwd) threads=$JULIA_NUM_THREADS NTRAJ=$NTRAJ"
echo "start: $(date)"

for f in vectorized_evolution.jl trajectory_evolution.jl circuit_native.jl \
         validate_circuit_native.jl F_diagnostics.jl; do
  [ -f "$f" ] || { echo "FATAL: $f not found in $(pwd)" >&2; exit 2; }
done

julia --threads="$JULIA_NUM_THREADS" validate_circuit_native.jl
JULIA_STATUS=$?
echo "end: $(date)"

if [ "$JULIA_STATUS" -ne 0 ]; then
  echo "FATAL: julia exited $JULIA_STATUS -- see the .err log" >&2
  exit "$JULIA_STATUS"
fi
[ -f validate_mpdo_population.csv ] || { echo "FATAL: no CSV produced" >&2; exit 3; }
echo "--- output ---"; ls -la validate_mpdo_population.csv
exit 0
