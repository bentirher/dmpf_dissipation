#!/bin/bash
# The whole hardware-aware study, one driver, selected by MODE.
#
#   sbatch --array=0-2,5,6 submit_circuit_study.sh  # STEP 2: theta + fidelity + init state
#   sbatch --array=3   submit_circuit_study.sh    # STEP 3: damping sweep
#   sbatch --array=4   submit_circuit_study.sh    # STEP 4: n scaling
#
# Run 0-2 first and look at them together before launching 3; the choice of
# theta made in Step 2 is an input to Steps 3 and 4.
#
# WHY THIS IS CHEAP. The earlier study integrated ~240 Trotter steps to reach
# t=0.45n. The hardware circuit is k=10 steps, full stop, so every point here is
# ~24x cheaper at the same chi. That is what makes n=24 affordable inside a
# parameter sweep rather than as a single heroic run.
#
# THE ONE THING TO WATCH. With k=10 the entanglement cap is set by k, not n:
# only the gates that cross a given cut can raise its Schmidt rank, and there
# are 2 per step in this first-order circuit. Expect S_op to saturate in n well
# before the chi ceiling bites, and expect MAXDIM not to bind at all for the
# trajectory route. If the `sat` column fires anywhere, the point is a lower
# bound and needs a rerun at larger MAXDIM before being quoted.
#SBATCH --job-name=cktstudy
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/cktstudy_%A_%a.out
#SBATCH --error=logs/cktstudy_%A_%a.err
mkdir -p logs
module load Julia/1.11.6-linux-x86_64

# Trajectories are many small independent problems: cores to Julia, BLAS to 1.
# The driver also calls BLAS.set_num_threads(1) so this cannot be got wrong.
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Shared. k=10 is the hardware budget (300 two-qubit depth / ~30 per step).
# p=0.05 gives total damping 1-(1-p)^10 = 0.40, which is where the earlier study
# found dissipation first becomes visible. EXCITED=neel rather than the single
# excitation used for validation: Neel is far more entangling and costs one
# depth-1 layer of X gates on hardware.
export K=10 P=0.05 EXCITED=neel CUTOFF=1e-12
export JREF=0.25 GREF=0.0625      # only used to print the implied (dt, t)

case $SLURM_ARRAY_TASK_ID in
  # --- STEP 2a: where is the circuit hardest in theta? -----------------------
  # MAXDIM=256, down from 1024, for three reasons. (i) At 1024 the MPDO
  # saturated from theta=0.2 onward, so the expensive number was a censored
  # lower bound anyway. (ii) S_op converges in maxdim far faster than chi does.
  # (iii) The earlier study already settled that the MPDO loses to trajectories
  # by 8-13 orders of magnitude, so the informative probe here is chi_traj,
  # which came in at 11-75 -- nowhere near any cap.
  # With that plus the bond_report gauge fix and snapshots only at the final
  # step, each theta point drops from ~2 h to ~1 min.
  0) export MODE=theta N=16 MAXDIM=256 NTRAJ=128
     export OUTDIR=ckt_theta_n16 TAG=n16 ;;
  1) export MODE=theta N=24 MAXDIM=256 NTRAJ=64
     export OUTDIR=ckt_theta_n24 TAG=n24 ;;

  # --- STEP 2c: the initial state is a first-class knob ---------------------
  # At theta=pi/2 the bond gate is locally iSWAP, a permutation on basis states:
  # from Neel the circuit stays EXACTLY product (S_op=0, measured exactly at
  # n=8), from |+>^n it reaches the maximum. One Hadamard layer on hardware.
  # CAUTION: at theta=pi/2 the gates are Clifford and |+>^n is a stabilizer
  # state, so that combination is Gottesman-Knill simulable regardless of
  # entanglement. :random is the non-stabilizer stress test; run all three and
  # site the operating point away from pi/2.
  5) export MODE=theta N=16 MAXDIM=256 NTRAJ=128 EXCITED=plus
     export OUTDIR=ckt_theta_plus TAG=plus ;;
  6) export MODE=theta N=16 MAXDIM=256 NTRAJ=128 EXCITED=random
     export OUTDIR=ckt_theta_rand TAG=rand ;;

  # --- STEP 2b: where does it stop being the master equation? ---------------
  # n=8 is exact (Liouville ceiling 4^4 = 256) and Trotter error is short
  # range, so this is representative and costs almost nothing. KREF=1000 is the
  # converged reference at the same (J*t, gamma*t).
  2) export MODE=fidelity N=8 MAXDIM=256 KREF=1000
     export OUTDIR=ckt_fidelity TAG=n8 ;;

  # --- STEP 3: how much damping can the circuit afford? ---------------------
  # THETA=1.05 is a PLACEHOLDER from the n=8 exploration (3.39 bits from Neel,
  # non-Clifford, infidelity ~0.15). Replace it with the Step 2 optimum.
  3) export MODE=damping N=20 THETA=1.05 MAXDIM=256 NTRAJ=128
     export PLIST=0.0,0.005,0.01,0.02,0.035,0.05,0.08,0.12,0.18,0.25
     export OUTDIR=ckt_damping TAG=n20 ;;

  # --- STEP 4: n scaling at the chosen (theta, p) --------------------------
  # THETA and P must both be replaced with the Step 2 / Step 3 optima first.
  4) export MODE=scaling NLIST=8,12,16,20,24,28,32 THETA=1.05 MAXDIM=256 NTRAJ=64
     export OUTDIR=ckt_scaling TAG=opt ;;
esac

mkdir -p "$OUTDIR"
echo "task=$SLURM_ARRAY_TASK_ID MODE=$MODE N=${N:-$NLIST} K=$K P=$P THETA=${THETA:-sweep} MAXDIM=$MAXDIM"
echo "OUTDIR=$OUTDIR threads=$JULIA_NUM_THREADS host=$(hostname) cwd=$(pwd)"
echo "start: $(date)"

for f in vectorized_evolution.jl trajectory_evolution.jl circuit_native.jl \
         circuit_study.jl F_diagnostics.jl; do
  [ -f "$f" ] || { echo "FATAL: $f not found in $(pwd)" >&2; exit 2; }
done

julia --threads="$JULIA_NUM_THREADS" circuit_study.jl
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
# manifest.csv is written before any physics, so <=1 means Julia died early.
if [ "$NCSV" -le 1 ]; then
  echo "FATAL: julia exited 0 but produced no data beyond the manifest" >&2
  exit 3
fi
exit 0
