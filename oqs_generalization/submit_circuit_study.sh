#!/bin/bash
# The whole hardware-aware study, one driver, selected by MODE.
#
#   sbatch --array=0-2,5,6 submit_circuit_study.sh  # STEP 2: theta + fidelity + init state  [DONE]
#   sbatch --array=7-9     submit_circuit_study.sh  # STEP 2b  [DONE -- but n=24 was CENSORED]
#   sbatch --array=10,11   submit_circuit_study.sh  # STEP 2c: uncensor n=24, and the HS fidelity
#   sbatch --array=12-14   submit_circuit_study.sh  # MPDO on its own terms, no prior context needed
#   sbatch --array=3,4     submit_circuit_study.sh  # STEPS 3-4 at the chosen (theta, init)
#   sbatch --array=16,17   submit_circuit_study.sh  # STANDALONE: MPDO vs trajectories, untruncated
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
#SBATCH --time=11:00:00
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

  # --- STEP 2b: confirm the initial-state gain at larger n, and redo the ----
  # fidelity curve for the initial state we will actually use.
  #
  # STEP 2 RESULT. chi_traj peaks on a broad plateau theta in [0.8, 1.25]:
  #     n=16 Neel 77 | |+> 170 | random 168        n=24 Neel 243
  # A non-basis initial state MORE THAN DOUBLES the cost at n=16, for one layer
  # of single-qubit gates on hardware. theta = pi/2 collapses for every initial
  # state because RZZ(2*theta) -> RZZ(pi) = -i ZZ is a PAULI there, leaving the
  # XX+YY part alone -- a free-fermion (matchgate) circuit, classically
  # simulable in O(n^3) whatever its entanglement. The Schmidt-tail factor
  # chi/2^S drops to 1.0-2.2 there against 4-10 elsewhere, which is the flat
  # spectrum such structure produces. Avoid pi/2; theta = 0.95 is the choice.
  7) export MODE=theta N=24 MAXDIM=256 NTRAJ=64 EXCITED=random
     export OUTDIR=ckt_theta_n24_rand TAG=n24rand ;;
  8) export MODE=theta N=24 MAXDIM=256 NTRAJ=64 EXCITED=plus
     export OUTDIR=ckt_theta_n24_plus TAG=n24plus ;;
  # The published fidelity curve used Neel; infidelity depends on the initial
  # state, so it must be remeasured for the one actually used. n=8, exact, cheap.
  9) export MODE=fidelity N=8 MAXDIM=256 KREF=1000 EXCITED=random
     export OUTDIR=ckt_fidelity_rand TAG=n8rand ;;

  # --- STEP 2c: REDO n=24 WITHOUT THE CAP -----------------------------------
  # Tasks 7 and 8 returned chi_traj = 256.0 exactly at eleven of seventeen
  # angles: that is MAXDIM, not a measurement, and the flat theta-plateau there
  # is clipping rather than physics. The printed table only showed the MPDO's
  # saturation flag, which is now fixed (a "!!" in the chi_traj column).
  #
  # MAXDIM=4096 with a narrowed theta grid around the plateau. Cost scales as
  # chi^3, but k=10 is only 1/24 of the 240-step continuum runs, so this is
  # ~30 min rather than the 5.6 h its continuum counterpart took.
  10) export MODE=theta N=24 MAXDIM=4096 NTRAJ=64 EXCITED=random
      export THETAS=0.50,0.65,0.80,0.95,1.10,1.25,1.40,1.90,2.10,2.40
      export OUTDIR=ckt_theta_n24_big TAG=n24big ;;

  # Fidelity again, now reporting BOTH max|dZ| and the Hilbert-Schmidt distance.
  # max|dZ| depends on the scale of the observable and hence on the initial
  # state: Neel starts every <Z_j> at +/-1, a random product state starts them
  # spread over [-1,1]. Part of the apparent 0.137 -> 0.062 improvement from
  # Neel to random is the observable shrinking, not the circuit improving. The
  # HS distance is a property of the states and is comparable across them.
  11) export MODE=fidelity N=8 MAXDIM=256 KREF=1000 EXCITED=random
      export OUTDIR=ckt_fidelity_hs TAG=hs ;;

  # --- THE MPDO ON ITS OWN TERMS -------------------------------------------
  # These three make the MPDO case self-contained: nothing here relies on the
  # earlier continuous-time study, so the result can be shown to someone seeing
  # the project for the first time.
  #
  # "Uncapped" needs care. The MPDO ceiling is 4^(n/2): 256 at n=8, 4096 at
  # n=12, 65536 at n=16, 1.7e7 at n=24. Truly uncapped is impossible above
  # n~12 -- but at n<=12 setting maxdim TO the ceiling means no truncation at
  # all, so those points are EXACT and carry no caveat.
  #
  # 12: exact MPDO vs n. MAXDIM=0 means "use 4^(n/2)". Both routes, same points.
  12) export MODE=scaling NLIST=6,8,10,12 THETA=0.95 MAXDIM=0 NTRAJ=128 EXCITED=random
      export OUTDIR=ckt_mpdo_exact TAG=exact ;;
  # 13: at n=16 the ceiling (65536) is out of reach, so show directly that the
  # MPDO is NOT converged: chi tracks the ladder instead of flattening, while
  # the trajectory chi at the same point sits far below any cap. That contrast
  # IS the argument, measured in one table.
  13) export MODE=mpdoladder N=16 THETA=0.95 MAXDIM=256 NTRAJ=64 EXCITED=random
      export MAXDIMS=128,256,512,1024,2048,4096
      export OUTDIR=ckt_mpdo_ladder TAG=n16 ;;
  # 14: the theta curve with NO truncation anywhere, at n=12. A clean
  # self-contained figure: both methods, exact, across the full angle range.
  14) export MODE=theta N=12 MAXDIM=0 NTRAJ=128 EXCITED=random
      export OUTDIR=ckt_theta_exact TAG=n12exact ;;

  # --- STANDALONE: MPDO vs TRAJECTORIES, BOTH UNTRUNCATED -------------------
  #
  # A self-contained head-to-head that presumes nothing from the earlier
  # continuous-time study. Small n is the POINT, not a limitation: the
  # vectorised density matrix is an MPS of local dimension 4, so its bond
  # dimension cannot exceed 4^(n/2) -- 256 at n=8, 1024 at n=10, 4096 at n=12.
  # Setting MAXDIM to that ceiling makes the MPDO EXACT, and the trajectory
  # route (bounded by 2^(n/2) per system cut) is exact too. Every row is then a
  # measurement rather than a bound, which the large-n runs can never be.
  #
  # MAXDIM only costs when it is REACHED, so pinning it at the ceiling is free
  # whenever the physical chi stays below -- which at k=10 it does.
  #
  # Task 12 is the table: agreement on <Z_j> in units of the trajectory standard
  # error, chi_MPDO vs chi_traj, the ratio against chi_traj^2 (= 1 exactly if
  # rho stayed pure), and the cost ratio with N taken from the measured variance.
  # Task 13 is the supporting convergence ladder at a single n, showing what
  # "censored" looks like when maxdim is set below the ceiling -- worth having
  # next to task 12 so the exactness claim is visibly earned.
  16) export MODE=headtohead NLIST=4,6,8,10,12 THETA=0.95 MAXDIM=4096 NTRAJ=512
      export EXCITED=random OUTDIR=ckt_headtohead TAG=h2h ;;
  17) export MODE=mpdoladder N=10 THETA=0.95 MAXDIM=1024 NTRAJ=256
      export MAXDIMS=32,64,128,256,512,1024 EXCITED=random
      export OUTDIR=ckt_mpdoladder_n10 TAG=n10 ;;

  # --- STEP 3: how much damping can the circuit afford? ---------------------
  # THETA=0.95 and EXCITED=random come from Step 2: the plateau maximum, away
  # from the pi/2 free-fermion point, with infidelity ~0.14 (Neel reference;
  # task 9 remeasures it for :random). :random rather than :plus because at
  # theta=pi/2 the |+> circuit is Clifford on a stabilizer state, and while 0.95
  # is not pi/2 it is better not to have a Gottesman-Knill argument anywhere
  # near the operating point.
  3) export MODE=damping N=20 THETA=0.95 MAXDIM=2048 NTRAJ=128 EXCITED=random
     export PLIST=0.0,0.005,0.01,0.02,0.035,0.05,0.08,0.12,0.18,0.25
     export OUTDIR=ckt_damping TAG=n20 ;;

  # --- STEP 4: n scaling at the chosen (theta, p) --------------------------
  # THETA and P must both be replaced with the Step 2 / Step 3 optima first.
  4) export MODE=scaling NLIST=8,12,16,20,24,28,32 THETA=0.95 MAXDIM=4096 NTRAJ=64 EXCITED=random
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
