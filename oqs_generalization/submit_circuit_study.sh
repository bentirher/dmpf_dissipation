#!/bin/bash
# =============================================================================
# STATUS -- only two things are still outstanding.
#
#   DONE, do not rerun:
#     0,1,5,6   theta sweeps at n=16 and n=24, Neel / |+> / random
#     2,9,11    fidelity curves (max|dZ|, HS distance, trace distance)
#     10        n=24 anchor: chi_traj >= 1741 (capped at 2048)
#     18,19,20  n-scaling: chi_traj = 13.8 / 48.8 / 168.6 at n = 8 / 12 / 16
#
#   OUTSTANDING:
#     24        the (theta, p) colormap          <-- run this first
#     28,29,30  the map at n = 8, 12, 20. NOW THE PRIORITY: at n=16 the low-p
#               band is at the finite-size ceiling 2^(n/2)=256 (chi/256 = 0.94
#               to 0.98), so its theta structure is washed out.
#     31        Trotter-step convergence at fixed physics, on correlators.
#     25,26,27  the EXPONENT at other (theta, p) points. chi was optimised at
#               n=16; hardware is at n~40, where the exponent decides. ~6 min each.
#     16,17     untruncated MPDO-vs-trajectory table, for the write-up
#
#   ABANDONED, and why:
#     3,4       MODE=damping and the multi-n MODE=scaling. Both are subsumed:
#               damping is a slice of task 24, and the scaling ladder is done.
#     21,22,23  n = 20, 24, 28 scaling. n=24 cost 7.4 h for ONE trajectory and
#               n=28 extrapolates to ~158 h. n=24 is the cluster ceiling and
#               tasks 18-20 already pin the exponent.
#
# THE SCALING RESULT, since it is what everything else was for:
#     ln(chi_traj) = 0.313 n + 0.13   from the three uncensored points
#                                     (local slopes 0.3158, 0.3099 -- flat)
#     predicts 2068 at n=24; measured >= 1741 at a cap of 2048. Consistent.
#     The continuum operating line gave 0.221 per site, so the hardware circuit
#     is STEEPER, and N*chi^3 crosses 1e18 at n ~ 37 rather than ~47.
#
#   sbatch --array=24    submit_circuit_study.sh   # the colormap, ~4 h
#   sbatch --array=16,17 submit_circuit_study.sh   # the head-to-head table
# =============================================================================
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
  # --- THE EXPONENT AT SEVERAL POINTS ON THE MAP (25-27) --------------------
  #
  # WHY THIS MATTERS MORE THAN IT LOOKS. Tasks 18-20 measured
  #     ln chi_traj = 0.313 n + 0.13   ->  crossover at n ~ 37
  # at ONE point, (theta, p) = (0.95, 0.05). The hardware runs near n = 40, and
  # there what matters is the EXPONENT, not the value at n = 16. Maximising
  # chi(n=16) and maximising chi(n=40) are different optimisations, and only the
  # first has been done. If the exponent varies across the map, the operating
  # point moves.
  #
  # Also: p = 0.05 was inherited from the continuum result that dissipation
  # first shows at gamma*t ~ 0.4, i.e. 1-(1-p)^10 = 0.40. That argument belongs
  # to a framing we have dropped and has never been tested in circuit units.
  #
  # Cost is trivial -- measured wall per trajectory is 9 s, 13 s, 59 s at
  # n = 8, 12, 16 -- so each task is ~6 min. Set THETA and P from the colormap
  # first; the defaults below bracket the current point in damping.
  25) export MODE=scaling NLIST=8,12,16 THETA=0.95 P=0.01 NTRAJ=64 EXCITED=random
      export MAXDIM_TRAJ=4096 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export OUTDIR=ckt_exp_lowp TAG=lowp ;;
  26) export MODE=scaling NLIST=8,12,16 THETA=0.95 P=0.15 NTRAJ=64 EXCITED=random
      export MAXDIM_TRAJ=4096 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export OUTDIR=ckt_exp_highp TAG=highp ;;
  # The second plateau: equally costly at n=16, much coarser as a discretisation.
  # If its exponent is higher, the trade-off deserves a second look.
  27) export MODE=scaling NLIST=8,12,16 THETA=2.10 P=0.05 NTRAJ=64 EXCITED=random
      export MAXDIM_TRAJ=4096 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export OUTDIR=ckt_exp_th21 TAG=th21 ;;

  # --- THE MAP AT OTHER SYSTEM SIZES (28-30) -------------------------------
  #
  # WHY THIS IS NOW THE PRIORITY. At n=16 the low-p band is FINITE-SIZE
  # SATURATED: the pure-state half-cut ceiling is 2^(n/2) = 256 and chi/256 runs
  # 0.94-0.98 for p <= 0.02 across theta in [0.5,1.25]. The theta structure is
  # washed out there and chi(p=0) understates the closed-system cost, so every
  # "damping costs X%" number from the n=16 map is an upper bound.
  #
  # n=8 and n=12 are nearly free (wall/traj 9 s and 13 s) so they get the full
  # grid. n=20 does NOT: wall/traj extrapolates to ~1250 s, and 117 cells x 2
  # waves would be ~80 h. It gets a coarse grid instead -- enough to test
  # whether the STRUCTURE (peak location, the pi/2 column, monotonicity in p)
  # moves with n, which is the actual question.
  28) export MODE=map N=8  K=10 NTRAJ=32 EXCITED=random
      export MAXDIM_TRAJ=2048 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export THETAS=0.20,0.35,0.50,0.65,0.80,0.95,1.10,1.25,1.40,1.5708,1.90,2.10,2.40
      export PLIST=0.0,0.005,0.01,0.02,0.035,0.05,0.08,0.12,0.20
      export OUTDIR=ckt_map_n8 TAG=m8 ;;
  29) export MODE=map N=12 K=10 NTRAJ=32 EXCITED=random
      export MAXDIM_TRAJ=2048 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export THETAS=0.20,0.35,0.50,0.65,0.80,0.95,1.10,1.25,1.40,1.5708,1.90,2.10,2.40
      export PLIST=0.0,0.005,0.01,0.02,0.035,0.05,0.08,0.12,0.20
      export OUTDIR=ckt_map_n12 TAG=m12 ;;
  # Coarse grid, 7 x 5 = 35 cells, NTRAJ=16 (one wave). ~12 h: submit with
  #     sbatch --time=16:00:00 --array=30 submit_circuit_study.sh
  30) export MODE=map N=20 K=10 NTRAJ=16 EXCITED=random
      export MAXDIM_TRAJ=4096 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export THETAS=0.35,0.65,0.95,1.25,1.5708,2.10,2.40
      export PLIST=0.0,0.02,0.05,0.12,0.20
      export OUTDIR=ckt_map_n20 TAG=m20 ;;

  # --- TROTTER STEP AT FIXED PHYSICS (31) ----------------------------------
  # Holds J*t = k*theta/2 and gamma*t = -k*ln(1-p) fixed at the operating point
  # and varies k, so every k is the SAME master equation discretised differently.
  # Measures magnetisation AND two-point correlators -- the higher-weight ones
  # feel discretisation error that single-site averages wash out.
  # MPDO-only and n=10, so it is minutes. BLAS gets the cores here because one
  # large SVD is the job, not many small trajectories.
  31) export MODE=ksweep N=10 K=10 THETA=0.95 P=0.05 EXCITED=random
      export MAXDIM_MPDO=1024 KREF=200 BLAS_THREADS=16
      export KLIST=2,3,4,5,6,8,10,14,20,30
      export OUTDIR=ckt_ksweep TAG=n10 ;;

  # --- THE COLORMAP. One job, and it replaces both the theta sweep and the
  # damping sweep: each is a one-dimensional slice of this grid.
  # 13 theta x 9 p = 117 points. Measured wall per trajectory at n=16 is 59 s,
  # and NTRAJ=32 is two waves on 16 threads, so ~4 h.
  # The CSV is rewritten after every theta row, so a timeout still leaves a
  # usable partial map rather than nothing.
  24) export MODE=map N=16 K=10 NTRAJ=32 EXCITED=random
      export MAXDIM_TRAJ=2048 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export THETAS=0.20,0.35,0.50,0.65,0.80,0.95,1.10,1.25,1.40,1.5708,1.90,2.10,2.40
      export PLIST=0.0,0.005,0.01,0.02,0.035,0.05,0.08,0.12,0.20
      export OUTDIR=ckt_map TAG=m16 ;;

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
  # ONE theta point at n=24, uncensored. The plateau SHAPE is already settled
  # by the n=16 sweeps (uncensored) and the n=24 Neel sweep; what is missing is
  # an uncensored MAGNITUDE at n=24, and that is a single number.
  #
  # Two things fixed after the 11-hour, one-point attempt:
  #   CUTOFF_TRAJ=1e-8 instead of 1e-10. The stored bond dimension is set by the
  #     cutoff, not by what we report: that run held 4096 Schmidt values to
  #     quote chi_req(1e-6) = 954. Cost goes as chi^3, so it paid ~75x for
  #     precision it then discarded. 1e-8 is still 100x tighter than the
  #     tolerance quoted.
  #   NTRAJ=16, one wave on 16 threads, so the wall clock is one trajectory.
  #     The chi statistics converge in a few tens of trajectories, and N for the
  #     <Z> error bar is extrapolated from the variance, not run.
  # Expect ~3-4 h.
  10) export MODE=theta N=24 NTRAJ=16 EXCITED=random
      export MAXDIM_TRAJ=2048 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export THETAS=0.95
      export OUTDIR=ckt_theta_n24_big TAG=n24big ;;
  # WHY 2048 AND WHY IT NEEDS A LONGER WALL. Both previous n=24 attempts
  # returned chi_traj ~ 1000 capped at 1024 (a run_theta bug made it ignore
  # MAXDIM_TRAJ and SKIP_MPDO). Extrapolating from the uncensored n=16 value
  # (168) at ~0.22 per site suggests the true value is ~1000-1500, so 2048
  # leaves headroom without paying for 4096. The MPDO is now genuinely skipped,
  # which recovers roughly half the previous 7.4 h -- but chi^3 at ~1300 is ~2x
  # the cost at 1024. Submit with a longer wall, which overrides the header:
  #     sbatch --time=12:00:00 --array=10 submit_circuit_study.sh

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
  3) export MODE=damping N=16 THETA=0.95 NTRAJ=64 EXCITED=random
     export MAXDIM_TRAJ=2048 MAXDIM_MPDO=256 CUTOFF_TRAJ=1e-8
     export PLIST=0.0,0.005,0.01,0.02,0.035,0.05,0.08,0.12,0.18,0.25
     export OUTDIR=ckt_damping TAG=n20 ;;

  # --- STEP 4: n scaling at the chosen (theta, p) --------------------------
  # THETA and P must both be replaced with the Step 2 / Step 3 optima first.
  # STEP 4 IS NOW ONE TASK PER n (18-23). Cost rises steeply -- measured ~4 min
  # per trajectory at n=24, chi=256, and it scales as n*chi^3 -- so a single job
  # covering n=8..28 cannot fit any wall. One n per task also means a timeout
  # costs one point instead of the whole scaling curve.
  18) export MODE=scaling NLIST=8  THETA=0.95 NTRAJ=64 EXCITED=random
      export MAXDIM_TRAJ=4096 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export OUTDIR=ckt_scaling_n8 TAG=n8 ;;
  19) export MODE=scaling NLIST=12 THETA=0.95 NTRAJ=64 EXCITED=random
      export MAXDIM_TRAJ=4096 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export OUTDIR=ckt_scaling_n12 TAG=n12 ;;
  20) export MODE=scaling NLIST=16 THETA=0.95 NTRAJ=64 EXCITED=random
      export MAXDIM_TRAJ=4096 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export OUTDIR=ckt_scaling_n16 TAG=n16 ;;
  21) export MODE=scaling NLIST=20 THETA=0.95 NTRAJ=32 EXCITED=random
      export MAXDIM_TRAJ=4096 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export OUTDIR=ckt_scaling_n20 TAG=n20 ;;
  22) export MODE=scaling NLIST=24 THETA=0.95 NTRAJ=16 EXCITED=random
      export MAXDIM_TRAJ=4096 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export OUTDIR=ckt_scaling_n24 TAG=n24 ;;
  23) export MODE=scaling NLIST=28 THETA=0.95 NTRAJ=16 EXCITED=random
      export MAXDIM_TRAJ=4096 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export OUTDIR=ckt_scaling_n28 TAG=n28 ;;

  4) export MODE=scaling NLIST=8,12,16 THETA=0.95 NTRAJ=64 EXCITED=random
     export MAXDIM_TRAJ=2048 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
     export OUTDIR=ckt_scaling TAG=opt ;;
esac

mkdir -p "$OUTDIR"
echo "task=$SLURM_ARRAY_TASK_ID MODE=$MODE N=${N:-$NLIST} K=$K P=$P THETA=${THETA:-sweep} MAXDIM_TRAJ=${MAXDIM_TRAJ:-unset} MAXDIM_MPDO=${MAXDIM_MPDO:-unset} SKIP_MPDO=${SKIP_MPDO:-false} CUTOFF_TRAJ=${CUTOFF_TRAJ:-unset}"
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
