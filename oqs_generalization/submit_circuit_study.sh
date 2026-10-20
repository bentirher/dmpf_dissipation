#!/bin/bash
# =============================================================================
# STATUS
#
#   THE PARAMETER SEARCH IS OVER, AND IT CAME BACK NEGATIVE ON theta.
#
#   The four maps (n = 8, 12, 16, 20; tasks 24, 28, 29, 30) are all in, and the
#   conclusion is that (theta, p) is not where the hardness lives. Fitting
#   beta = dln(chi)/dn on the cells that are NOT pinned at 2^(n/2):
#
#       p      theta=0.35   0.65   0.95   1.25   pi/2    2.1    2.4
#       0.02      0.211    0.330  0.338  0.319  -0.108  0.325  0.332
#       0.05      0.220    0.306  0.320  0.302  -0.076  0.313  0.309
#       0.12      0.159    0.252  0.273  0.264  -0.028  0.272  0.270
#       0.20      0.097    0.192  0.200  0.189  -0.008  0.206  0.207
#
#   against a ceiling slope of ln2/2 = 0.3466. On the whole plateau theta in
#   [0.65, 2.4] the exponent is 92-97% of that ceiling at p <= 0.05 and the
#   spread ACROSS theta is under 5% -- smaller than the scatter over random
#   initial states. chi is just tracking 2^(n/2). Verified independently by
#   exact statevector at theta=0.95, p=0, k=10: chi/2^(n/2) = 0.951, 0.953,
#   0.948, 0.936 at n = 18, 20, 22, 24.
#
#   So theta is a "do not pick these" list, not an optimisation target:
#     * theta <~ 0.5 is genuinely cheaper (beta ~ 0.21): the light cone is too
#       short at k=10 to cross the chain.
#     * theta = pi/2 is structurally simulable -- rzz(2*theta) becomes the Pauli
#       ZZ and the XY part becomes a full iSWAP, leaving a matchgate (free
#       fermion) and Clifford circuit. beta is NEGATIVE there: chi falls from 32
#       at n=16 to 2 at n=20 (an exact revival), reproduced exactly offline.
#     * everything else in [0.65, 2.4] is equivalent to within the error bars.
#       theta = 0.95 is the nominal argmax and 2.4 ties with it.
#
#   WHAT IS LEFT, therefore, is depth. The analytic cap is chi <= 4^k (one
#   rank-4 two-site gate crosses any cut per Trotter step), so k=10 caps chi at
#   4^10 ~ 1.0e6, and 2^(n/2) only reaches 1.0e6 at n = 40. That makes n = 40
#   the point where more qubits stop buying classical cost at k = 10.
#
#   NEXT UP:
#     46,47     MODE=frontier -- the accuracy-vs-hardness plot, BOTH axes in one
#               CSV. Run 47 first (4 cells, ~20 min, reproduces the k=10 row you
#               can cross-check against the maps), then 46 for the full grid.
#                   sbatch --array=47 submit_circuit_study.sh
#     44,45,53,54  THE ACCURACY-vs-HARDNESS CANDIDATES. chi is FLAT in theta at
#               fixed (J*t, gamma*t), so the coarse step was never what made the
#               circuit hard -- theta=0.95 was buying Trotter error for nothing.
#               Candidate B (theta=0.30, k=15) beats the old choice on BOTH
#               axes. Run these four before fixing a hardware protocol.
#                   sbatch --array=44,45,53,54 submit_circuit_study.sh
#
#     50,51,52  MODE=evolve -- the hardware baseline. Observable time series
#               (mean magnetisation and the CENTRAL two-qubit correlator) for
#               the k=10 circuit AND for a 200-substep quasi-exact reference at
#               matched physical times, step by step. This is what the device
#               data gets plotted against, and the gap between the two curves is
#               the Trotter error the hardware will also show.
#                   sbatch --array=50 submit_circuit_study.sh    # minutes, exact
#
#   OUTSTANDING, in priority order:
#     40,41,42  MODE=depth -- chi(n, k) at the chosen (theta, p). THE measurement
#               the maps could not give. Run 40 first; it is ~2 h.
#     31        Trotter-step convergence at fixed physics, on correlators. This
#               is now the other half of the story: task 40 says how deep the
#               circuit must be to be hard, task 31 says how coarse the step may
#               be before the circuit stops representing the master equation.
#               If those two windows do not overlap, there is no experiment.
#     43        map at n=24 on the plateau only, to extend the beta fit one point.
#     16,17     untruncated MPDO-vs-trajectory table, for the write-up.
#
#   DONE, do not rerun:
#     0,1,5,6   theta sweeps at n=16 and n=24, Neel / |+> / random
#     2,9,11    fidelity curves (max|dZ|, HS distance, trace distance)
#     10        n=24 anchor: chi_traj >= 1741 (capped at 2048)
#     18,19,20  n-scaling: chi_traj = 13.8 / 48.8 / 168.6 at n = 8 / 12 / 16
#     24,28-30  the (theta, p) maps at n = 16, 8, 12, 20
#
#   ABANDONED, and why:
#     3,4       MODE=damping and the multi-n MODE=scaling: damping is a slice of
#               the map and the scaling ladder is done.
#     21,22,23  n = 20, 24, 28 scaling. n=24 cost 7.4 h for ONE trajectory.
#     25,26,27  the exponent at other (theta, p) points. Superseded: the maps
#               give beta at 35 (theta, p) points at once, which is what these
#               three were for.
#
# THE SCALING RESULT, since it is what everything else was for:
#     ln(chi_traj) = 0.3198 n + 0.07   at theta = 0.95, p = 0.05, k = 10
#     => chi = 643 (n=20), 2.3e3 (n=24), 8.3e3 (n=28), 1.6e4 (n=30)
#     => N*chi^3 crosses 1e18 at n ~ 28, and the 64 GB-per-node memory wall
#        (n*d*chi^2*16 B) arrives first, at n ~ 28.
#     So the classical frontier at k=10 is n = 26-30, NOT n = 40.
#
#   sbatch --array=40    submit_circuit_study.sh   # chi(n,k), ~2 h
#   sbatch --array=31    submit_circuit_study.sh   # Trotter step vs correlators
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

  # --- THE HARDWARE BASELINE (50-52) ----------------------------------------
  #
  # Observable time series for the Option-B protocol -- fixed Trotter step,
  # time advanced by repeating the layer, so step s IS the time point. Produces
  # the circuit curve and a quasi-exact reference curve at MATCHED physical
  # times, which is what the device data gets plotted against.
  #
  # MAXDIM_MPDO is chosen to be the EXACT vectorised ceiling 4^(n/2), so these
  # runs are not truncated at all and the only error in them is the Trotter
  # step itself:   n=8 -> 4^4 = 256,  n=10 -> 4^5 = 1024,  n=12 -> 4^6 = 4096.
  # Watch the trace_* columns: they must stay at 1 to ~1e-10.
  #
  # Run 50 FIRST as a smoke test -- it is minutes, and it is the one run in the
  # whole study with truncation identically zero, so if the circuit and the
  # reference disagree there it is real Trotter error and nothing else.
  #
  # BLAS gets the cores here: MPDO is one large SVD per gate, not many small
  # independent trajectories.
  50) export MODE=evolve N=8  K=10 THETA=0.95 P=0.05 MREF=20 EXCITED=random
      export MAXDIM_MPDO=256  CUTOFF=1e-14 BLAS_THREADS=16
      # REF_EVERY=1 also writes evolve_dense_ref_* : the reference recorded at
      # EVERY substep, i.e. the smooth continuous-time curve the circuit is a
      # coarse sampling of. Roughly doubles the reference cost; trivial at n=8.
      export REF_EVERY=1
      export OUTDIR=ckt_evolve_n8  TAG=e8 ;;
  51) export MODE=evolve N=10 K=10 THETA=0.95 P=0.05 MREF=20 EXCITED=random
      export MAXDIM_MPDO=1024 CUTOFF=1e-14 BLAS_THREADS=16
      export OUTDIR=ckt_evolve_n10 TAG=e10 ;;
  # n=12 at the exact ceiling 4096 is the expensive one (200 reference steps
  # with 4096 x 4096 SVDs). Give it the full wall and check `sat_ref` before
  # quoting anything from it.
  52) export MODE=evolve N=12 K=10 THETA=0.95 P=0.05 MREF=20 EXCITED=random
      export MAXDIM_MPDO=4096 CUTOFF=1e-14 BLAS_THREADS=16
      export OUTDIR=ckt_evolve_n12 TAG=e12 ;;

  # --- DEPTH: chi(n, k) AT THE CHOSEN POINT (40-42) -------------------------
  #
  # The one measurement the maps cannot give. theta is settled (0.95, anywhere
  # on the [0.65, 2.4] plateau would do), so sweep the two knobs that are left.
  #
  # Each row of the printed table is one k; each column one n. The number in
  # brackets is chi / min(2^(n/2), 4^k). While that stays near 0.95 the circuit
  # is maximally entangling and the ceiling is the only thing setting the cost.
  # The n at which it falls clearly below 1 at fixed k = 10 is the point past
  # which extra qubits buy NO extra classical cost -- the thing that decides
  # whether a 40-qubit run is more interesting than a 28-qubit one.
  #
  # p = 0.05 is chosen WITHOUT reference to any J. The only damping quantity the
  # circuit defines by itself is the per-qubit probability that the channel has
  # fired at least once over the whole run, D = 1 - (1-p)^k, which is the
  # `total_damping` column in the map CSVs. At k=10:
  #     p    0.02   0.05   0.12   0.20
  #     D    0.18   0.40   0.72   0.89
  #     beta 0.338  0.320  0.273  0.200     (ceiling ln2/2 = 0.3466)
  # D ~ 0.4 is large enough that the dissipation shapes the state and small
  # enough that it has not driven everything toward |0...0>, and it costs only
  # 8% of the entanglement exponent. (The continuum operating line would have
  # said p = 0.0611; that it lands in the same place is a cross-check, not an
  # input, and nothing here depends on it.)
  #
  # Cost: n <= 20 at k <= 16 with NTRAJ=16 is ~2 h. 41 extends k at small n only
  # (cheap, and k=24 at n=12 is where the depth bound 4^k stops binding). 42 is
  # the expensive one -- n=24 at k=16 means chi ~ 4e3 -- so it gets its own
  # longer wall and a coarse k list.
  40) export MODE=depth NLIST=8,12,16,20 K=10 THETA=0.95 P=0.05 EXCITED=random
      export NTRAJ=16 MAXDIM_TRAJ=4096 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export KLIST=2,4,6,8,10,12,14,16
      export OUTDIR=ckt_depth TAG=d1 ;;
  41) export MODE=depth NLIST=8,12,16 K=10 THETA=0.95 P=0.05 EXCITED=random
      export NTRAJ=16 MAXDIM_TRAJ=4096 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export KLIST=18,20,24,28,32
      export OUTDIR=ckt_depth_deep TAG=d2 ;;
  # sbatch --time=23:00:00 --array=42 submit_circuit_study.sh
  42) export MODE=depth NLIST=22,24 K=10 THETA=0.95 P=0.05 EXCITED=random
      export NTRAJ=8 MAXDIM_TRAJ=8192 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export KLIST=8,10,12,14
      export OUTDIR=ckt_depth_big TAG=d3 ;;

  # --- 46-47: THE ACCURACY-vs-HARDNESS FRONTIER ------------------------------
  #
  # One job, one CSV, both axes -- the plot that picks the hardware protocol.
  # Every cell sits on the physical ray gamma/J = 0.108, so p is determined:
  #     J*t = k*theta/2,  gamma*t = 0.108*J*t,  p = 1-exp(-gamma*t/k)
  # and for each (k, theta) it measures chi (trajectories, every n in NLIST)
  # AND the Trotter error (MPDO at N_ERR against a fine reference at the same
  # J*t), plus a second reference at mref/2 so the error has its own error bar.
  #
  # Plot chi_n16 against err_z_max: up and to the LEFT wins. Check every err_*
  # against its ref_err_* first -- a ratio below ~10 means raise REF_STEPS.
  #
  # N_ERR=8 with MAXDIM_MPDO=256 is the exact vectorised ceiling 4^4, so the
  # error runs are untruncated and err_* is pure Trotter error.
  #
  # ERR_SEEDS: the Trotter error is strongly initial-state dependent. Measured
  # at n=8, k=10, theta=0.95 over 14 random product states, max_j|Delta<Z_j>|
  # ranges 0.046..0.185 -- a factor of FOUR, sd ~50% of the mean. The first
  # run of task 47 used one state and produced a NON-MONOTONIC error column
  # (0.039, 0.091, 0.079, 0.061 for theta = 0.30, 0.475, 0.65, 0.95), which is
  # scatter, not signal. Each error is now the mean over ERR_SEEDS states and
  # the CSV carries its sd; a difference smaller than the sd means nothing.
  # err_z_mean (mean over sites) is the steadier discriminator than err_z_max.
  #
  # 46 is the production grid: 4 k x 4 theta = 16 cells. With ERR_SEEDS=5 the
  # error part is ~5x the smoke test's ~80 s/cell, so budget ~2 h for errors
  # plus ~1 h of trajectories. Submit with --time=11:00:00 as usual.
  # 47 is a 4-cell smoke test at k=10 only -- run it first, ~1 h with 5 seeds,
  # and it reproduces the k=10 row, which you can cross-check against the maps.
  46) export MODE=frontier GJ=0.107986 NLIST=8,12,16 N_ERR=8 REF_STEPS=240
      export ERR_SEEDS=5
      export KLIST=10,15,20,32
      export THETAS=0.30,0.475,0.65,0.95
      export NTRAJ=16 MAXDIM_TRAJ=4096 MAXDIM_MPDO=256
      export CUTOFF=1e-14 CUTOFF_TRAJ=1e-8 EXCITED=random BLAS_THREADS=4
      export OUTDIR=ckt_frontier TAG=fr ;;
  47) export MODE=frontier GJ=0.107986 NLIST=8,12,16 N_ERR=8 REF_STEPS=240
      export ERR_SEEDS=5
      export KLIST=10
      export THETAS=0.30,0.475,0.65,0.95
      export NTRAJ=16 MAXDIM_TRAJ=4096 MAXDIM_MPDO=256
      export CUTOFF=1e-14 CUTOFF_TRAJ=1e-8 EXCITED=random BLAS_THREADS=4
      export OUTDIR=ckt_frontier_smoke TAG=fr0 ;;

  # --- 44-45 / 53-54: THE ACCURACY-vs-HARDNESS CANDIDATES --------------------
  #
  # WHY THESE EXIST. Holding the PHYSICS fixed (J*t = 4.75, gamma*t = 0.513) and
  # varying only the Trotter step gives chi(n=16) = 160, 180, 173, 198, 185 at
  # theta = 0.20, 0.30, 0.475, 0.65, 0.95 -- FLAT. Hardness is set by (J*t,
  # gamma*t), not by how coarsely you discretise. The coarse step was never what
  # made the circuit hard, so theta = 0.95 was buying Trotter error for nothing.
  #
  # At fixed depth k, theta only sets how much physical time you reach,
  # J*t = k*theta/2, and the entanglement barrier is broad and flat past
  # J*t ~ 2.4. So shortening the run costs almost no hardness and buys a lot of
  # accuracy. Measured at n=8 (error) and n=16 (chi):
  #
  #     k   theta  depth   J*t  gamma*t | err_z   chi(n=16)
  #    10   0.950    300  4.75    0.513 | 0.128       185     <- the old choice
  #    10   0.475    300  2.38    0.257 | 0.066       174     <- candidate A
  #    15   0.300    450  2.25    0.243 | 0.029       204     <- candidate B
  #    20   0.300    600  3.00    0.324 | 0.039       181
  #
  # Candidate B DOMINATES the old choice on both axes: more chi AND 4.4x less
  # Trotter error, for 1.5x the two-qubit depth. Candidate A is free -- same
  # depth, same hardness, half the error.
  #
  # Hardness peaks on a ridge at gamma*t ~ 0.3-0.55 (the entanglement barrier of
  # the master equation). Past it the damping wins: k=32, theta=0.95 gives
  # gamma*t = 1.64 and chi collapses to 102.
  #
  # 44/45 measure chi(n) at the two candidates; 53/54 measure their Trotter
  # error. Run all four before committing to a hardware protocol. The numbers
  # above are 6 trajectories at n=16, so +-10% -- enough to see that chi is flat
  # in theta, not enough to rank 204 against 185.
  44) export MODE=depth NLIST=8,12,16,20,22 K=10 THETA=0.475 P=0.0253 EXCITED=random
      export NTRAJ=16 MAXDIM_TRAJ=4096 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export KLIST=10
      export OUTDIR=ckt_candA TAG=cA ;;
  45) export MODE=depth NLIST=8,12,16,20,22 K=15 THETA=0.300 P=0.0161 EXCITED=random
      export NTRAJ=16 MAXDIM_TRAJ=4096 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export KLIST=15
      export OUTDIR=ckt_candB TAG=cB ;;
  53) export MODE=evolve N=10 K=10 THETA=0.475 P=0.0253 MREF=20 EXCITED=random
      export MAXDIM_MPDO=1024 CUTOFF=1e-14 BLAS_THREADS=16 REF_EVERY=1
      export OUTDIR=ckt_evolve_candA TAG=cA ;;
  54) export MODE=evolve N=10 K=15 THETA=0.300 P=0.0161 MREF=20 EXCITED=random
      export MAXDIM_MPDO=1024 CUTOFF=1e-14 BLAS_THREADS=16 REF_EVERY=1
      export OUTDIR=ckt_evolve_candB TAG=cB ;;

  # --- 43: one more n for the beta fit, plateau theta only -------------------
  # Drops theta = 0.35 and pi/2 (both settled) and p = 0, which is pinned at the
  # ceiling at every n and so contributes nothing to the fit. 4 theta x 4 p = 16
  # cells at n=24. Expensive: chi ~ 4e3 at p=0.02. Give it 23 h.
  43) export MODE=map N=24 K=10 NTRAJ=8 EXCITED=random
      export MAXDIM_TRAJ=8192 SKIP_MPDO=true CUTOFF_TRAJ=1e-8
      export THETAS=0.65,0.95,1.25,2.40
      export PLIST=0.02,0.05,0.12,0.20
      export OUTDIR=ckt_map_n24 TAG=m24 ;;

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
