# Open-System DMPF on Hyperion — The Road We Travelled (Chronological)

**Purpose.** A step-by-step record of the actual path — what we tried, what broke, what we
learned, in the order it happened — so future sessions don't re-tread solved ground. A
companion document (`findings_thematic.md`) organizes the same material by theme. Both end
with an Outlook section. Where a step revised an earlier conclusion, that is stated plainly.

**Context.** The goal was to run two experiments for the open-system DMPF method on the
DIPC Hyperion cluster: (1) the operator-Schmidt spectrum / bond-dimension growth of F under
middle-out contraction (MOC), and (2) a DMPF truncation sweep to see whether cheaper bond
dimensions still give faithful results.

---

## Step 0 — Starting point and first wall

Workflow at the outset: request an interactive node with `srun --pty bash`, edit the
Remote-SSH config to attach VS Code, clone the repo, run notebooks. The bond-dimension
notebook hit **"Out of memory"** starting the n=6 case (requested 20 GB).

First realization: this is not just a "request more RAM" problem. The direct Liouville-MPO
F saturates its bond-dimension ceiling `16^min(n/2, n-n/2)` (n=5 -> 256, n=6 -> 4096), and
SVD cost ~chi^3 plus memory ~chi^2 means n=6 is a hardware wall, not a tuning issue. Decided
to move off interactive runs to `sbatch` job arrays and adapt the workload properly.

---

## Step 1 — Cluster workflow: sbatch, scratch, arrays

Set up the production pattern:
- Work from `/scratch/$USER/...` (fast, not backed up).
- `sbatch` scripts, not interactive `srun --pty` (which dies with the SSH connection).
- Job array with one task per `(n, gamma)` pair; index math
  `n_idx = TASK_ID / 4`, `g_idx = TASK_ID % 4`.
- Split cheap (n=3,4,5) from expensive (n=6) since all array tasks share one resource
  request.
- Files: `run_case.jl` + `submit_small.sh`. Standard loop: commit -> push -> `git pull` on
  cluster -> `sbatch` -> `tail -f` the log.

Learned along the way: `git pull` is how updated files reach the cluster (forgot this once
and ran a stale script); `module spider julia` is a lookup command and should NOT live in
the batch script; the module name here is `Julia/1.11.6-linux-x86_64`.

---

## Step 2 — First runs, and a mysterious slowness

n=3 and n=4 finished fine. n=5 tasks TIMED OUT (first at 4 h, then at 8 h). `sacct` showed
`TIMEOUT`, MaxRSS ~10-14 GB (memory was fine; this was purely a speed problem). Something
was making n=5 far slower than it should be.

Instrumented `run_case.jl` with per-layer timing (inline copy of `track_Fii_bond_dimension`
so no project files changed). Hit two Julia gotchas here:
- **soft scope:** `F` mutated in the top-level `for` loop needed `global F` (else
  `UndefVarError`).
- flushed prints (`flush(stdout)`) needed so SLURM logs update live.

---

## Step 3 — The BLAS oversubscription bug

The instrumented test's first line was the giveaway: `BLAS threads: 64 / Julia CPU threads:
128`, while we had requested `--cpus-per-task=4`. BLAS had grabbed the whole node's cores
and was oversubscribing 64 threads onto a 4-core allocation — massive contention.

**Fix:** `BLAS.set_num_threads(SLURM_CPUS_PER_TASK)` in Julia AND
`export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK` in the script (`JULIA_NUM_THREADS` alone
does not control BLAS). Moved to `--cpus-per-task=16`. The MPO build dropped from
">10 min hang" to ~68 s. This was the single biggest performance bug of the whole effort.

---

## Step 4 — The maxdim pathology

Even with BLAS fixed, a single n=5 layer still would not finish in 10 min, then not in 1 h.
This was genuinely weird — chi=256 SVDs should be seconds, not an hour. The instrumentation
localized it: the MPO build finished in ~68 s, but the first `apply` in the layer loop hung.

Root cause: `apply` was called with `maxdim=4000`, while the true ceiling at n=5 is 256.
The loose cap forced `apply` to build and SVD-factorize huge *intermediate* tensors before
truncating. Capping at the true ceiling (256) fixed it:

    n=5, gamma=0.01, maxdim=256:  build 80 s, then layers 24/6/14/22/35/33/32/33 s
    bond_dim per layer: 13/65/126/194/224/238/246/249  (ceiling 256, not capped)
    whole case ~4 min  (was ">8 h, timed out")

**Lesson:** always set `maxdim = theoretical_max_bond_dim(n)`; the value passed to `apply`
controls intermediate SVD size even when the final rank is far below it.

(Along the way: a run at maxdim=256 FAILED fast with `UndefVarError: F` — the soft-scope
bug in the timing loop, fixed with `global F, bond_dims`.)

---

## Step 5 — The sweep: same bugs, then a cost surprise

Turned to the DMPF truncation sweep (`run_sweep_case.jl`). It had been launched the old way
and timed out for 24 h per task. Applied the same BLAS fix, then instrumented per-block
(`time_ref.jl`) — and hit two of my own bugs first: a `do`-block argument-order mistake
(`timed(label, f)` must be `timed(f, label)`), and it took a couple of tries to get a clean
timing run.

The timing breakdown was decisive:

    M_ref_full (gram, maxdim=256):  5261.5 s  (~88 min)
    M_opt (maxdim=16):                 2.5 s
    L_opt (maxdim=16, k_ref=40):       5.3 s

The maxdim=256 Gram build is the tent pole (~88 min), and the sweep was rebuilding it — plus
L_ref and purity — ONCE PER md (5-6 redundant ~88-min builds per task). That is why it never
finished.

---

## Step 6 — Gram convergence: can we cheapen the reference?

Before optimizing, asked whether maxdim_ref could be lowered below 256. `gram_convergence.jl`
computed M at maxdim = 64/96/128/192/256:

    relative dM: 3.1e-2 (64->96), 7.8e-3 (96->128), 6.2e-3 (128->192), 2.4e-3 (192->256)
    build time:  160 / 232 / 718 / 2224 / 4183 s   (~chi^3 scaling)

M is still drifting ~0.2% at 256 (the ceiling at n=5), so the reference genuinely needs the
full 256 — cannot cheapen it. But the de-duplication (compute M_ref/L_ref/purity ONCE,
reuse across md) plus dropping the degenerate md=256 candidate brought the sweep from
"cannot finish one md in 24 h" to **~3 h total** (`run_sweep_case.jl`, de-duplicated).

---

## Step 7 — Negative errors: a real bug in the error estimator

The de-duplicated sweep ran, but `E_mpf` came out NEGATIVE for md in {32,64,128}
(-6e-4 .. -9e-4). A squared Frobenius norm cannot be negative.

- **First hypothesis (WRONG):** purity mismatch. purity used a cheap MPS evolution; L_ref
  used an MPO sandwich; `verify_purity.jl` confirmed they differ by 8.5e-4. But making them
  consistent (`run_sweep_case_v2.jl`) made E_mpf MORE negative. So purity was only masking
  the problem.
- **Correct diagnosis:** the identity `||rho_ref-mu||^2 = purity + cMc - 2Lc` holds only in
  exact arithmetic. M_ref, L_ref, purity are independently-truncated sandwiches; their ~2e-3
  truncation inconsistency swamps a true error ~5e-4 and flips the sign.
- **Decisive test (`direct_norm_diag.jl`):** compute `||rho_ref - mu||^2` directly from MPS.
  At md=128: E_decomp = -1.494e-3 vs E_direct = +5.48e-4 (correct). Gap 2.04e-3 = the
  truncation floor.

Fix: evaluate the error via the direct norm (`run_sweep_case_v3.jl`). Numerically stable and
much cheaper. Flagged as a change to how a published quantity is evaluated (document it).

---

## Step 8 — Closing the original question: MOC vs plain, closed vs open

Returned to the opening motivation (can F be built cheaply?) with a controlled experiment
(`closed_moc_vs_plain.jl`): F_ii (exact final F = identity), tracking bond dimension per step
under MOC (lockstep) vs plain (forward-then-backward), at gamma=0.

    Closed system: MOC bond dim = 1 at EVERY sync (whole pass ~30 s).
                   PLAIN saturates to 256 by step 4, collapses to 113 at the end.

![MOC vs plain, closed and open](fig_moc_vs_plain.svg)

This confirmed the near-identity mechanism: in the closed system S^dag = S^{-1}, so MOC keeps
F the identity throughout — essentially free. Then the dissipative check
(`open_moc_vs_plain.jl`, gamma=0.05):

    Open system: MOC bond dim = 13 -> 168 -> 238 -> ... -> 256 (saturates).
                 PLAIN also 256.

MOC's advantage vanishes with dissipation (S^dag != S^{-1}), and both saturate. This both
verifies the implementation (MOC=1 closed, saturates open = correct dissipative step MPO)
and shows the open-system cost is intrinsic to dissipation, not the contraction order.

---

## Step 9 — The S^-1 idea

Hypothesis (the user's): F' built with S^{-1} instead of S^dag might recover the near-identity
cancellation (since S^{-1}S = I) and be cheap, IF it gives similar DMPF coefficients.
Construction: only the dissipator gates change to `inv(forward channel) = exp(-dt L_diss)`;
unitary layers already have adjoint = inverse (`sinv_vs_sdag_coeffs.jl`).

Single point n=5, gamma=0.05: conditioning of the inverse dissipator was FINE (compounded
amplification ~1.2, not a blow-up — my "ill-conditioned" worry was overstated). But the
coefficients differed drastically: c_dag=[-0.16,1.16] vs c_inv=[-1.25,2.25]. The matrices
showed why: M_inv entries ~0.97 vs M_dag ~0.57 — S^{-1} inflates every overlap toward its
undamped value.

---

## Step 10 — The gamma scan, a self-inflicted bug, and a physical objection

Ran a cheap gamma scan (n=4, maxdim=64) to see WHERE the two diverge (`sinv_gamma_scan.jl`).
First attempt FAILED the built-in gamma=0 check (c_dag != c_inv where they must be equal) —
my hand-rolled F' sandwich ignored the clock synchronization and used a wrong contraction
convention. Fixed by reusing `build_open_F`'s exact clock-sync loop with only the inverse
operator swapped in; the gamma=0 check then passed exactly (maxdiff = 0).

The corrected scan showed c_inv smooth and monotonic, mean(M_inv) FROZEN at ~0.999 for all
gamma, while c_dag jumped around non-monotonically and mean(M_dag) fell 0.999 -> 0.568.

**The user raised a sharp physical objection:** c_dag sometimes weighted k=3 MORE than k=8
(e.g. [0.96,0.04]), which is wrong — k=8 has the least Trotter error and should dominate.
This did not look like mere noise.

---

## Step 11 — Chasing the objection: is the coefficient computation buggy?

`coeff_sanity.jl` (n=4, maxdim=64) checked every link:
- **[A]** library single-Trotter error formula gave E_k3 < E_k8 (VIOLATED) and negative
  errors.
- **[B]** DIRECT errors gave the correct ordering: E_k8 = 2.2e-4 << E_k3 = 1.3e-2. The
  formula disagreed with ground truth by >10x and in ordering — the formula was corrupted
  (same root cause as Step 7).
- **[C]** coefficients swung wildly with k_ref (40->160): [0.96,0.04] -> [0.23,0.77] ->
  [-0.42,1.42].

At this point I mis-diagnosed it as "k_ref=40 is an under-converged reference." The next step
corrected that.

---

## Step 12 — k_ref convergence: the mis-diagnosis corrected

`kref_convergence.jl` (n=4, gamma=0.05, DIRECT overlaps) swept k_ref = 40..640:

![k_ref convergence](fig_kref_convergence.svg)

    c: [-0.302,1.302] -> [-0.309,1.309] -> [-0.310,1.310] -> [-0.311,1.311] -> [-0.312,1.312]
    ref change: ~1e-7 from k_ref=80 on ; coeff change ~1e-3 (truncation noise)
    E_k8 << E_k3 at every k_ref (physical ordering holds)

So via the DIRECT route the coefficients are ROCK-STABLE and physically correct, and the
reference is already converged at k_ref=40. **The wild swings in Step 11 were NOT an
under-converged reference** — they were the SANDWICH M/L truncation corruption, worst at
n=4/maxdim=64 because that sits exactly at the truncation ceiling. (Honest note: over Steps
10-12 the diagnosis moved from "conditioning noise" to "bad reference" to "sandwich
corruption"; this step is what disentangled them.)

---

## Step 13 — Head-to-head at the study size (n=5): the surprise

`direct_vs_sandwich.jl` computed M, L, and c BOTH ways at n=5, maxdim=256, gamma=0.05:

    M: max|dM| = 1.0e-3 ; L: max|dL| = 7.3e-4
    c_sandwich = [-0.142,1.142] ; c_direct = [-0.145,1.145] ; max|dc| = 2.7e-3
    error achieved: 2.689e-5 (sandwich) vs 2.683e-5 (direct) — essentially equal
    E_k8 = 1.9e-4 << E_k3 = 1.05e-2 (physical)
    build time: sandwich ~10548 s (~2.9 h) vs direct ~28 s

![sandwich vs direct at n=5](fig_sandwich_vs_direct.svg)

**At the actual study size the two AGREE.** The n=4 catastrophe did NOT reproduce at n=5,
because maxdim=256 gives headroom above the true entanglement whereas n=4/maxdim=64 sat at
the ceiling. So: **the n=5 sweep coefficients are correct and stand.** The sandwich route is
more fragile than direct near the ceiling, but not wrong at n=5. (This walked back the
alarm from Steps 11-12 honestly.)

---

## Step 14 — The decisive constraint on the direct route

The direct route is 375x faster and cleaner — but the user made the key point: **using it
defeats the purpose.** If you can efficiently evolve the full state classically, you can read
off any observable directly and never need DMPF or a QPU. So the direct route presupposes
exactly the capability whose absence justifies the whole method. It is therefore disqualified
as a production route (a logical constraint, not an efficiency one) and kept only as a
VALIDATION ORACLE.

This reframed the project's real target: the classical preprocessing for the coefficients
must be cheaper than full state simulation AND not simply be state simulation. The current
F-sandwich is correct but above that cost threshold; the direct route is below the threshold
but on the wrong side of the logical line.

---

## Where we are now

- Cluster workflow, BLAS pinning, and the maxdim rule are solved and documented.
- The sweep is de-duplicated and uses a numerically-stable (direct-norm) ERROR estimator.
- The MOC closed-vs-open experiment cleanly demonstrates why the open system is hard.
- The n=5 DMPF coefficients are validated as correct (sandwich ~ direct).
- Two shortcuts are ruled out with clear mechanisms: direct overlaps (self-defeating), S^-1
  (blind to dissipation).
- The direct Liouville-MPO method is effectively infeasible at n=6 on this hardware.

---

## Outlook — where to go next

Open problem, sharpened: **compute the DMPF scalars M and L by a classical method that is
BOTH correct AND cheaper than full state-based tensor-network simulation, in the dissipative
regime.** Closed system achieves this (MOC -> bond dim 1); the open system does not, because
dissipation destroys the near-identity cancellation.

Candidate directions (must beat state simulation, must not BE state simulation):
1. **Cancel the unitary bulk, keep only the dissipative correction in F.** The unitary part
   drives most of F's entanglement and cancels exactly; a dissipation-aware reference that
   removes it while leaving the weaker dissipative structure could keep F low-rank — the
   "right operator" S^-1 failed to be.
2. **Perturbative expansion in gamma.** Expand F around the cheap closed-system F_0 (bond dim
   1); leading corrections may be low-rank even if the full F is not. Exploits that only
   dissipation spoils the closed limit.
3. **Estimate only the needed scalars.** M is 2x2; one needs a handful of overlaps, not all
   of F. Explore contraction paths or stochastic/sketching estimators for those scalars that
   beat both full-F and full-state cost.

De-prioritized / discarded:
- Direct state overlaps — validation oracle only (Step 14).
- S^-1 surrogate — blind to dissipation (Steps 9-10).
- Kraus representation — de-prioritized in earlier project discussions.
- n=6 via direct Liouville-MPO — infeasible on this hardware without a cheaper representation.

Quick confirmatory task before relying broadly on the sandwich coefficients: check that
sandwich ~ direct agreement (Step 13) holds across a couple more gamma values / seeds at n=5.

---

## Script inventory
Same set referenced throughout, each with a matching `submit_*.sh` (16-core + pinned-BLAS):
`run_case.jl`; `run_sweep_case.jl`/`_v2`/`_v3`; `time_ref.jl`; `gram_convergence.jl`;
`verify_purity.jl`; `direct_norm_diag.jl`; `closed_moc_vs_plain.jl`/`open_moc_vs_plain.jl`;
`sinv_vs_sdag_coeffs.jl`; `sinv_gamma_scan.jl`; `coeff_sanity.jl`; `kref_convergence.jl`;
`direct_vs_sandwich.jl`; `plot_sweep.jl`; `plot_moc_compare.jl`.
