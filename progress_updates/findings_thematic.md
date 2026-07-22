# Open-System DMPF on Hyperion — Thematic Findings & Technical Record

**Scope.** This document records the theory, numerical results, and cluster/implementation
know-how developed while adapting the open-system dynamic multi-product formula (DMPF)
study to the DIPC Hyperion cluster. It is organized by theme. A companion document
(`findings_chronological.md`) tells the same story as the actual step-by-step road we
travelled. Both end with an Outlook section.

**One-line summary.** The open-system F object does not compress (its operator-Schmidt
spectrum saturates the bond-dimension ceiling), which makes the current classical
preprocessing for DMPF coefficients *more* expensive than full state simulation. Two
would-be shortcuts were ruled out — direct state overlaps (logically self-defeating) and
an S^-1 surrogate (blind to dissipation). The core numerical routines were validated
(coefficients are correct at n=5) and a real error-estimator bug was fixed. The open
problem is sharpened: find a classical route to the coefficients that is cheaper than
state simulation in the dissipative regime.

---

## 1. Problem setting and the governing constraint

The open-system generalization of DMPF works in Liouville (vectorized) space. The central
object is

    F_ij = [S(t/k_i)^{k_i}]^dag  S(t/k_j)^{k_j}

where S(dt) is one Trotter step of the (dissipative) channel. From F one builds the Gram
matrix and overlap vector

    M_ij = Tr(rho_ki rho_kj) = <<rho_ki(t) | rho_kj(t)>>          (candidate-candidate)
    L_j  = Tr(rho(t)  rho_kj) = <<rho_ref(t) | rho_kj(t)>>         (reference-candidate)

and solves a small constrained least-squares problem for the DMPF coefficients c
(constraint sum(c)=1, via a Lagrange multiplier lambda).

**The governing constraint (why this is subtle).** On a QPU one cannot store rho(t); one
runs circuits (the Trotterizations) and measures overlaps. The classical preprocessing
that produces c must therefore be *cheaper than* full state-based tensor-network
simulation — otherwise there is no reason to use a QPU at all (you could just simulate the
state classically and read off any observable). This constraint is the yardstick for the
whole project: a coefficient method that requires efficient classical state evolution is
disqualified in principle, not merely inefficient.

---

## 2. Why the open-system F is hard: the near-identity cancellation and its loss

In the **closed** system S is unitary, so S^dag = S^{-1} exactly. When the middle-out
contraction (MOC) advances the two Trotter clocks in lockstep, F stays a *near-identity*
at every synchronization point — the forward and backward pieces cancel. A near-identity
MPO has a rapidly decaying operator-Schmidt spectrum, hence tiny bond dimension.

In the **open** system (gamma>0), S^dag is the adjoint of a non-unitary channel, NOT its
inverse. The cancellation is destroyed, F saturates the bond-dimension ceiling, and the
operator becomes expensive to store.

### Numerical demonstration (closed vs open, MOC vs plain multiplication)

We built F_ii (k_i=k_j, so the exact final F is the identity) two ways and tracked the
middle bond dimension per step, at gamma=0 and gamma=0.05, n=5, k=8, maxdim cap 256.

    step / sync      MOC closed (g=0)   MOC open (g=0.05)   PLAIN (both)
    1                1                  13                  16
    2                1                  168                 212
    3                1                  238                 253
    4                1                  248                 256
    5-8 (fwd)        1                  253 -> 256          256 (saturated)
    9-16 (bwd)       -                  -                   256 -> 113 (closed)
                                                            256 (open)

    Closed MOC max bond dim: 1     (whole pass ~30 s)
    Open   MOC max bond dim: 256   (per-step chi->256 SVDs, ~25 min)
    PLAIN  max bond dim:     256   (in both cases; forward half saturates)

![MOC vs plain, closed and open](fig_moc_vs_plain.svg)

**Interpretation.**
- Closed MOC is essentially free (bond dim 1 throughout) — exact near-identity maintained.
- Plain forward-then-backward multiplication builds the full time-t propagator in the
  middle (saturates to 256 by step 4) and only collapses near the end; it never had the
  advantage.
- Open MOC's advantage vanishes: it saturates to 256 like plain multiplication, because
  dissipation removes the cancellation.
- **Conclusion:** MOC is provably optimal where optimality exists (closed system); the
  open-system cost is intrinsic to dissipation, not an artifact of the contraction order.
  This doubles as an implementation check — MOC returning bond dim 1 (closed) and
  saturating (open) is exactly the signature of a correct dissipative step MPO.

Scripts: `closed_moc_vs_plain.jl`, `open_moc_vs_plain.jl` (single `dissipation`/`gamma`
toggle apart), plotted with `plot_moc_compare.jl`.

---

## 3. Bond-dimension scaling of the direct Liouville-MPO method

Tracking the middle bond dimension of F per Trotter layer (n=5, gamma=0.05, k=8):

    layer:      1    2    3    4    5    6    7    8
    bond_dim:   13   65   126  194  224  238  246  249     (ceiling 16^2 = 256)

- Bond dimension climbs toward the theoretical ceiling `theoretical_max_bond_dim(n) =
  16^min(n/2, n-n/2)` and nearly saturates; higher gamma pushes it harder.
- Per-layer time plateaus around ~33 s once chi nears saturation; memory peaked ~10-14 GB
  at n=5 (gamma-dependent).

**n=6 extrapolation.** The ceiling at n=6 is 4096 (16x larger chi). With SVD cost ~chi^3,
a saturated n=6 layer could be thousands of times slower than n=5 and require hundreds of
GB. **n=6 by the direct Liouville-MPO method is effectively infeasible on this hardware.**
This is itself a result and a motivation to change representation.

Scripts: `run_case.jl` (instrumented single case), driven per `(n, gamma)`.

---

## 4. The critical `maxdim` pathology (technical, high-impact)

**Symptom.** A single n=5 Trotter layer took **> 1 hour**; the run timed out at 8 h.

**Cause.** `apply` was called with a blanket `maxdim=4000`, while the true operator-Schmidt
ceiling at n=5 is only 256. Even though the *final* bond dimension never exceeds 256, a
loose `maxdim` forces `apply` to build and SVD-factorize enormous *intermediate* tensors
before truncating. The `4000` told ITensor not to cap those intermediates.

**Fix.** Set `maxdim = theoretical_max_bond_dim(n)` (16 / 64 / 256 / 4096 for n = 3/4/5/6),
never a loose upper bound. This dropped the whole n=5 case from ">8 h, timed out" to
**~4 minutes**.

**Take-away for all future runs:** the `maxdim` passed to `apply` matters even when the
final rank is far below it, because it controls intermediate SVD size.

---

## 5. Cluster workflow and environment know-how

### Submission model
- **Do not use interactive `srun --pty` for production** — it dies with the SSH/VS Code
  connection. Use `sbatch` scripts.
- **Work from `/scratch/$USER/...`**, the fast filesystem intended for job I/O. It is NOT
  backed up — move final results to `/data` or home when done.
- **Job arrays** map one independent parameter combination per task. All tasks in one array
  share the same resource request, so split cases with very different memory/time needs.
- **Right-size from measurement:** run a short instrumented test (10-min `test` QoS, or a
  1 h `regular` job) first, read per-block timings, THEN size the production run. Repeatedly,
  doubling `--time` and hoping wasted a day; a 10-minute instrumented test gave the answer.

### The single biggest performance bug: BLAS thread oversubscription
- Julia and OpenBLAS detect the whole physical node (e.g. 48-128 cores), not the
  `--cpus-per-task` SLURM allocated. Left unset, BLAS spawned ~64 threads onto a 4-core
  allocation, causing contention that turned a ~68 s build into a **>10-min hang**.
- **Fix (both required):**
  - In Julia: `BLAS.set_num_threads(parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")))`
  - In the batch script: `export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK`
  - Setting only `JULIA_NUM_THREADS` does NOT control BLAS.
- After the fix, **16 cores with pinned BLAS** was a good operating point for chi<=256
  dense-ish SVD workloads. More cores rarely helped (sequential Trotter steps; the heavy
  lifting is within-SVD BLAS).

### Julia gotchas discovered the hard way
- **Soft scope:** variables mutated inside a top-level `for` loop (e.g. `F`, accumulators)
  must be declared `global` inside the loop, else you get `UndefVarError`.
- **`do`-block argument order:** `f(...) do ... end` passes the do-block as the FIRST
  positional argument, so a timing helper must be `timed(f, label)`, not `timed(label, f)`.
- **Name collisions:** `dim` is exported by ITensors, Distributions, NDTensors, PDMats.
  Qualify as `ITensors.dim(...)` in standalone scripts.
- **Standard-library modules** (`Dates`, `LinearAlgebra`, `Random`) need no `Project.toml`
  entry; only third-party packages do.
- Do not leave `module spider julia` inside a batch script (it clutters logs / can return
  nonzero); it is an interactive lookup command.

### Instrumentation discipline
- Wrap every expensive block with flushed, timestamped prints (`flush(stdout)` matters on
  SLURM, else output buffers until the end). Per-layer / per-block timing lets a short test
  extrapolate a long run.

### Standard submission pattern
```
git add <files>; git commit -m "..."; git push
cd /scratch/$USER/.../oqs_generalization && git pull
sbatch submit_<x>.sh
tail -f logs/<x>_<jobid>.out
# after: sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode ; seff <jobid>
```

---

## 6. The DMPF truncation sweep: cost structure and optimization

The sweep evaluates the DMPF error vs candidate maxdim (md) against a reference at
maxdim_ref, k_ref_eval.

- **Reference builds dominate.** A single maxdim=256 Gram build at n=5 takes ~70-90 min
  (it builds F for all pairs in ks=[3,8], each an O(chi^4) sandwich). This is intrinsic.
- **The original sweep recomputed the md-INDEPENDENT references (M_ref, L_ref, purity) once
  PER md** — effectively 5-6 redundant ~70-min builds per task. **Fix: compute them once,
  reuse across all md.** This was the difference between "cannot finish one md in 24 h" and
  "whole sweep in ~3 h."
- **Gram convergence test:** M still drifts ~0.2% between maxdim 192 and 256 (relative dM:
  3.1e-2, 7.8e-3, 6.2e-3, 2.4e-3 across 64->96->128->192->256). So maxdim_ref cannot be
  safely lowered below 256 at n=5; cost scales ~chi^3 (128->256 was a 5.8x time increase).
- **Dropped the md=256 candidate** (comparing a 256 candidate against a 256 reference is
  degenerate). Sweep md in {16, 32, 64, 128}.

Scripts: `run_sweep_case.jl` (de-duplicated), `time_ref.jl` (per-block timing),
`gram_convergence.jl`.

---

## 7. The negative-error investigation and the direct-norm fix

**Symptom.** The DMPF error `E_mpf` came out NEGATIVE for several md (e.g. -6e-4 to -9e-4),
though it is a squared Frobenius norm and must be >= 0.

**Diagnosis (two hypotheses, first one wrong).**
1. *Purity mismatch (wrong):* purity was computed via a cheap O(chi^2) MPS evolution while
   L_ref used an O(chi^4) MPO sandwich; verified difference ~8.5e-4. But making purity
   consistent with L_ref made E_mpf MORE negative — so purity was only cosmetically masking
   the issue.
2. *Decomposition truncation (correct):* the identity
   `||rho_ref - mu||^2 = purity + c^T M c - 2 L^T c` holds only in EXACT arithmetic. M_ref,
   L_ref, purity are each INDEPENDENTLY truncated MPO sandwiches; their ~2e-3 truncation
   inconsistency (matching the Gram convergence floor) swamps a true error of ~5e-4 and
   flips the sign. Catastrophic cancellation.

**Decisive test.** Compute `||rho_ref - mu||^2` DIRECTLY from explicit MPS. At md=128:

    E_decomp (decomposition) = -1.494e-3
    E_direct (direct norm)   = +5.48e-4    (correct, non-negative)
    gap                      =  2.04e-3    (matches the truncation floor)

**Fix.** Evaluate E_mpf via the direct norm (numerically stable; guaranteed >= 0). This is
the standard stable form of ||a-b||^2 vs ||a||^2+||b||^2-2<a,b> when the result is a small
difference of large numbers. Note for collaborators: this is a change to how a
published-method quantity is *evaluated*, and should be documented, not swapped silently.

Scripts: `verify_purity.jl`, `direct_norm_diag.jl`, `run_sweep_case_v2.jl` (failed purity
fix), `run_sweep_case_v3.jl` (direct-norm sweep).

---

## 8. Are the coefficients correct? (physical objection -> full validation)

**The objection (physically motivated).** In a gamma scan the S^dag coefficients sometimes
weighted the k=3 circuit MORE than k=8 (e.g. c=[0.96, 0.04] at gamma=0.05), which is
physically wrong: k=8 has the least Trotter error and should be weighted most.

**What we found, in order.**
- **coeff_sanity at n=4, maxdim=64:** the library single-Trotter error formula
  `purity + M[j,j] - 2L[j]` gave E_k3 < E_k8 (VIOLATED) and even negative errors, while the
  DIRECT errors gave the correct ordering E_k8 (2.2e-4) << E_k3 (1.3e-2). The formula was
  corrupted (same catastrophic-cancellation root cause as Section 7). Coefficients from the
  sandwich route swung wildly with k_ref.
- ![k_ref convergence](fig_kref_convergence.svg)

  **k_ref convergence (n=4, gamma=0.05, DIRECT overlaps):** coefficients are ROCK-STABLE
  across k_ref = 40..640 (c ~ [-0.31, 1.31], k=8-dominant, physically correct), and the
  reference is already converged at k_ref=40 (ref change ~1e-7). So the wild swings in the
  sandwich route were NOT an under-converged reference — they were the sandwich M/L
  truncation corruption, worst at n=4 because maxdim=64 sits exactly at the truncation
  ceiling.
- **Head-to-head at n=5, maxdim=256 (the study size):** sandwich and direct AGREE.

![sandwich vs direct at n=5](fig_sandwich_vs_direct.svg)

        M: max|dM| = 1.0e-3 ;  L: max|dL| = 7.3e-4
        c_sandwich = [-0.142, 1.142] ; c_direct = [-0.145, 1.145] ; max|dc| = 2.7e-3
        E achieved: c_sandwich 2.689e-5 vs c_direct 2.683e-5  (essentially equal)
        single-Trotter: E_k8 = 1.9e-4 << E_k3 = 1.05e-2  (physical ordering holds)
        SANDWICH build ~10548 s (~2.9 h) vs DIRECT ~28 s.

**Conclusions (stated carefully).**
- **The n=5 sweep coefficients are correct and stand.** The sandwich route is not producing
  wrong coefficients at the study size.
- The sandwich route IS more fragile than direct near the truncation ceiling; the alarming
  n=4/maxdim=64 instability was a worst-case truncation regime, not a universal bug.
- The direct route is ~375x faster AND cleaner — but see Section 9: it is disqualified as a
  production route and usable only as a validation oracle.

Scripts: `sinv_gamma_scan.jl` (surfaced the objection), `coeff_sanity.jl`,
`kref_convergence.jl`, `direct_vs_sandwich.jl`.

---

## 9. Two ruled-out shortcuts (and why each fails)

### 9a. Direct state overlaps — logically self-defeating
Computing M and L by explicitly evolving rho(t) as an MPS and taking overlaps is fast and
numerically clean, but it **presupposes efficient classical state simulation** — exactly
the capability whose absence justifies using a QPU. If you can evolve the state cheaply,
you can read off any observable directly and never need DMPF or a QPU. **Therefore the
direct route is disqualified as a production method** and retained ONLY as a validation
oracle / ground truth against which candidate methods are checked. This is a logical
constraint, not an efficiency one.

### 9b. S^-1 surrogate F' — blind to dissipation
Idea: replace the correct adjoint S^dag with the inverse S^{-1} in F. Since S^{-1}S = I
exactly, F' might recover the closed-system near-identity cancellation and be cheap to
store, IF it yields similar coefficients. Construction: only the dissipator gates change
(unitary layers already have adjoint = inverse); the inverse dissipator is
`exp(-dt L_diss) = inv(forward channel)`, well-conditioned per step (compounded
amplification stayed ~1.2, NOT a blow-up).

**Result (n=5, gamma=0.05):** coefficients differ drastically (c_dag=[-0.16,1.16] vs
c_inv=[-1.25,2.25]). Mechanism, visible in the matrices: mean(M_inv) is FROZEN at ~0.999
for all gamma, while mean(M_dag) falls 0.999 -> 0.568 with gamma. **The S^-1 Gram matrix is
essentially blind to dissipation** — backward-dissipation on the bra cancels
forward-dissipation on the ket in the overlaps, leaving them near their undamped value ~1.
So F' cannot reproduce the dissipation-dependent coefficients that S^dag correctly
captures. A gamma scan confirmed the disagreement is structural (present from the smallest
gamma>0), not a threshold effect.

**Silver lining:** the property that made S^-1 cheap (recovered cancellation) is exactly the
property worth chasing — just with an operator that stays sensitive to dissipation (see
Outlook).

Scripts: `sinv_vs_sdag_coeffs.jl`, `sinv_gamma_scan.jl`.

---

## 10. Outlook — where to go next

The open problem, sharpened: **compute the DMPF coefficients (the scalars M and L) by a
classical method that is BOTH correct AND cheaper than full state-based tensor-network
simulation, in the dissipative regime.** The closed system achieves this (MOC -> bond dim
1); the open system does not, because dissipation destroys the near-identity cancellation.
The direct-overlap route is fast but on the wrong side of the logical line (Section 9a);
the current F-sandwich is on the right side of the line but above the cost threshold.

Candidate directions (respecting the constraint: beat state simulation, do not BE state
simulation):

1. **Cancel the unitary part, keep only the dissipative correction in F.** The unitary
   evolution drives most of F's entanglement and cancels exactly (closed-system result). A
   dissipation-aware reference that cancels the unitary bulk while leaving only the (weaker)
   dissipative structure in F could keep F low-rank while still tracking dissipation — the
   "right operator" that S^-1 failed to be.

2. **Perturbative / low-order-in-gamma expansion.** Dissipation is often weak (gamma=0.05).
   Expand F around the closed-system F_0 (cheap, bond dim 1) in powers of gamma; the leading
   corrections may be low-rank even though the full F is not. Directly exploits that the
   closed limit is free and only dissipation spoils it.

3. **Estimate only the needed scalars, not the full F.** M is just 2x2 (for ks=[3,8]); one
   needs a handful of scalar overlaps, not all of F. Explore contraction paths or
   stochastic / sketching estimators for those specific scalars that beat both full-F and
   full-state cost.

**Explicitly de-prioritized / discarded:**
- **Direct state overlaps** — validation oracle only (Section 9a).
- **S^-1 surrogate** — blind to dissipation (Section 9b).
- **Kraus-operator representation** — de-prioritized in earlier project discussions as not
  seemingly promising; not pursued here.
- **Pushing n=6 with the direct Liouville-MPO** — effectively infeasible on this hardware
  (Section 3); would need a fundamentally cheaper representation first.

**Also worth a quick confirmatory pass:** verify sandwich ~ direct coefficient agreement
(Section 8) holds across a couple more gamma values / seeds at n=5, to be airtight that the
single tested point is not a fluke before relying on the sandwich coefficients broadly.

---

## Appendix: script inventory

Diagnostics / studies (each with a matching `submit_*.sh`, all using the 16-core +
pinned-BLAS pattern):
- `run_case.jl` — instrumented single (n, gamma) bond-dimension case.
- `run_sweep_case.jl` / `_v2` / `_v3` — DMPF sweep: de-duplicated; failed purity fix;
  direct-norm error estimator (v3).
- `time_ref.jl` — per-block timing of the sweep's expensive pieces.
- `gram_convergence.jl` — M vs maxdim convergence.
- `verify_purity.jl`, `direct_norm_diag.jl` — the negative-E_mpf investigation.
- `closed_moc_vs_plain.jl`, `open_moc_vs_plain.jl` — MOC vs plain, closed & open.
- `sinv_vs_sdag_coeffs.jl`, `sinv_gamma_scan.jl` — the S^-1 surrogate experiments.
- `coeff_sanity.jl`, `kref_convergence.jl`, `direct_vs_sandwich.jl` — coefficient validation.
- `plot_sweep.jl`, `plot_moc_compare.jl` — plotting.
