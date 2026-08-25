# N-route working set

Minimal file set for continuing the Trotter-error Gram matrix reformulation.
Full write-up and figures: `report.pdf` / `findings_N_reformulation.md`.

Drop these into `/scratch/$USER/bond_dim_study/oqs_generalization/` alongside the
existing project files. Nothing here modifies the existing code.

---

## Files

| file | purpose |
|---|---|
| `trotter_error_gram.jl` | **The library.** Defect construction, four-object recursion, `N`, coefficients, direct oracle, tracing. |
| `sanity_check.jl` + `submit_sanity.sh` | Fast regression test. **Run after any edit to the library.** |
| `comparison_study.jl` + `submit_comparison.sh` | M-route vs N-route at n=4 against the exact point. Writes `cmp_coefficients.csv`, `cmp_errors.csv`. |
| `spectra_study.jl` + `submit_spectra.sh` | Spectra of F, G, Xi, Psi, Phi + discarded-weight-vs-error. Writes `cmp_spectra.csv`, `cmp_weight_vs_error.csv`. Run **after** comparison. |
| `gamma_sweep.jl` + `submit_gamma_sweep.sh` | **The recommended next experiment.** Job array over gamma. Writes `gamma_<g>.csv`. |
| `make_figures.py` | Regenerates all five figures from the CSVs. `DATA=<dir> python3 make_figures.py` |

Everything else from the development phase (`gauge_diagnostic`,
`systematic_diagnostic`, `identity_localise`, `delta_microtest`,
`directsum_test`, `delta_fix_verify`, `run_N_route`) was diagnostic scaffolding.
Their conclusions are recorded in the report and their essential checks are
folded into `sanity_check.jl`. They are not needed again.

---

## Order to run

```bash
sbatch submit_sanity.sh        # minutes  -- must pass before anything else
sbatch submit_comparison.sh    # ~1-2 h
sbatch submit_spectra.sh       # ~15 min  -- after comparison
sbatch submit_gamma_sweep.sh   # array of 4, ~2-4 h each
python3 make_figures.py        # locally, once the CSVs exist
```

---

## Settings that matter (each of these was a real bug)

| setting | value | why |
|---|---|---|
| `exact_delta` | `true` (default) | Uses `alg="directsum"`. The default density-matrix `+` discarded 65% of `delta`'s weight. |
| `maxdim_G` | `-1` = same as `maxdim` (default) | `maxdim_G = maxdim/2` gave `dc = 9.4e-4` where `maxdim_G = maxdim` gave `1.0e-4`. G is **not** free to be sloppy. |
| `K_REF` | must equal `K0` | Otherwise the two routes target different references and the comparison mixes truncation with a systematic offset. |
| `K0` | 48 | Must be an integer multiple of every `k_j`. 40 is not divisible by 3. |
| `maxdim` | never above `16^min(l,n-l)` | 16 / 256 / 256 / 4096 for n = 3 / 4 / 5 / 6. A loose maxdim forces enormous intermediate tensors. |
| BLAS threads | pinned in **both** places | `JULIA_NUM_THREADS` alone does not control BLAS. |

---

## Numerical rules for this codebase

These cost four rounds of debugging. They apply anywhere two near-equal MPOs
are subtracted, which is the premise of the whole N-route.

1. **`+` on MPS/MPO defaults to `alg="densitymatrix"`** — it forms `rho = M M†`
   and diagonalises, so it carries only `sqrt(eps) ~ 1.5e-8` accuracy. For
   differences of near-equal operators use `alg="directsum"`, which takes **no**
   `cutoff`/`maxdim` (passing them raises a `MethodError`).

2. **`maxdim` governs loss in MPO addition; `cutoff` does nothing.** A large +
   small sum preserves the small term only if `maxdim >= chi(A) + chi(B)`.
   Measured: at `maxdim=16` the small term was lost with 95% relative error for
   cutoffs `1e-12`, `1e-14`, `1e-16` **and exactly `0`** — identical results.

3. **`sqrt(inner(X,X))` cannot resolve a difference below `sqrt(eps)*||A||`.**
   Any "verified to 1e-15" claim about a difference of large MPOs is an
   artifact. This misled us twice in opposite directions.

4. **Include chain:** `include("F_diagnostics.jl")` pulls the whole stack.
   Including `open_middle_out_contraction.jl` directly leaves
   `vectorized_initial_state_mps` and `middle_bond_dim` undefined.

5. **`build_open_F` takes a LIST of k's** and returns `Vector{MPO}`. Passing a
   scalar silently returns a 1-element vector.

6. Including both `trotter_error_gram.jl` and `open_optimization_problem.jl`
   loads the chain twice and produces `redefinition of constant Main.ID2`
   warnings. Harmless (identical values), unavoidable when both routes are
   needed in one script.

---

## Known open issues

1. **Floor at `max|dc| ~ 1.0e-4` at `chi=256`, where nothing is truncated.**
   Ruled out: `op_dag` (3e-15), reference pre-composition (8.5e-7), `cutoff`
   (1e-12 -> 1e-15 moves it only 1.6 -> 0.97e-4), the `delta` construction
   (bounded at 1e-8 by the measurement floor). Cause unidentified, likely
   upstream in `A` or `B`. **Not worth chasing** — it is four orders below the
   error of the route it replaces.
2. **Non-monotonic convergence in both routes**, pronounced bump at `chi=128`.
   Unexplained; a referee will ask.
3. **The reference is not converged.** `k0: 48 -> 96` shifts `E_k8` by +6.3%.
   The n=4 "exact" point is exact = untruncated, not physically correct.
4. **The k=3 branch is badly conditioned**: `||delta_3||/||A_3|| = 9.3e-2` vs
   `6.0e-3` for k=8, because the k=3 block spans `t/3 = 1.0`. Fix: subdivide
   the coarse step so every defect spans a short interval.
5. **Only gamma = 0.05 tested.** This is what `gamma_sweep.jl` addresses.

---

## What the reformulation does and does not do

**Does:** removes the catastrophic cancellation; `E_mpf, E_kj >= 0` structurally;
20-30x more accurate at every bond dimension; accuracy governed by a bound
rather than an uncontrolled error correlation; eliminates `L` and `P`.

**Does not:** reduce bond dimension (`Phi` is *higher* rank than `F` — effective
rank 193 vs 44, and it saturates at the first Trotter step); restore the
closed-system near-identity cancellation; reduce wall-clock cost (770 s vs
~80 s at n=4, chi=256); resolve `E_mpf` itself (unresolved in both routes,
because `N` is near rank-1).

**Honest headline:** *accurate at chi=32 where the current method returns a
negative squared norm at chi=64.* Conditioning and correctness, not efficiency.

For an asymptotic efficiency claim, neither structural obstacle is addressed
here. Candidates: the KMS/detailed-balance inner product (under which
`G -> exp(2t L_D)` becomes a genuine contraction, bond dimension 1 for on-site
dissipators — and the **dephasing test is nearly free**, since KMS = HS for
unital dissipators), the gamma-expansion, or temporal/influence-matrix
contraction.
