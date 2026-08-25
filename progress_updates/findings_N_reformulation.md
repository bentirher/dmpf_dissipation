# The Trotter-Error Gram Matrix
## A reformulation of the open-system DMPF coefficient problem

**Status:** implemented, tested at n=3,4,5. Companion to `findings_thematic.md`,
`moc_open_systems_summary.md`, `main.pdf`. Section refs "§7", "§8" point to
`findings_thematic.md`. Figures and full write-up: `report.pdf` / `report.tex`.

---

## 0. One-paragraph summary

The DMPF coefficients were being obtained by combining three separately computed objects — the
Gram matrix `M`, the overlap vector `L`, and the purity `P` — each of order 0.57, whose
combination is of order 1e-4. That is a catastrophic cancellation. The cost function can be
rewritten exactly so the small quantity — the **Trotter-error Gram matrix** `N` — is computed
directly and the large one is never formed. Measured at n=4 against an exact reference: the old
assembly step **amplifies** truncation error by ~550x; the new route **de-amplifies** by ~5.5x.
Ratio 3018. The new route is 20-30x more accurate at every bond dimension and never returns a
negative squared norm; the old route does at 5 of 8 bond dimensions. **It does not reduce bond
dimension and does not restore the closed-system near-identity cancellation.**

---

## 1. WHAT IT HELPS WITH (demonstrated numerically)

| Problem | Resolution |
|---|---|
| **Negative `E_mpf`** (§7) | `N` is a Gram matrix of error vectors, hence PSD by construction. Measured `lambda_min`: M-route negative at 5/8 bond dims, worst -1.46e-2 vs true +2.81e-5 (520x, wrong sign); N-route negative at 2/8, worst -5.3e-6 (16% perturbation). |
| **`E_k3 < E_k8` inversion** (§8) | `E_kj = N_jj`, a diagonal Gram entry. Correctly ordered and positive at every bond dimension. |
| **Coefficients unstable in `k_ref`** (§8) | `L` and `P` eliminated; only `k0` remains. |
| **MPS-vs-MPO method mismatch** (§7) | `P` removed from the pipeline. *Note: `P` was never inaccurate* — a Liouville MPS has Schmidt rank <= 4^min(l,n-l) = 16, below every maxdim tested, so `reference_purity` was **exact**. The failure was `P` exact while `L` was truncated, which makes the gauge cancellation structurally unavailable. |
| **Accuracy per unit bond dimension** | 20-30x better at every chi. At chi=64: max\|dc\| = 3.0e-3 vs 9.1e-2. |
| **Robustness to aggressive truncation** | N-route discards up to 29% of Phi's Schmidt weight and still lands within 1e-3 of exact. |

## 2. WHAT IT DOES **NOT** HELP WITH (measured, negative)

| Claim | Verdict |
|---|---|
| **Bond dimension** | `chi_Phi` saturates the ceiling at the **first** Trotter step; `chi_G` takes ~13 steps to climb 111->256. Phi is **higher**-rank than F: effective rank at 1e-3 retained weight is **193 (Phi) vs 44 (F)**. Mechanism: `Delta_j = S_j - E` is a *difference* of propagators, and differences generically carry higher Schmidt rank than either operand. The same cancellation that makes \|\|Phi\|\| small makes chi_Phi large. |
| **Closed-system efficiency claim** | Not restored. `S†S != 1` for gamma>0 stands. The interleaved schedule here only avoids forming a one-sided full-time propagator (chi ~ e^{v1 t}); it does **not** keep the running object near identity. |
| **Wall-clock cost** | **Worse.** n=4, chi=256: 770 s vs ~80 s. Four tracked objects plus an MPO addition per step. Driver is `G`, not `delta`. |
| **`E_mpf` itself** | Unresolved in **both** routes. `N` is near rank-1 (all `drho_j` are Trotter errors of the same dynamics, hence nearly parallel), so `lambda_min` is tiny. `c` needs only the eigen*vector* direction and converges; `E_mpf` needs the eigen*value* magnitude and does not. |
| **Monotonic convergence** | Neither route converges monotonically. Both bump at chi=128. **Unexplained.** |

**Honest framing for the paper:** this is a *numerical conditioning and correctness* result, not
an efficiency result. "Accurate at chi=32 where the current method returns a negative squared
norm at chi=64" — not "cheaper".

---

## 3. Diagnosis

### 3.1 Magnitudes (n=4, gamma=0.05, t=3, ks=[3,8], order 2, exact chi=256)
```
M_ij, L_j, P  ~ 0.57
E_k3 = 1.33e-2   E_k8 = 2.50e-4   E_mpf = 3.76e-5   c = [-0.144708, 1.144708]
```

### 3.2 The gauge structure
On `sum(c)=1`, the perturbation `M -> M + u 1^T + 1 u^T + a 11^T`, `L -> L + u + a 1`,
`P -> P + a` leaves the cost **exactly** unchanged for any `u`, `a`. So `(M,L,P)` has r+1
redundant directions: r(r+1)/2 + r + 1 numbers encode only r(r+1)/2. The invariant content is
exactly `N_ij = M_ij - L_i - L_j + P`. Gauge directions carry ~0.57; invariant content carries
1e-4 to 1e-2.

### 3.3 Measured: the assembly amplifies, it does not cancel

| case | \|\|dM\|\| | \|\|dL\|\| | \|dP\| | \|\|dN\|\| | **amplification** | resid/gauge |
|---|---|---|---|---|---|---|
| n=4, 32->64 (ord 1) | 7.2e-2 | 7.5e-2 | 0 | 1.60e-1 | 1.54 | 0.91 |
| n=5, 64->256 (ord 1) | 1.7e-3 | 1.8e-3 | 0 | 5.0e-3 | 2.04 | 1.46 |
| n=5, 128->256 (ord 1) | 3.8e-3 | 1.1e-2 | 0 | 2.7e-2 | 2.40 | 0.92 |
| n=4, 32->64 (ord 2) | 2.6e-3 | 4.2e-3 | 0 | 9.6e-3 | 1.95 | 1.77 |
| n=4, 64->256 (ord 2) | 5.5e-3 | 2.2e-3 | 0 | 1.13e-2 | 1.90 | 0.67 |

Since `dP = 0` identically, `dN_ij = dM_ij - dL_i - dL_j`: each `dL` enters **twice with the
same sign**. Last row: 5.5e-3 + 2(2.2e-3) = 9.9e-3 vs measured 1.13e-2 — near-perfect
constructive addition. **There is no cancellation to rely on.** Note row 5 has resid/gauge=0.67:
a *majority* of that error was gauge, and it still destroyed the answer.

### 3.4 F and G are the same object

| | \|\|.\|\|_F | s_1 | s_1^2/tot | eff rank @1e-3 | s_2/s_1 |
|---|---|---|---|---|---|
| F = S†S | 9.3986 | 9.3317 | **98.58%** | 44 | 0.0499 |
| G = E†E | 9.3972 | 9.3340 | **98.66%** | 40 | 0.0474 |
| Xi = D†E | 0.1796 | 0.0599 | 11.1% | 129 | 0.981 |
| Phi = D†D | **0.0061** | 0.0036 | 35.4% | 193 | 0.380 |

```
sum (s_F - s_G)^2 / ||F||^2 = 2.24e-5        ||Phi||^2 / ||F||^2 = 4.2e-7
```
**Deck slides 22, 23, 27 were characterising E†E.** Slide 27 sharpens from "computing F is
unfeasible" to "**F is the wrong object: 99.99996% of its weight cancels**."

---

## 4. The reformulation

Because `sum(c)=1`, write `rho = (sum c_j) rho`, so

```
mu - rho = sum_j c_j [rho_kj - rho] = sum_j c_j drho_j
```

The residual is a combination of **errors**, not of **states**. Hence

```
C(c) = c^T N c ,   N_ij = Tr[drho_i drho_j] = <<rho0| Delta_i^dag Delta_j |rho0>>
c = N^-1 1 / (1^T N^-1 1) ,  E_mpf = 1/(1^T N^-1 1) ,  E_kj = N_jj
```

`F_ij = G + Xi_i + Psi_j + Phi_ij` with `G` of order 1 and i,j-independent (cancels), `Xi,Psi`
first order, only `Phi` surviving.

**Assumptions.** A1 Hermiticity (all scalars real). A2 `Tr rho_kj = 1` — holds because
`Tr(L[rho])=0` identically, so `exp(sL)` is trace-preserving for *every* real s, surviving the
negative substep of order 4 (§5.3's caveat is about positivity only). A3 adjoint not inverse.
A4 finite reference `E = S(t/k0)^k0`. A5 `k0` divisible by every `k_j`. A6 time-independent
generator. **L1:** the two forms agree *only* on `sum(c)=1`.

### 4.1 The four-object recursion

`A_j = S(t/k_j)`, `B_j = S(t/k0)^{k0/k_j}` (reference over the *same* duration),
`delta_j = A_j - B_j`.

| | right advance (one A_j) | left advance (one A_i) |
|---|---|---|
| G | `G B_j` | `B_i† G` |
| Xi | `Xi B_j` | `A_i† Xi + delta_i† G` |
| Psi | `Psi A_j + G delta_j` | `B_i† Psi` |
| Phi | `Phi A_j + Xi delta_j` | `A_i† Phi + delta_i† Psi` |

Initial `G=1`, `Xi=Psi=Phi=0`. Schedule: advance whichever clock is behind — identical to
`build_open_F`. Consistency: `(G+Xi)(B_j+delta_j) + (Psi+Phi)A_j = F A_j`, since
`B_j + delta_j = A_j`. **This identity is load-bearing and must hold numerically.**

**Why left AND right.** Each object is a sandwich. Running all right steps first leaves
`G = E(t_b)` — a bare full-time propagator, chi ~ g e^{v1 t}. Interleaving keeps objects
*balanced*. That is the only job the ordering does here; in the closed system it additionally
forces chi=1 via `U†U=1`, and that is gone for gamma>0.

---

## 5. Numerical pitfalls discovered (general, and costly)

Four debugging rounds went into these. They apply anywhere two near-equal MPOs are subtracted —
which is the premise of the N-route.

1. **`maxdim` governs loss in MPO addition; `cutoff` does nothing.** Adding a large and a small
   MPO preserves the small term only if `maxdim >= chi(A)+chi(B)`. Once rank truncation occurs,
   retained directions represent the *sum* (dominated by the large term). Measured: at
   maxdim=16 the small term was lost with 95% relative error for cutoffs 1e-12, 1e-14, 1e-16
   **and exactly 0** — identical results.
2. **`+` on MPS/MPO defaults to `alg="densitymatrix"`**, which forms `rho = M M†` and
   diagonalises, costing half the digits (`sqrt(eps) ~ 1.5e-8`). Use `alg="directsum"` for
   differences of near-equal operators; it takes **no** cutoff/maxdim (MethodError if passed).
3. **`sqrt(inner(X,X))` cannot resolve a difference below `sqrt(eps)*||A||`.** It computes a
   squared quantity by contraction. Any "verified to 1e-15" claim about a difference of large
   MPOs is an artifact. This bit us twice in opposite directions: a density-matrix subtraction
   reported 2.9e-15 for a true 1.5e-8.
4. **`maxdim` must never exceed `16^min(l,n-l)`** (16/256/256/4096 for n=3/4/5/6). Pin BLAS
   threads in **both** Julia and the batch script.

---

## 6. Open issues

1. **Floor at max|dc| ~ 1.0e-4 at chi=256, where nothing is truncated.** Ruled out: `op_dag`
   (3e-15); reference pre-composition (8.5e-7); cutoff (1e-12 -> 1e-15 moves it only
   1.6 -> 0.97e-4); the delta construction (bounded at 1e-8 by the measurement floor, giving
   <=5e-6 after amplification by cond(N)=483). `alg="directsum"` changed the sweep by <1%.
   **Cause unidentified; likely upstream in A or B.**
2. **Non-monotonic convergence in both routes**, pronounced bump at chi=128. Unexplained.
3. **The reference is not converged.** `k0: 48 -> 96` shifts `E_k8` by **+6.3%**. The n=4
   "exact" point is exact = *untruncated*, not *physically correct*.
4. **The k=3 branch is badly conditioned** (L5): `||delta_3||/||A_3|| = 9.3e-2` vs `6.0e-3` for
   k=8, because the k=3 block spans t/3 = 1.0. Fix: subdivide the coarse step.
5. **Only gamma=0.05 tested.** Slide 22 swept {0.01, 0.05, 0.1, 0.2}. Whether the 3018x
   advantage survives at gamma=0.2 is **the single most important open question**.

## 7. Recommended next steps

1. **gamma sweep** (highest value). n=4 head-to-head at gamma in {0.01, 0.1, 0.2}; plot
   max|dc| at fixed chi=32 vs gamma for both routes. Direct successor to slide 22.
2. **Subdivide the k=3 coarse step**, re-measure N_11.
3. **Do not chase the 1e-4 floor.** It is four orders below the error of the route it replaces,
   which additionally returns negative squared norms.
4. **For an asymptotic claim**, neither structural obstacle is addressed here (G saturates;
   Phi saturates because it is a difference). Candidates: KMS/detailed-balance inner product
   (under which `G -> exp(2t L_D)` becomes a genuine contraction, bond dim 1 for on-site
   dissipators — and the **dephasing test is nearly free**, since KMS = HS for unital
   dissipators), the gamma-expansion, or temporal/influence-matrix contraction.

---

## 8. Code

**New:** `trotter_error_gram.jl` (library), `gauge_diagnostic.jl`, `comparison_study.jl`,
`spectra_study.jl`, `systematic_diagnostic.jl`, `identity_localise.jl`, `delta_microtest.jl`,
`directsum_test.jl`, `delta_fix_verify.jl`, `make_figures.py`, plus `submit_*.sh`.

**Replaced:** `open_gram_matrix`, `open_L_vector`, `reference_purity`,
`dynamic_mpf_coefficients`, `open_dynamic_mpf_error`, `open_single_trotter_errors` — six
functions collapse to `trotter_error_gram` + `trotter_error_coefficients`.

**Reused unchanged:** `liouville_space.jl`, `open_product_formula_generation.jl`, the
`op_dag`/`left_multiply`/`right_multiply` helpers and two-clock schedule in
`open_middle_out_contraction.jl`, `bond_dimension_tracking.jl`, `F_diagnostics.jl`.

**Include chain:** `include("F_diagnostics.jl")` pulls the whole stack. Including
`open_middle_out_contraction.jl` directly leaves `vectorized_initial_state_mps` and
`middle_bond_dim` undefined.

**Key settings:** `exact_delta=true` (alg="directsum"), `maxdim_G = maxdim` (NOT maxdim/2 —
that was a bug in the original sweep that depressed every N-route number), `K0=48` (multiple of
both 3 and 8), `ORDER=2`.
