#!/usr/bin/env python3
"""Full re-analysis of an existing step2_scaling.csv. No simulation, no reruns.

Four sections:

  1. MATCHED-chi COMPARISON   the actual result: classical vs DMPF vs the
                              zero-classical-cost baseline, at the same chi.
  2. CORRECTED chi*           fixes the log(0) bug that made every chi_direct*
                              read as the last grid point before the ceiling.
  3. CROSS-OBSERVABLE CHECK   does the advantage hold on ZZ_mid and Z_mean, or
                              only on the one observable that was printed?
  4. MECHANISM                corr, max|dc| vs sqrt(dc'N dc), dc_frac_null --
                              the columns that say WHY, which dropped out of the
                              printed table when the per-family columns went in.

Standard library only: no numpy, no matplotlib.

Usage:  python3 recompute_chi_star.py            # print the tables
        python3 recompute_chi_star.py --csv      # also write CSVs
Env:    DATA (dir, default .)  TAG  CHI_MIN (default 16)
"""
import csv
import math
import os
import sys
from collections import defaultdict

U = os.environ.get("DATA", "./")
TAG = os.environ.get("TAG", "")
SFX = ("_" + TAG) if TAG else ""
# Below this, the truncated state is not approximating anything: at n=8, chi=4
# gives err_direct = 1.4e-1 on a quantity of order 0.5. Ratios computed there
# compare against noise rather than against a competitor.
CHI_MIN = int(os.environ.get("CHI_MIN", 16))
WRITE = "--csv" in sys.argv

path = os.path.join(U, f"step2_scaling{SFX}.csv")
if not os.path.exists(path):
    sys.exit(f"missing: {path}\nSet DATA=<dir> (and TAG=<tag> if the run used one).")
rows = list(csv.DictReader(open(path)))
print(f"read {len(rows)} rows from {path}\n")


def chi_star(pairs, target):
    """Smallest chi reaching `target`, log-interpolated between brackets.

    THE FIX: a point with err == 0 is the ceiling run scored against ITSELF, not
    a measurement. Including it put log(0) = -Inf in the denominator, driving the
    interpolation weight to zero and returning the LOWER bracket -- which is why
    chi_direct* came back as 128.0 at both n=8 and n=10 for targets three orders
    below anything actually achieved there. Zero-error points are dropped; if the
    target is met only at one, the true chi* is at or just below the ceiling and
    the grid cannot resolve it, reported as 'ceiling-only'.
    """
    pairs = sorted(pairs)
    real = [(c, e) for c, e in pairs if e > 0]
    zeros = [c for c, e in pairs if e <= 0]
    for i, (c, e) in enumerate(real):
        if e <= target:
            if i == 0:
                return float(c), "grid"
            cp, ep = real[i - 1]
            if ep <= target:
                return float(c), "grid"
            f = (math.log(ep) - math.log(target)) / (math.log(ep) - math.log(e))
            return math.exp(math.log(cp) + f * (math.log(c) - math.log(cp))), "interp"
    if zeros:
        return float(min(zeros)), "ceiling-only"
    return float("inf"), "off-grid"


# Index everything by (n, family, observable).
D = defaultdict(lambda: defaultdict(list))
for r in rows:
    D[(int(r["n"]), r["family"])][r["obs"]].append(r)
for key in D:
    for obs in D[key]:
        D[key][obs].sort(key=lambda x: int(x["chi"]))
keys = sorted(D)
obs_all = sorted({r["obs"] for r in rows})
PRIMARY = "Z_MAE" if "Z_MAE" in obs_all else "Z_mid"
print(f"observables present: {', '.join(obs_all)}")
print(f"primary metric: {PRIMARY}\n")

# ---------------------------------------------------------------- section 1 --
print("=" * 104)
print(f"1. MATCHED-chi COMPARISON   metric={PRIMARY}, chi >= {CHI_MIN}, ceiling rows excluded")
print("=" * 104)
print(f"{'n':>3} {'family':>10} {'chi':>6} | {'classical':>11} {'DMPF':>11} {'free':>11}"
      f" | {'vs classical':>12} {'vs free':>9}")
print("-" * 104)
matched = []
for n, fam in keys:
    pts = D[(n, fam)].get(PRIMARY, [])
    for r in pts:
        chi = int(r["chi"])
        if r["at_ceiling"] == "yes" or chi < CHI_MIN:
            continue
        ed, eh = float(r["err_direct"]), float(r["err_dmpf"])
        es = float(r["err_best_single"])
        rec = dict(n=n, family=fam, chi=chi, err_direct=ed, err_dmpf=eh,
                   err_best_single=es, vs_classical=ed / max(eh, 1e-300),
                   vs_free=es / max(eh, 1e-300))
        matched.append(rec)
        print(f"{n:>3} {fam:>10} {chi:>6} | {ed:>11.4e} {eh:>11.4e} {es:>11.4e}"
              f" | {rec['vs_classical']:>12.2f} {rec['vs_free']:>9.2f}")
    print("-" * 104)
print("  'free' = best single Trotter circuit, i.e. the answer with ZERO classical")
print("  computation. With sum(c)=1 and |c| bounded, any normalised combination of")
print("  candidates lands near the candidates themselves, so DMPF cannot do much")
print("  WORSE than this regardless of the coefficients. Beating it is the real test;")
print("  beating the classical simulation alone is not sufficient.\n")

# ---------------------------------------------------------------- section 2 --
print("=" * 104)
print("2. CORRECTED chi*   (target = 1.5 x err_proj; zero-error ceiling points excluded)")
print("=" * 104)
print(f"{'n':>3} {'family':>10} | {'target':>11} | {'chi_direct*':>12} {'how':>13}"
      f" | {'chi_dmpf*':>10} {'how':>13} | {'speedup':>8}")
print("-" * 104)
stars = []
for n, fam in keys:
    pts = D[(n, fam)].get(PRIMARY, [])
    if not pts:
        continue
    target = 1.5 * float(pts[0]["err_proj"])
    # MONOTONE ENVELOPE on the classical curve. err_direct passes through zero as
    # the observable's sign flips, which produces isolated dips: at n=10 the
    # chi=384 point reads 2.7e-6 while err_direct/sqrt(eps_chi) is 0.3-0.4 at
    # every other chi and 0.013 there. Interpolating chi* through such a point
    # anchors the answer on noise and UNDERSTATES the classical requirement --
    # 319 instead of ~590 at n=10. Taking a running minimum from the right makes
    # the curve non-increasing, which is what the underlying convergence is.
    def envelope(pairs):
        pairs = sorted(pairs)
        out, run = [], float("inf")
        for c, e in reversed(pairs):
            run = e if e <= 0 else min(run, e)
            out.append((c, run))
        return list(reversed(out))

    csd, hd = chi_star(envelope([(int(r["chi"]), float(r["err_direct"])) for r in pts]), target)
    csh, hh = chi_star([(int(r["chi"]), float(r["err_dmpf"])) for r in pts], target)
    ok = all(math.isfinite(x) for x in (csd, csh)) and hd == "interp" and hh == "interp"
    sp = csd / csh if ok else float("nan")
    stars.append(dict(n=n, family=fam, target=target, chi_direct_star=csd, how_direct=hd,
                      chi_dmpf_star=csh, how_dmpf=hh, speedup=sp))
    print(f"{n:>3} {fam:>10} | {target:>11.4e} | {csd:>12.1f} {hd:>13}"
          f" | {csh:>10.1f} {hh:>13} | {sp:>8.2f}")
print()
print("  speedup is reported ONLY when both sides were genuinely interpolated on the")
print("  grid. 'ceiling-only' means the target was reached only at the exact point, so")
print("  chi* sits at or just below the ceiling and this grid cannot resolve it -- that")
print("  is the state of play at n=8 and n=10 until the 128->ceiling gap is filled.\n")

# ---------------------------------------------------------------- section 3 --
print("=" * 104)
print("3. CROSS-OBSERVABLE CHECK   err_direct / err_dmpf, per observable")
print("=" * 104)
# Z_mean is EXCLUDED, and its exclusion is a physics statement, not a filter.
# The Hamiltonian conserves total magnetisation, so <sum_m Z_m> evolves under the
# dissipator alone -- and the dissipator layer is applied EXACTLY by every
# product formula. So every candidate reproduces Z_mean to round-off, any
# combination with sum(c)=1 does too, and err_dmpf is round-off divided into a
# real truncation error. That is where the ratios of 1e9 to 5e10 come from. It
# is a good check that the vectorisation and the dissipator sign are right; it is
# not evidence for the method.
DROP = {"Z_mean"}
show = [o for o in ["Z_MAE", "Z_mid", "ZZ_mid"] if o in obs_all and o not in DROP]
extra = [o for o in obs_all if o not in show and o not in DROP]
if extra:
    print(f"  (also present, not shown: {', '.join(extra)})")
print(f"{'n':>3} {'family':>10} {'chi':>6} | " + " ".join(f"{o:>11}" for o in show))
print("-" * 104)
for n, fam in keys:
    ref = D[(n, fam)].get(show[0], [])
    for r0 in ref:
        chi = int(r0["chi"])
        if r0["at_ceiling"] == "yes" or chi < CHI_MIN:
            continue
        cells = []
        for o in show:
            m = [x for x in D[(n, fam)].get(o, []) if int(x["chi"]) == chi]
            if not m:
                cells.append(f"{'--':>11}")
                continue
            ed, eh = float(m[0]["err_direct"]), float(m[0]["err_dmpf"])
            cells.append(f"{ed / max(eh, 1e-300):>11.2f}")
        print(f"{n:>3} {fam:>10} {chi:>6} | " + " ".join(cells))
    print("-" * 104)
print("  The advantage must hold across observables. If it appears on one and not the")
print("  others, it is a sign-flip artifact of that observable passing through zero,")
print("  not a property of the method.\n")

# ---------------------------------------------------------------- section 4 --
print("=" * 104)
print("4. MECHANISM")
print("=" * 104)
print(f"{'n':>3} {'family':>10} {'chi':>6} | {'relerr_N':>10} {'corr':>7}"
      f" | {'max|dc|':>10} {'sqrt(dcNdc)':>12} {'ratio':>9} | {'dc_null':>8} {'cond(N)':>10}")
print("-" * 104)
mech = []
for n, fam in keys:
    for r in D[(n, fam)].get(PRIMARY, []):
        chi = int(r["chi"])
        if r["at_ceiling"] == "yes":
            continue
        dcm = float(r["dc_max"])
        sq = math.sqrt(max(float(r["dcNdc"]), 0.0))
        rec = dict(n=n, family=fam, chi=chi, relerr_N=float(r["relerr_N"]),
                   corr_min=float(r["corr_min"]), dc_max=dcm, sqrt_dcNdc=sq,
                   ratio=dcm / max(sq, 1e-300), dc_frac_null=float(r["dc_frac_null"]),
                   cond_N=float(r["cond_N"]))
        mech.append(rec)
        print(f"{n:>3} {fam:>10} {chi:>6} | {rec['relerr_N']:>10.3e} {rec['corr_min']:>7.3f}"
              f" | {dcm:>10.3e} {sq:>12.3e} {rec['ratio']:>9.1f}"
              f" | {rec['dc_frac_null']:>8.3f} {rec['cond_N']:>10.2e}")
    print("-" * 104)
print("  ratio = max|dc| / sqrt(dc'N dc). Large means max|dc| massively overstates the")
print("  damage -- the quantitative form of the exact identity")
print("      E(c* + dc) = E_mpf + dc'N dc        (the cross term vanishes identically)")
print("  and the direct rebuttal of main.pdf's claim that truncation requirements must")
print("  be set by coefficient behaviour rather than by the achieved error.")
print()
print("  corr near 1 -> the truncation errors on rho_ref and rho_kj are the SAME error,")
print("  cancelling in Delta. That is the L-MPF eq. (21)-(23) mechanism. Near 0.5, as in")
print("  Step 1, it is present but partial and is not the whole explanation.")
print()
print("  dc_frac_null near 1 -> dc points along the near-null direction of N, where it")
print("  does least damage (L-MPF Appendix D). Meaningful ONLY for r >= 3: at r = 2 the")
print("  constraint sum(c)=1 leaves exactly one direction, so the statement is vacuous.")

if WRITE:
    print()
    for name, data in [("step2_matched", matched), ("step2_chistar", stars),
                       ("step2_mechanism", mech)]:
        if not data:
            continue
        out = f"{name}{SFX}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"wrote {out}")
