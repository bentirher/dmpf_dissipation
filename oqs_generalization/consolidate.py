#!/usr/bin/env python3
"""Consolidated analysis of every tagged Step 2 run. No simulation.

Reads, from DATA (default .):
    step2_scaling_g{gamma}.csv       n=8 gamma sweep      (submit_gamma_sweep.sh)
    step2_scaling_n10g{gamma}.csv    n=10 gamma sweep     (submit_step2_n10_gamma.sh)

and fixes the two problems that the per-run summaries could not:

 1. SIGN-CANCELLATION NOISE. A single observable's error passes through zero as
    <O> crosses the exact value, so err_dmpf dips BELOW its own floor (8.9e-6
    against err_proj = 2.1e-5 at n=10, gamma=0.01, chi=128) and err_proj itself
    breaks trend (2.7e-7 at gamma=0.20 where E_mpf says it should be ~1e-6).
    Averaging |error| over every single-site Z_m damps this. The per-site rows
    Z1..Zn are already in every CSV written since the dense run; they were just
    never averaged. This is the 'Z_MAE' metric.

 2. CONFOUNDED TARGETS. chi* was scored at 1.5 x err_proj, which differs by up
    to 25x between n=8 and n=10 at the same gamma, so chi* values at different n
    were measured at different accuracies. Here chi* is ALSO computed at FIXED
    absolute targets, which makes n=8 vs n=10 an honest comparison.

chi* for the classical curve uses a Theil-Sen power-law fit of err against
eps_chi (monotone, noise-free), inverted for the target -- robust to the dips.
chi* for DMPF uses grid interpolation (its curve is flat once converged, so a
fit is unnecessary and can mislead).

Usage:  python3 consolidate.py              # tables
        python3 consolidate.py --csv        # also write consolidated_*.csv
Env:    DATA  TARGETS (default "1e-4,3e-5,1e-5")  CHI_MIN (default 16)
        FAMILY (default 4-8-16)
"""
import csv, glob, math, os, re, sys
from collections import defaultdict

U = os.environ.get("DATA", "./")
CHI_MIN = int(os.environ.get("CHI_MIN", 16))
FAMILY = os.environ.get("FAMILY", "4-8-16")
TARGETS = [float(x) for x in os.environ.get("TARGETS", "1e-4,3e-5,1e-5").split(",")]
WRITE = "--csv" in sys.argv

# ------------------------------------------------------------------ loading --
files = []
for pat, n_hint in [("step2_scaling_g*.csv", 8), ("step2_scaling_n10g*.csv", 10)]:
    for f in sorted(glob.glob(os.path.join(U, pat))):
        m = re.search(r"g([0-9.]+)\.csv$", f)
        if m:
            files.append((f, float(m.group(1)), n_hint))
if not files:
    sys.exit(f"no step2_scaling_g*.csv or step2_scaling_n10g*.csv in {U}")

print(f"found {len(files)} runs:")
for f, g, n in files:
    print(f"  n={n:<3} gamma={g:<5} {os.path.basename(f)}")
print()


def _f(x):
    """float() that maps NaN/blank to None (eps_chi is NaN under PROTOCOL=capped)."""
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def theil_sen(xs, ys):
    sl = sorted((ys[j] - ys[i]) / (xs[j] - xs[i])
                for i in range(len(xs)) for j in range(i + 1, len(xs)) if xs[j] != xs[i])
    if not sl:
        return None, None
    m = len(sl); s = sl[m // 2] if m % 2 else 0.5 * (sl[m // 2 - 1] + sl[m // 2])
    ic = sorted(y - s * x for x, y in zip(xs, ys)); k = len(ic)
    return s, (ic[k // 2] if k % 2 else 0.5 * (ic[k // 2 - 1] + ic[k // 2]))


def chi_star_fit(pts, target):
    """Classical chi*: robust power-law fit, inverted for the target.

    Preferred abscissa is the discarded weight eps_chi, which is monotone and
    essentially noise-free. Under PROTOCOL=capped there is no eps_chi (the
    dissipator gates are not norm preserving, so per-gate discarded weight is
    not recoverable by the norm trick) and it is written as NaN; we then fit
    against chi itself, which is noisier but is also the axis every claim is
    stated on. Either way we never extrapolate beyond the observed range."""
    d = [(c, e, y) for c, e, y in pts if y > 0 and c >= CHI_MIN]
    if len(d) < 4:
        return None
    use_eps = all(e is not None and math.isfinite(e) and e > 0 for _, e, _ in d)
    xs = [math.log(e if use_eps else c) for c, e, _ in d]
    ys = [math.log(y) for _, _, y in d]
    s, a = theil_sen(xs, ys)
    if s is None or s == 0:
        return None
    # err falls with chi (s < 0) and rises with eps (s > 0)
    if (use_eps and s <= 0) or ((not use_eps) and s >= 0):
        return None
    lx = (math.log(target) - a) / s
    if not use_eps:                      # lx is already log(chi)
        lo, hi = min(xs), max(xs)
        return math.exp(lx) if lo <= lx <= hi else None
    q = sorted(((math.log(e), math.log(c)) for c, e, _ in d), reverse=True)
    for i in range(len(q) - 1):
        if q[i][0] >= lx >= q[i + 1][0]:
            f = (q[i][0] - lx) / (q[i][0] - q[i + 1][0])
            return math.exp(q[i][1] + f * (q[i + 1][1] - q[i][1]))
    return None


def chi_star_grid(pts, target):
    """DMPF chi*: the SUSTAINED crossing -- the smallest chi from which the error
    stays at or below target for every larger chi on the grid -- log-interpolated
    against the last point above target.

    Why sustained and not first: single-observable errors pass through zero, so
    the DMPF curve dips below target by luck and then rises again. At n=10,
    gamma=0.40, <Z_mid> reads 1.6e-5 at chi=16 and 6.7e-5 at chi=32; a
    first-crossing rule reports chi*=16 and a speedup of 6.9x, the sustained rule
    reports 67.6 and 1.6x. Only the second is defensible. Zero-error (ceiling)
    points are excluded throughout."""
    d = sorted((c, y) for c, _, y in pts if y > 0 and c >= CHI_MIN)
    if not d:
        return None
    # index of the first point after which EVERY remaining point is <= target
    k = None
    for i in range(len(d)):
        if all(y <= target for _, y in d[i:]):
            k = i
            break
    if k is None:
        return None
    if k == 0:
        return float(d[0][0])
    (cp, yp), (c, y) = d[k - 1], d[k]
    f = (math.log(yp) - math.log(target)) / (math.log(yp) - math.log(y))
    return math.exp(math.log(cp) + f * (math.log(c) - math.log(cp)))


# (n, gamma) -> metric -> list of (chi, eps, err_direct, err_dmpf, err_proj, err_single)
runs = {}
for f, g, n_hint in files:
    rows = [r for r in csv.DictReader(open(f)) if r["family"] == FAMILY]
    if not rows:
        print(f"  ! {os.path.basename(f)}: no rows for family {FAMILY}"); continue
    n = int(rows[0]["n"])
    by = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["at_ceiling"] == "yes":
            continue
        by[int(r["chi"])][r["obs"]] = r
    site_obs = sorted({o for c in by for o in by[c] if re.fullmatch(r"Z\d+", o)},
                      key=lambda s: int(s[1:]))
    have_mae = len(site_obs) >= 2
    M = {}
    for metric in (["Z_MAE"] if have_mae else []) + ["Z_mid", "ZZ_mid"]:
        pts = []
        for c in sorted(by):
            if metric == "Z_MAE":
                rs = [by[c][o] for o in site_obs if o in by[c]]
                if len(rs) != len(site_obs):
                    continue
                avg = lambda k: sum(float(x[k]) for x in rs) / len(rs)
                pts.append((c, _f(rs[0]["eps_chi"]), avg("err_direct"), avg("err_dmpf"),
                            avg("err_proj"), avg("err_best_single")))
            elif metric in by[c]:
                x = by[c][metric]
                pts.append((c, _f(x["eps_chi"]), float(x["err_direct"]), float(x["err_dmpf"]),
                            float(x["err_proj"]), float(x["err_best_single"])))
        if pts:
            M[metric] = pts
    runs[(n, g)] = M
    if not have_mae:
        print(f"  ! n={n} gamma={g}: no per-site Z rows -- Z_MAE unavailable, using Z_mid")

PRIMARY = "Z_MAE" if all("Z_MAE" in m for m in runs.values()) else "Z_mid"
print(f"primary metric: {PRIMARY}\n")

# ------------------------------------------------------- 1. floor per metric --
print("=" * 100)
print(f"1. THE FLOOR err_proj PER METRIC  (family {FAMILY})")
print("   Z_MAE should be MONOTONE in gamma where Z_mid is not -- that is the test")
print("   of whether averaging has removed the sign-cancellation noise.")
print("=" * 100)
ns = sorted({n for n, _ in runs}); gs = sorted({g for _, g in runs})
for metric in ["Z_MAE", "Z_mid", "ZZ_mid"]:
    print(f"\n  {metric}")
    print("   n  | " + " ".join(f"g={g:<9}" for g in gs))
    for n in ns:
        cells = []
        for g in gs:
            m = runs.get((n, g), {}).get(metric)
            cells.append(f"{m[0][4]:<11.3e}" if m else f"{'--':<11}")
        print(f"  {n:>3} | " + " ".join(cells))

# -------------------------------------------- 2. chi* at FIXED targets, n vs n --
print()
print("=" * 100)
print(f"2. chi* AT FIXED ABSOLUTE TARGETS, metric {PRIMARY}")
print("   The same accuracy at every n and gamma, so n=8 vs n=10 is a clean comparison.")
print("   '--' = not reachable within the observed range (no extrapolation).")
print("=" * 100)
out = []
for T in TARGETS:
    print(f"\n  target = {T:.0e}")
    print(f"  {'gamma':>6} | {'n':>3} {'chi_cl*':>9} {'chi_dmpf*':>10} {'speedup':>8} | floor ok?")
    print("  " + "-" * 60)
    for g in gs:
        for n in ns:
            m = runs.get((n, g), {}).get(PRIMARY)
            if not m:
                continue
            pts_cl = [(c, e, yd) for c, e, yd, _, _, _ in m]
            pts_hy = [(c, e, yh) for c, e, _, yh, _, _ in m]
            floor = m[0][4]
            ccl = chi_star_fit(pts_cl, T)
            chy = chi_star_grid(pts_hy, T) if floor < T else None
            sp = ccl / chy if (ccl and chy) else None
            out.append(dict(target=T, gamma=g, n=n, chi_classical=ccl, chi_dmpf=chy,
                            speedup=sp, err_proj=floor, metric=PRIMARY))
            f = lambda x, w=9: f"{x:>{w}.1f}" if x else f"{'--':>{w}}"
            print(f"  {g:>6} | {n:>3} {f(ccl)} {f(chy,10)} {f(sp,8)} | "
                  f"{'yes' if floor < T else 'NO: floor ' + format(floor, '.1e') + ' > target'}")
        print("  " + "-" * 60)

# ----------------------------------- 3. growth from n=8 to n=10 at fixed target --
print()
print("=" * 100)
print("3. GROWTH n=8 -> n=10 AT FIXED TARGET  (the scaling claim)")
print("   Prediction: chi_dmpf* roughly flat in n, chi_classical* growing.")
print("=" * 100)
for T in TARGETS:
    print(f"\n  target = {T:.0e}")
    print(f"  {'gamma':>6} | {'cl n=8':>8} {'cl n=10':>8} {'growth':>7} | "
          f"{'hy n=8':>8} {'hy n=10':>8} {'growth':>7}")
    for g in gs:
        a = {r["n"]: r for r in out if r["target"] == T and r["gamma"] == g}
        if 8 in a and 10 in a:
            c8, c10 = a[8]["chi_classical"], a[10]["chi_classical"]
            h8, h10 = a[8]["chi_dmpf"], a[10]["chi_dmpf"]
            gr = lambda x, y: f"{y / x:>7.2f}" if (x and y) else f"{'--':>7}"
            f = lambda x: f"{x:>8.1f}" if x else f"{'--':>8}"
            print(f"  {g:>6} | {f(c8)} {f(c10)} {gr(c8, c10)} | {f(h8)} {f(h10)} {gr(h8, h10)}")
print()
print("  NOTE: at n=8 a '--' for chi_classical* means the fitted target lies beyond the")
print("  last grid point below the ceiling (192) -- and the ceiling (256) is exact, so")
print("  the true value is BRACKETED in (192, 256]. The n=8 -> n=10 growth is then")
print("  bracketed in [chi_n10/256, chi_n10/192]: e.g. gamma=0.01 gives 1.70-2.26.")

if WRITE:
    with open("consolidated_chistar.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("\nwrote consolidated_chistar.csv")
