#!/usr/bin/env python3
"""
fit_cost.py -- turn the map CSVs into the resource curve that justifies a
system size.

Works two ways.

  From a notebook:
      from fit_cost import discover, fit, resource_table, walls
      maps = discover('../results/ckt_map_n*/map_n*_m*.csv')    # {8: df, 12: df, ...}
      f = fit(maps, theta=0.95, p=0.05)
      print(f.beta, f.a, f.resid_sd)        # ln chi = beta*n + a
      resource_table(f)                     # DataFrame: n, chi, memory, flops
      walls(f)                              # n at which each machine runs out

  From a shell:
      python3 fit_cost.py '../results/ckt_map_n*/map_n*_m*.csv'
      python3 fit_cost.py --theta 2.40 --p 0.12 ../results/ckt_map_n*/map_n*.csv

-----------------------------------------------------------------------------
WHAT IT DOES, in three steps.

1. Pull chi_mean at one (theta, p) cell from every map, one per system size, and
   DISCARD any point that is finite-size pinned.

   The pure-state half-cut Schmidt rank cannot exceed 2^(n/2). A cell sitting at
   that value is reporting the Hilbert space, not the circuit, and including it
   flattens the fitted slope. The cut is chi >= 0.9 * 2^(n/2). At theta = 0.95,
   p = 0.05 nothing is discarded (chi/ceiling runs 0.87 -> 0.63 as n goes 8 ->
   20); at p = 0 almost everything is, which is why the p = 0 column cannot be
   fitted at all.

2. Ordinary least squares of ln(chi) on n. The half-cut entropy grows linearly
   in n in this regime and chi ~ 2^S, so ln(chi) is linear in n -- that is the
   only modelling assumption, and the residuals test it. At theta = 0.95,
   p = 0.05 the four points give

       ln chi = 0.3198(6) n + 0.072,   residual sd 0.5%

   The slope to compare against is ln(2)/2 = 0.3466, the slope of the ceiling
   itself. 0.3198 is 92% of that: the circuit is very nearly as entangling as
   the Hilbert space allows, which is why no choice of theta improves on it.

3. Convert chi(n) into the two things that actually stop a classical run.

   MEMORY, and it is the binding one. One trajectory is an MPS of L sites,
   local dimension d = 2, bond chi, in complex128:

       mem = L * d * chi^2 * 16 bytes

   with L = 2n, because circuit_trajectory.jl interleaves one ancilla per system
   qubit and those tensors carry full bond dimension even though reset_ancilla!
   leaves them disentangled. (For a plain n-site MPS the same formula
   OVERestimates by ~8x, because only the middle bonds reach chi while the ends
   ramp as 2, 4, 8, ...; on the 2n-site chain nearly every bond is at chi, and
   the two errors happen to cancel to ~12%. Set `ancillas=False` for the plain
   count.) Allow a factor ~2 for the two-site tensor and SVD workspace during
   `apply`, so a node with M bytes is exhausted when mem > M/2.

   TIME. TEBD applies n*k two-site gates, each an SVD of a (d*chi) x (d*chi)
   matrix, so

       flops ~ N_traj * n * k * d^3 * chi^3

   N_traj is set by the statistical error you need on an observable, not by the
   bond dimension; 1e3 is the figure the earlier trajectory study measured for
   SEM(<Z>) = 0.01. Note this is N * <chi^3>, not N * <chi>^3 -- the ensemble
   spread in chi is wide enough that Jensen's inequality costs a factor ~3, and
   the maps carry the chi3_mean column for exactly that reason.

CAVEAT worth stating out loud in any write-up. This bounds ONE family of
classical methods: trajectory-unravelled MPS. It says nothing about a cleverer
contraction order, a different ansatz (PEPS, neural, Pauli-path / sparse Pauli
dynamics), or a method that exploits the specific gate structure. Quote it as
"beyond reach for matrix-product methods at bond dimension chi", never as
"classically impossible".
"""
import math, glob as _glob, argparse, sys
from dataclasses import dataclass, field
import pandas as pd

CEIL_FRAC = 0.90     # discard points at or above this fraction of 2^(n/2)
DTYPE     = 16       # bytes per complex128
D         = 2        # physical dimension of one qubit in the trajectory MPS

MACHINES = [("64 GB node", 64e9), ("256 GB node", 256e9),
            ("2 TB fat node", 2e12), ("100 TB distributed", 100e12)]


def discover(pattern="../results/ckt_map_n*/map_n*_m*.csv"):
    """{n: DataFrame} for every map CSV matching the glob (n from the CSV's own column)."""
    out = {}
    for path in sorted(_glob.glob(pattern)):
        df = pd.read_csv(path)
        if "n" not in df.columns or df.empty:
            print(f"  ! skipping {path}: no `n` column or empty", file=sys.stderr)
            continue
        n = int(df["n"].iloc[0])
        if n in out:
            print(f"  ! {path} duplicates n={n}; keeping the first", file=sys.stderr)
            continue
        df.attrs["path"] = path
        out[n] = df
    if not out:
        print(f"  ! nothing matched {pattern!r}", file=sys.stderr)
    return out


@dataclass
class Fit:
    beta: float                  # d ln(chi) / dn
    a: float                     # intercept
    slope_se: float
    resid_sd: float              # in nats
    theta: float
    p: float
    k: int
    ntraj: int
    ancillas: bool
    points: pd.DataFrame = field(repr=False)   # what went into the fit

    @property
    def ceiling_slope(self):
        return math.log(2) / 2

    @property
    def frac_of_ceiling(self):
        return self.beta / self.ceiling_slope

    @property
    def depth_cap(self):
        return 4.0 ** self.k

    def chi(self, n):
        """Fitted chi at size n, capped by the Hilbert ceiling and the depth cap."""
        return min(math.exp(self.beta * n + self.a), 2.0 ** (n / 2), self.depth_cap)

    def memory(self, n):
        """Bytes for one trajectory MPS (see the module docstring)."""
        L = 2 * n if self.ancillas else n
        return L * D * self.chi(n) ** 2 * DTYPE

    def flops(self, n):
        return self.ntraj * n * self.k * D ** 3 * self.chi(n) ** 3


def fit(maps, theta=0.95, p=0.05, k=None, ntraj=1000, ancillas=True):
    """Fit ln(chi) = beta*n + a at one (theta, p) cell across system sizes.

    `maps` is {n: DataFrame} from discover(), or {n: path}, or a list of paths.
    """
    if isinstance(maps, (list, tuple)):
        maps = {int(pd.read_csv(q)["n"].iloc[0]): pd.read_csv(q) for q in maps}
    maps = {n: (pd.read_csv(v) if isinstance(v, str) else v) for n, v in maps.items()}

    rec = []
    for n in sorted(maps):
        df = maps[n]
        sel = df[(df["theta"].round(4) == round(theta, 4)) &
                 (df["p"].round(4) == round(p, 4))]
        if sel.empty:
            print(f"  ! n={n}: no cell at theta={theta}, p={p}", file=sys.stderr)
            continue
        r = sel.iloc[0]
        if str(r.get("censored", "false")).lower() == "true":
            print(f"  ! n={n}: censored, chi is a LOWER bound -- dropped", file=sys.stderr)
            continue
        ceil = 2 ** (n // 2)
        rec.append(dict(n=n, chi=float(r["chi_mean"]), sem=float(r.get("chi_sem", 0.0)),
                        ceiling=ceil, frac=float(r["chi_mean"]) / ceil,
                        pinned=float(r["chi_mean"]) >= CEIL_FRAC * ceil,
                        k=int(r["k"])))
    pts = pd.DataFrame(rec)
    if pts.empty:
        raise ValueError(f"no data at theta={theta}, p={p}")
    used = pts[~pts["pinned"]].copy()
    if len(used) < 2:
        raise ValueError(
            f"only {len(used)} unpinned point(s) at theta={theta}, p={p} -- cannot fit.\n"
            "Every other cell is at the finite-size ceiling 2^(n/2). Use a larger p, "
            "or add a larger n.")

    used["ln_chi"] = used["chi"].apply(math.log)
    mx, my = used["n"].mean(), used["ln_chi"].mean()
    sxx = ((used["n"] - mx) ** 2).sum()
    beta = ((used["n"] - mx) * (used["ln_chi"] - my)).sum() / sxx
    a = my - beta * mx
    res = used["ln_chi"] - (beta * used["n"] + a)
    m = len(used)
    sd = math.sqrt((res ** 2).sum() / (m - 2)) if m > 2 else float("nan")
    se = sd / math.sqrt(sxx) if m > 2 else float("nan")
    return Fit(beta=beta, a=a, slope_se=se, resid_sd=sd, theta=theta, p=p,
               k=k or int(pts["k"].iloc[0]), ntraj=ntraj, ancillas=ancillas, points=pts)


def resource_table(f, ns=range(16, 49, 2)):
    rows = []
    for n in ns:
        c = f.chi(n)
        limit = ("depth cap 4^k" if c >= f.depth_cap * 0.999 else
                 "Hilbert ceiling" if c >= 2.0 ** (n / 2) * 0.999 else "")
        rows.append(dict(n=n, chi=c, mem_GB=f.memory(n) / 1e9,
                         flops=f.flops(n), binding=limit))
    return pd.DataFrame(rows)


def walls(f, machines=MACHINES):
    """System size at which ONE trajectory stops fitting (half the node usable)."""
    rows = []
    for name, M in machines:
        lo, hi = 8.0, 64.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f.memory(mid) < M / 2 else (lo, mid)
        rows.append(dict(machine=name, bytes=M, n=lo, chi=f.chi(lo)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="CSV paths, or a quoted glob")
    ap.add_argument("--theta", type=float, default=0.95)
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=None, help="Trotter steps; sets the depth cap 4^k")
    ap.add_argument("--ntraj", type=int, default=1000)
    ap.add_argument("--no-ancillas", action="store_true",
                    help="count a plain n-site MPS instead of the 2n-site ancilla chain")
    args = ap.parse_args()

    paths = [q for pat in args.csvs for q in (_glob.glob(pat) or [pat])]
    maps = {}
    for q in sorted(paths):
        df = pd.read_csv(q)
        maps[int(df["n"].iloc[0])] = df
    f = fit(maps, theta=args.theta, p=args.p, k=args.k, ntraj=args.ntraj,
            ancillas=not args.no_ancillas)

    print(f"cell: theta = {f.theta}, p = {f.p}, k = {f.k}\n")
    print(f.points.to_string(index=False,
          formatters={"chi": "{:.2f}".format, "sem": "{:.2f}".format,
                      "frac": "{:.3f}".format}))
    print(f"\nln chi = {f.beta:.4f} n + {f.a:.4f}        "
          f"({(~f.points['pinned']).sum()} unpinned points)")
    if not math.isnan(f.resid_sd):
        print(f"   slope std err {f.slope_se:.4f}, residual sd {f.resid_sd:.4f} nats "
              f"= {(math.exp(f.resid_sd)-1)*100:.1f}%")
    print(f"   ceiling slope ln2/2 = {f.ceiling_slope:.4f}; "
          f"this is {f.frac_of_ceiling*100:.0f}% of maximal")
    print(f"\ndepth cap chi <= 4^k = {f.depth_cap:.3g}  "
          f"(one rank-4 two-site gate crosses each cut per Trotter step)")
    print(f"memory counts the {'2n-site ancilla chain' if f.ancillas else 'plain n-site MPS'}\n")

    t = resource_table(f)
    print(t.to_string(index=False, formatters={
        "chi": "{:11.3g}".format, "mem_GB": "{:10.1f}".format, "flops": "{:11.2e}".format}))
    print("\nsystem size at which one trajectory no longer fits (half the node usable):")
    for _, r in walls(f).iterrows():
        print(f"   {r['machine']:20s}  n = {r['n']:5.1f}   (chi = {r['chi']:.3g})")


if __name__ == "__main__":
    main()
