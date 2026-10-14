#!/usr/bin/env python3
"""
symmetrize_map.py -- build the full 0..2pi map from a sweep over [0, pi/2]-ish,
using the exact symmetries of the gate instead of more cluster time.

Works two ways.

  From a notebook:
      from symmetrize_map import symmetrize, discover
      maps = discover('../results/ckt_map_n*/map_n*_m*.csv')   # {8: df, 12: df, ...}
      full = {n: symmetrize(df) for n, df in maps.items()}
      full[16].plot...                       # or full[16].to_csv(...)

  From a shell:
      python3 symmetrize_map.py ../results/ckt_map_n16/map_n16_m16.csv
      python3 symmetrize_map.py '../results/ckt_map_n*/map_n*_m*.csv' --suffix _full

-----------------------------------------------------------------------------
THE SYMMETRY. The circuit gate is

    U(theta) = exp(-i theta A),     A = (XX + YY)/2 + ZZ

and the eigenvalues of A are {-2, 0, 1, 1}: ALL INTEGERS. Therefore

    exp(-2 pi i A) = identity          =>  U(2pi - theta) = conj(U(theta))
    exp(-i pi A)   = diag(-1,-1,1,1)
                   = -Z (x) Z         =>  U(pi - theta)  = (-Z(x)Z) conj(U(theta))

Both reflections are therefore trivial for entanglement:

  * theta -> 2pi - theta is EXACT. A is real, so exp(+i theta A) is the
    elementwise conjugate of U(theta); the amplitude-damping Kraus operators
    K0 = diag(1, sqrt(1-p)) and K1 = sqrt(p) sigma^- are real too, and Born
    probabilities |.|^2 are conjugation invariant. The Haar measure on single-
    qubit initial states is conjugation invariant as well. So the whole
    trajectory ensemble maps to itself and chi_bar(2pi - theta) = chi_bar(theta)
    identically. Nothing is approximated.

  * theta -> pi - theta is exact UP TO BOUNDARY TERMS. Per brickwall sublayer
    the -Z(x)Z factors multiply to the global parity operator prod_j Z_j, which
    commutes with every gate in the circuit (XX+YY and ZZ both conserve parity),
    leaving only two uncancelled Z's at the open ends of the chain. Measured at
    n=8, p=0.05 the two sides agree to ~2%, inside the trajectory SEM, and the
    edge effect shrinks as 1/n.

The fundamental domain is thus theta in [0, pi/2]. Everything else is a copy,
which is why a sweep over [0.2, 2.4] already determines the whole circle, and
why the unmeasured window [2.4, pi] is just the mirror of [0, 0.742].

EXACT POINTS inserted automatically (chi = 1, S = 0, for ANY product initial
state and ANY p), because they are measure-zero and a grid will miss them:

    theta = 0, pi, 2pi    U is the identity or -Z(x)Z, i.e. a product of
                          single-qubit gates: zero entangling power. theta = pi
                          is trivial for exactly the same reason theta = 0 is --
                          they are the same point of the fundamental domain.

NOT inserted, because it is only trivial for a computational-basis input:

    theta = pi/2, 3pi/2   U is a permutation of basis states times phases.
                          chi = 1 for a basis input at any p; for a generic
                          (Haar) product input it is merely cheap, and the value
                          depends on n, so it has to be measured.

Every output row carries a `source` column: measured | mirror_pi | mirror_2pi |
exact. Plot with the real theta values (pcolormesh with explicit edges), NOT
imshow on an index grid -- an uneven theta grid rendered as equal-width cells is
what makes the pi/2 notch look broad and hides the gap.
"""
import math, glob as _glob, argparse, sys
import pandas as pd

TWO_PI  = 2 * math.pi
DECIMAL = 4      # grid points closer than ~1e-4 rad are the same physical point
EXACT_THETAS = (0.0, math.pi, TWO_PI)


def discover(pattern="../results/ckt_map_n*/map_n*_m*.csv"):
    """{n: DataFrame} for every map CSV matching the glob.

    n is read from the CSV's own `n` column, not parsed out of the filename, so
    a renamed file still lands under the right key.
    """
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


def _retime(df):
    """dt and t follow from theta and k; recompute them on mirrored rows."""
    df = df.copy()
    df["dt"] = df["theta"] / 2
    df["t"] = df["k"] * df["dt"]
    return df


def _exact_frame(template, plist):
    rows = []
    k = int(template["k"])
    for th in EXACT_THETAS:
        for p in plist:
            r = dict(template)
            r.update(theta=th, p=p, dt=th / 2, t=k * th / 2,
                     total_damping=1 - (1 - p) ** k,
                     S_traj_mean=0.0, S_traj_p95=0.0,
                     chi_mean=1.0, chi_sem=0.0, chi_max=1, chi3_mean=1.0,
                     censored=False, saturated=False, source="exact")
            rows.append(r)
    return pd.DataFrame(rows)


def symmetrize(df, mirror_pi=True, add_exact=True):
    """Extend a map DataFrame to theta in [0, 2pi] using the gate's symmetries.

    Returns a new DataFrame with an added `source` column. Measured rows always
    win over mirrored ones, and mirrored over exact, so nothing real is
    overwritten.
    """
    df = df.copy()
    df["source"] = "measured"
    plist = sorted(df["p"].unique())
    out = [df]

    def pending():
        return pd.concat(out, ignore_index=True)

    # pi - theta first: it fills the interior window. Then 2pi - theta over
    # everything, which reflects both the measured and the newly mirrored rows.
    if mirror_pi:
        m = _retime(pending().assign(theta=lambda d: math.pi - d["theta"],
                                     source="mirror_pi"))
        out.append(m[(m["theta"] >= -1e-9) & (m["theta"] <= TWO_PI + 1e-9)])
    m = _retime(pending().assign(theta=lambda d: TWO_PI - d["theta"],
                                 source="mirror_2pi"))
    out.append(m[(m["theta"] >= -1e-9) & (m["theta"] <= TWO_PI + 1e-9)])

    if add_exact:
        out.append(_exact_frame(df.iloc[0].to_dict(), plist))

    full = pd.concat(out, ignore_index=True)
    rank = {"measured": 0, "mirror_pi": 1, "mirror_2pi": 2, "exact": 3}
    full["_r"] = full["source"].map(rank)
    full["_th"] = full["theta"].round(DECIMAL)
    full = (full.sort_values(["_th", "p", "_r"])
                .drop_duplicates(["_th", "p"], keep="first")
                .drop(columns=["_r", "_th"])
                .sort_values(["theta", "p"])
                .reset_index(drop=True))
    return full


def report(df, full, label=""):
    th_in, th_out = df["theta"].nunique(), full["theta"].nunique()
    counts = full["source"].value_counts().to_dict()
    ths = sorted(full["theta"].unique())
    gaps = [(a, b) for a, b in zip(ths, ths[1:]) if b - a > 0.25]
    print(f"{label}{len(df)} rows / {th_in} theta  ->  {len(full)} rows / {th_out} theta "
          f"over [0, 2pi]   {counts}")
    print("    gaps > 0.25 rad: " +
          (", ".join(f"({a:.3f}, {b:.3f})" for a, b in gaps) if gaps else "none"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a CSV path, or a quoted glob matching several")
    ap.add_argument("-o", "--out", default=None, help="output path (single input only)")
    ap.add_argument("--suffix", default="_full",
                    help="appended to each stem when -o is not given")
    ap.add_argument("--no-mirror-pi", action="store_true",
                    help="use only the exact 2pi-theta mirror")
    args = ap.parse_args()

    paths = sorted(_glob.glob(args.path)) or [args.path]
    if args.out and len(paths) > 1:
        sys.exit("-o takes a single input; use --suffix for a glob")
    for path in paths:
        df = pd.read_csv(path)
        full = symmetrize(df, mirror_pi=not args.no_mirror_pi)
        dest = args.out or path.replace(".csv", f"{args.suffix}.csv")
        full.to_csv(dest, index=False)
        report(df, full, label=f"{path}\n    ")
        print(f"    wrote {dest}\n")
    print("Plot with pcolormesh on the real theta values, not imshow on an index grid.")


if __name__ == "__main__":
    main()
