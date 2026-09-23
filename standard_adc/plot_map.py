#!/usr/bin/env python3
"""
plot_map.py -- colormaps from MODE=map output.

    python3 plot_map.py results/map_n16_m16.csv

Produces map_<tag>.png with three panels sharing the (theta, p) axes:

    chi_traj        the classical cost, log colour scale
    S_traj          entanglement, for comparison
    chi / 2^S       the Schmidt-TAIL factor

The third is the one worth having next to the other two. A value near 1 means a
flat Schmidt spectrum, which is what stabilizer and free-fermion structure
produce -- and at theta = pi/2 the ZZ rotation becomes RZZ(pi) = -i ZZ, a Pauli,
leaving the XX+YY part alone. That is a free-fermion circuit, classically
simulable in O(n^3) however entangled it looks. The cost panel alone cannot see
that; this one can.

Secondary axes give the physics units implied by JREF/GREF:
    dt = theta/(2 J),  t = k dt,  total damping = 1-(1-p)^k
"""
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

JREF = 0.25

if len(sys.argv) < 2:
    sys.exit("usage: plot_map.py <map_*.csv> [JREF]")
path = sys.argv[1]
if len(sys.argv) > 2:
    JREF = float(sys.argv[2])

d = pd.read_csv(path)
n, k = int(d.n.iloc[0]), int(d.k.iloc[0])
th = np.sort(d.theta.unique())
pp = np.sort(d.p.unique())

def grid(col):
    g = np.full((len(pp), len(th)), np.nan)
    for _, r in d.iterrows():
        g[np.searchsorted(pp, r.p), np.searchsorted(th, r.theta)] = r[col]
    return g

CHI, S = grid("chi_mean"), grid("S_traj_mean")
d["_cen"] = d.censored.astype(str).str.lower().isin(["true", "1"])
CEN = grid("_cen") > 0.5
TAIL = CHI / np.power(2.0, S)

plt.rcParams.update({"figure.dpi": 180, "font.size": 9, "font.family": "serif",
                     "mathtext.fontset": "dejavuserif", "savefig.bbox": "tight"})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ext = [th[0], th[-1], pp[0], pp[-1]]
kw = dict(origin="lower", aspect="auto", extent=ext, interpolation="nearest")

im0 = ax[0].imshow(CHI, norm=LogNorm(), cmap="magma", **kw)
plt.colorbar(im0, ax=ax[0], label=r"$\chi_{\rm traj}$")
ax[0].set_title(r"(a) classical cost  $\chi_{\rm traj}$")

im1 = ax[1].imshow(S, cmap="viridis", **kw)
plt.colorbar(im1, ax=ax[1], label=r"$\langle S_{\rm traj}\rangle$ [bits]")
ax[1].set_title(r"(b) entanglement  $S_{\rm traj}$")

im2 = ax[2].imshow(TAIL, norm=LogNorm(), cmap="cividis", **kw)
plt.colorbar(im2, ax=ax[2], label=r"$\chi\,/\,2^{S}$")
ax[2].set_title(r"(c) Schmidt tail  $\chi/2^{S}$" "\n" r"$\approx 1$: free-fermion / stabilizer")

for a in ax:
    a.set_xlabel(r"$\theta$  (rxx angle)")
    if th[0] < np.pi/2 < th[-1]:
        a.axvline(np.pi/2, color="w", ls=":", lw=1.4)
    # mark censored points so a clipped cell is never read as a measurement
    if CEN.any():
        yy, xx = np.where(CEN)
        a.plot(th[xx], pp[yy], "x", color="red", ms=5, mew=1.2)

sec = ax[0].secondary_xaxis("top", functions=(lambda x: x/(2*JREF), lambda x: 2*JREF*x))
sec.set_xlabel(r"$\delta t = \theta/2J$")
# Total damping over the circuit as a second label on the p axis. A secondary
# axis on panel (c) collided with its colourbar, so it goes in the ylabel.
ax[0].set_ylabel(r"$p$ per step" "\n" r"(total $1-(1-p)^k$: "
                 + f"{1-(1-pp[0])**k:.2f}" + r"$\rightarrow$" + f"{1-(1-pp[-1])**k:.2f})")

fig.suptitle(f"$n={n}$, $k={k}$ Trotter steps   (red x = reported $\\chi$ within 10% of the cap)",
             y=1.04)
out = path.rsplit(".", 1)[0] + ".png"
plt.savefig(out)
print("wrote", out)
