#!/usr/bin/env python3
# Regenerates all figures from the CSVs. Usage: DATA=<dir> python3 make_figures.py
import csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
# Reads CSVs from the working directory by default; override with DATA=<dir>.
import os, sys
U=os.environ.get("DATA", "./")
def P(name):
    for c in (U+name, U+"1787307686875_"+name, U+"1787307686876_"+name, U+"1787307686877_"+name):
        if os.path.exists(c): return c
    sys.exit(f"missing: {name} (looked in {U})")
plt.rcParams.update({"figure.dpi":150,"font.size":9,"axes.grid":True,
                     "grid.alpha":.3,"axes.axisbelow":True,"legend.framealpha":.9})
CM, CN, CE = "#c0392b", "#1f6fb4", "#444444"

# ---- data ----
co=defaultdict(dict); er=defaultdict(dict)
for r in csv.DictReader(open(P("cmp_coefficients.csv"))):
    co[r["route"]][int(r["maxdim"])]=(float(r["c1"]),float(r["dc_max"]))
for r in csv.DictReader(open(P("cmp_errors.csv"))):
    er[r["route"]][int(r["maxdim"])]=dict(E_mpf=float(r["E_mpf"]),E_k3=float(r["E_k3"]),
        E_k8=float(r["E_k8"]),lam=float(r["lambda_min"]))
we=defaultdict(dict)
for r in csv.DictReader(open(P("cmp_weight_vs_error.csv"))):
    we[r["object"]][int(r["maxdim"])]=(float(r["discarded_weight"]),float(r["dc_max"]))
sp=defaultdict(list)
for r in csv.DictReader(open(P("cmp_spectra.csv"))):
    sp[r["object"]].append(float(r["sigma"]))
n5=[]
import os as _os
if not _os.path.exists(P.__wrapped__ if False else (U+"sweep_v2.csv")):
    n5=None
if n5 is not None:
    for r in csv.DictReader(open(P("sweep_v2.csv"))):
        c=[float(x) for x in r["coeffs"].split(";")]
        n5.append((int(r["maxdim"]),c[0],float(r["E_mpf"])))
    n5.sort()
else:
    n5=[]

CX,EK3X,EK8X,EMPFX = -0.144708, 1.334740e-02, 2.502581e-04, 3.755694e-05
# post-fix N-route (directsum delta, maxdim_G = maxdim)
V2=[(16,-0.344108,1.994e-1,843.77,4102.40,1.2231e-3),(32,-0.131177,1.353e-2,23.56,93.92,2.0191e-4),
    (48,-0.132625,1.208e-2,38.13,-10.72,-2.9772e-5),(64,-0.147730,3.022e-3,21.60,5.36,-5.3314e-6),
    (96,-0.151453,6.745e-3,9.69,3.09,3.5430e-6),(128,-0.126569,1.814e-2,12.68,2.35,5.1968e-5),
    (192,-0.145195,4.865e-4,0.24,2.82,3.2094e-5),(256,-0.144810,1.017e-4,-0.00,-0.58,2.6821e-5)]
md=sorted(co["M"]); v2md=[v[0] for v in V2]

def sgn(ax,x,y,c,lab,mk="o"):
    """plot |y| on log axis; filled = positive, hollow = NEGATIVE."""
    x=np.array(x); y=np.array(y)
    ax.plot(x,np.abs(y),"-",color=c,lw=1.2,zorder=2,label=lab)
    p,n=y>0,y<=0
    ax.plot(x[p],np.abs(y[p]),mk,color=c,ms=6,zorder=3)
    ax.plot(x[n],np.abs(y[n]),mk,mfc="white",mec=c,mew=1.8,ms=8,zorder=3)

# ===== FIG 1 : slide 26, before and after ==========================
# Panels (a),(b) need the n=5 M-route sweep (sweep_v2.csv). Skipped if absent.
f,a=plt.subplots(1,3,figsize=(13,3.6))
a[0].plot([v[0] for v in n5],[v[1] for v in n5],"o-",color=CM,label="$c_1$ ($k$=3)")
a[0].plot([v[0] for v in n5],[1-v[1] for v in n5],"o-",color="#e08a2e",label="$c_2$ ($k$=8)")
a[0].set(xlabel="maxdim",ylabel="coefficient",title="(a) slide 26 as presented\n$M$-route, $n$=5")
a[0].legend(fontsize=8)
sgn(a[1],[v[0] for v in n5],[v[2] for v in n5],CM,"$E_{mpf}$ ($M$-route)")
a[1].set(xlabel="maxdim",ylabel="$|E_{mpf}|$",yscale="log",
         title="(b) same data, sign exposed\nhollow = NEGATIVE")
a[1].legend(fontsize=8)
a[2].axhline(CX,color=CE,ls="--",lw=1.2,label="exact")
a[2].plot(md,[co["M"][m][0] for m in md],"s-",color=CM,label="$M$-route")
a[2].plot(v2md,[v[1] for v in V2],"o-",color=CN,label="$N$-route")
a[2].set(xlabel="maxdim",ylabel="$c_1$",ylim=(-0.7,0.7),
         title="(c) $n$=4 vs EXACT ($\\chi$=256)\n$M$-route $\\chi$=96 is off-scale ($-10.8$)")
a[2].legend(fontsize=8)
f.tight_layout(); f.savefig("fig1_slide26.png",bbox_inches="tight"); plt.close(f)

# ===== FIG 2 : the money plot ======================================
f,a=plt.subplots(1,2,figsize=(9.5,3.8))
a[0].loglog(md,[co["M"][m][1] for m in md],"s-",color=CM,label="$M$-route ($M,\\vec L,P$)")
a[0].loglog(v2md,[v[2] for v in V2],"o-",color=CN,label="$N$-route (Gram of errors)")
a[0].axhline(1e-4,color=CE,ls=":",lw=1,label="$N$-route floor")
a[0].set(xlabel="maxdim",ylabel=r"$\max|c-c_{\rm exact}|$",
         title="(a) coefficient error vs bond dimension\n$n$=4, $\\gamma$=0.05, $t$=3, order 2")
a[0].legend(fontsize=8)
sgn(a[1],md,[er["M"][m]["E_k8"] for m in md],CM,"$M$-route","s")
sgn(a[1],v2md,[EK8X*(1+v[4]/100) for v in V2],CN,"$N$-route","o")
a[1].axhline(EK8X,color=CE,ls="--",lw=1.2,label="exact $E_{k_8}$")
a[1].set(xlabel="maxdim",ylabel="$|E_{k_8}|$",yscale="log",xscale="log",
         title="(b) single-Trotter error $E_{k_8}=\\|\\rho-\\rho_{k_8}\\|^2$\nhollow = NEGATIVE (impossible)")
a[1].legend(fontsize=8)
f.tight_layout(); f.savefig("fig2_money.png",bbox_inches="tight"); plt.close(f)

# ===== FIG 3 : PSD ==================================================
f,a=plt.subplots(1,2,figsize=(9.5,3.8))
sgn(a[0],md,[er["M"][m]["lam"] for m in md],CM,"$M$-route","s")
sgn(a[0],v2md,[v[5] for v in V2],CN,"$N$-route","o")
a[0].axhline(2.811e-5,color=CE,ls="--",lw=1.2,label="exact $\\lambda_{\\min}$")
a[0].set(xlabel="maxdim",ylabel="$|\\lambda_{\\min}(N)|$",yscale="log",xscale="log",
         title="(a) smallest eigenvalue of $N$\nhollow = NEGATIVE ($N$ must be PSD)")
a[0].legend(fontsize=8)
sgn(a[1],md,[er["M"][m]["E_mpf"] for m in md],CM,"$M$-route","s")
sgn(a[1],md,[er["N"][m]["E_mpf"] for m in md],CN,"$N$-route","o")
a[1].axhline(EMPFX,color=CE,ls="--",lw=1.2,label="exact $E_{mpf}$")
a[1].set(xlabel="maxdim",ylabel="$|E_{mpf}|$",yscale="log",xscale="log",
         title="(b) $E_{mpf}$ — UNRESOLVED in both routes\n$N$ near rank-1: $c$ needs the eigenvector,\n$E_{mpf}$ needs the eigenvalue")
a[1].legend(fontsize=8)
f.tight_layout(); f.savefig("fig3_psd.png",bbox_inches="tight"); plt.close(f)

# ===== FIG 4 : discarded weight vs error ============================
f,a=plt.subplots(1,2,figsize=(9.8,3.8))
wm=[we["F"][m] for m in md if we["F"][m][0]>0]; wn=[we["Phi"][m] for m in md if we["Phi"][m][0]>0]
a[0].loglog([w[0] for w in wm],[w[1] for w in wm],"s",color=CM,ms=8,label="$M$-route (truncating $\\mathbb{F}$)")
a[0].loglog([w[0] for w in wn],[w[1] for w in wn],"o",color=CN,ms=8,label="$N$-route (truncating $\\Phi$)")
for w,m in zip(wm,[m for m in md if we["F"][m][0]>0]): a[0].annotate(m,(w[0],w[1]),fontsize=6,xytext=(3,3),textcoords="offset points")
for w,m in zip(wn,[m for m in md if we["Phi"][m][0]>0]): a[0].annotate(m,(w[0],w[1]),fontsize=6,xytext=(3,3),textcoords="offset points")
xs=np.logspace(-6,0,10)
a[0].loglog(xs,xs,color=CE,ls=":",lw=1,label="slope 1 (no amplification)")
a[0].set(xlabel="fraction of Schmidt weight discarded",ylabel=r"$\max|c-c_{\rm exact}|$",
         title="(a) THE amplification plot\nupper-left = bad, lower-right = good")
a[0].legend(fontsize=7,loc="lower right")
amp=[we["F"][m][1]/we["F"][m][0] for m in md if we["F"][m][0]>0]
ampn=[we["Phi"][m][1]/we["Phi"][m][0] for m in md if we["Phi"][m][0]>0]
a[1].semilogy([m for m in md if we["F"][m][0]>0],amp,"s-",color=CM,label="$M$-route")
a[1].semilogy([m for m in md if we["Phi"][m][0]>0],ampn,"o-",color=CN,label="$N$-route")
a[1].axhline(1,color=CE,ls=":",lw=1)
a[1].text(20,1.4,"amplification = 1",fontsize=7,color=CE)
a[1].set(xlabel="maxdim",ylabel="error in $c$ per unit weight discarded",
         title="(b) amplification factor\ngeometric mean: $M$=548, $N$=0.182 (3018$\\times$)")
a[1].legend(fontsize=8)
f.tight_layout(); f.savefig("fig4_amplification.png",bbox_inches="tight"); plt.close(f)

# ===== FIG 5 : spectra =============================================
f,a=plt.subplots(1,3,figsize=(13,3.6))
for k,c,l in (("F","#c0392b","$\\mathbb{F}=\\mathbb{S}^\\dagger\\mathbb{S}$"),("G","#e08a2e","$\\mathbb{G}=\\mathbb{E}^\\dagger\\mathbb{E}$"),
              ("Xi","#2e9e5b","$\\Xi=\\Delta^\\dagger\\mathbb{E}$"),("Phi","#1f6fb4","$\\Phi=\\Delta^\\dagger\\Delta$")):
    s=np.array(sp[k]); a[0].semilogy(np.arange(1,len(s)+1),s,color=c,lw=1.3,label=l)
a[0].set(xlabel="singular value index",ylabel="$\\sigma_i$ (absolute)",
         title="(a) slide 23, absolute scale\n$\\Phi$ sits 3 orders below $\\mathbb{F}$")
a[0].legend(fontsize=8)
for k,c,l in (("F","#c0392b","$\\mathbb{F}$"),("G","#e08a2e","$\\mathbb{G}$"),
              ("Xi","#2e9e5b","$\\Xi$"),("Phi","#1f6fb4","$\\Phi$")):
    s=np.array(sp[k]); a[1].semilogy(np.arange(1,len(s)+1),s/s[0],color=c,lw=1.3,label=l)
a[1].set(xlabel="singular value index",ylabel="$\\sigma_i/\\sigma_1$",
         title="(b) normalised — MISLEADING\n$\\mathbb{F}$'s $\\sigma_1$ is $\\mathbb{G}$, which cancels")
a[1].legend(fontsize=8)
F,G=np.array(sp["F"]),np.array(sp["G"])
a[2].semilogy(np.arange(1,len(F)+1),F,color="#c0392b",lw=2.5,alpha=.55,label="$\\mathbb{F}$")
a[2].semilogy(np.arange(1,len(G)+1),G,color="#e08a2e",lw=1,ls="--",label="$\\mathbb{G}$")
a[2].set(xlabel="singular value index",ylabel="$\\sigma_i$",
         title="(c) $\\mathbb{F}$ and $\\mathbb{G}$ are the SAME object\n$\\sum(\\sigma^F-\\sigma^G)^2/\\|\\mathbb{F}\\|^2=2.2\\times10^{-5}$")
a[2].legend(fontsize=8)
f.tight_layout(); f.savefig("fig5_spectra.png",bbox_inches="tight"); plt.close(f)
print("ok")
