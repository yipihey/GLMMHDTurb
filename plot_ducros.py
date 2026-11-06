"""Plot time-averaged spectra written by test_ducros.jl (ducros_out/spec_N*_*.csv).

    python3 plot_ducros.py ducros_out/ N [out.png]

Top: compensated kinetic (sqrt(rho) v) and magnetic spectra, k^2 E(k) (Burgers-like scaling).
Bottom: ratio of each config's E_kin(k) to the HLL baseline (>1 at high k = less dissipation).
"""
import csv, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1] if len(sys.argv) > 1 else "ducros_out"
N = sys.argv[2] if len(sys.argv) > 2 else "32"
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(d, f"spectra_N{N}.png")

runs = {}
for f in sorted(glob.glob(os.path.join(d, f"spec_N{N}_*.csv"))):
    name = os.path.basename(f)[len(f"spec_N{N}_"):-4]
    a = np.loadtxt(f, delimiter=",", skiprows=1)
    runs[name] = a

fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
base = runs.get("hll")
for name, a in runs.items():
    k = a[1:, 0]; Ek = a[1:, 1]; Eb = a[1:, 3]
    ax[0].loglog(k, k**2 * Ek, label=name)
    ax[1].loglog(k, k**2 * Eb, label=name)
    if base is not None:
        ax[2].semilogx(k, Ek / base[1:, 1], label=name)
ax[0].set_ylabel(r"$k^2 E_{kin}(k)$  [$\sqrt{\rho}v$]")
ax[1].set_ylabel(r"$k^2 E_{mag}(k)$")
ax[2].set_ylabel(r"$E_{kin}/E_{kin}^{HLL}$")
ax[2].axhline(1, color="k", lw=0.5)
ax[2].set_xlabel("k  [2π/L]")
ax[0].legend(fontsize=8)
for a_ in ax: a_.grid(alpha=0.3, which="both")
fig.suptitle(f"Driven GLM-MHD turbulence, N={N}^3, time-averaged")
fig.tight_layout()
fig.savefig(out, dpi=130)
print(out)
