"""Plot time-averaged spectra written by test_ducros.jl (ducros_out/spec_N*_*.csv).

    python3 plot_ducros.py ducros_out/ N [out.png]
    python3 plot_ducros.py --snap "ducros_out/gpu_n128_r1e-3_k0_s*" 128 [out.png] [tmin] [tmax]   # matched-t, seeds

Top: compensated kinetic (sqrt(rho) v) and magnetic spectra, k^2 E(k) (Burgers-like scaling).
Bottom: ratio of each config's E_kin(k) to the HLL baseline (>1 at high k = less dissipation).
"""
import csv, glob, os, re, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def snap_mode(dirs_glob, N, out, tmin, tmax, kshow=(4, 8, 16, 32)):
    """Matched-t comparison from per-snapshot spectra (snap_N*_cfg_t*.csv) over seed dirs.
    Averages over seeds at each nominal t, then over t in [tmin,tmax]; ratios vs hll."""
    dirs = sorted(glob.glob(dirs_glob))
    spec = defaultdict(lambda: defaultdict(list))   # cfg -> t -> [array per seed]
    ffl = defaultdict(list)
    pat = re.compile(rf"snap_N{N}_(.+)_t([0-9.]+)\.csv$")
    for dd in dirs:
        for f in glob.glob(os.path.join(dd, f"snap_N{N}_*_t*.csv")):
            m = pat.search(os.path.basename(f))
            if not m: continue
            cfg, t = m.group(1), float(m.group(2))
            hdr = open(f).readline()
            fm = re.search(r"floorfrac=([0-9.eE+-]+)", hdr)
            if fm: ffl[cfg].append(float(fm.group(1)))
            spec[cfg][t].append(np.loadtxt(f, delimiter=",", skiprows=1))
    if "hll" not in spec:
        sys.exit("no hll snapshots found")
    # only t present in every cfg with the same number of seeds as hll (matched realizations)
    ts = sorted(t for t in spec["hll"] if tmin <= t <= tmax and
                all(len(spec[c].get(t, [])) == len(spec["hll"][t]) for c in spec))
    print(f"dirs={dirs}\nmatched t: {ts}")
    mean_t = {c: np.mean([np.mean(spec[c][t], axis=0) for t in ts], axis=0) for c in spec}
    # per-seed ratio scatter at fixed t -> error bar on the seed+time mean of the ratio
    def ratio_stats(c, col):
        rs = [np.array(spec[c][t][i])[:, col] / np.array(spec["hll"][t][i])[:, col]
              for t in ts for i in range(len(spec["hll"][t]))]
        rs = np.array(rs); return rs.mean(0), rs.std(0) / np.sqrt(max(len(rs) - 1, 1))
    fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    for c, a in sorted(mean_t.items()):
        k = a[1:, 0]
        ax[0].loglog(k, k**2 * a[1:, 1], label=c)
        ax[1].loglog(k, k**2 * a[1:, 3], label=c)
        for col, ls in ((1, "-"), (3, "--")):
            m, e = ratio_stats(c, col)
            ax[2].semilogx(k, m[1:], ls, label=f"{c} {'kin' if col == 1 else 'mag'}")
            ax[2].fill_between(k, (m - e)[1:], (m + e)[1:], alpha=0.2)
    ax[0].set_ylabel(r"$k^2 E_{kin}(k)$"); ax[1].set_ylabel(r"$k^2 E_{mag}(k)$")
    ax[2].set_ylabel("ratio to HLL (solid kin, dashed mag)"); ax[2].axhline(1, color="k", lw=0.5)
    ax[2].set_xlabel("k  [2π/L]"); ax[0].legend(fontsize=8); ax[2].legend(fontsize=7, ncol=2)
    for a_ in ax: a_.grid(alpha=0.3, which="both")
    fig.suptitle(f"N={N}^3, matched t in [{ts[0]:.2f},{ts[-1]:.2f}], {len(dirs)} seed dirs")
    fig.tight_layout(); fig.savefig(out, dpi=130); print(out)
    print(f"{'cfg':16s}" + "".join(f"  kin@k={k:<3d}      mag@k={k:<3d}    " for k in kshow) + "  floorfrac(mean)")
    for c in sorted(spec):
        mk, ek = ratio_stats(c, 1); mb, eb = ratio_stats(c, 3)
        row = "".join(f"  {mk[k]:.3f}±{ek[k]:.3f}  {mb[k]:.3f}±{eb[k]:.3f}" for k in kshow if k < len(mk))
        print(f"{c:16s}{row}  {np.mean(ffl[c]) if ffl[c] else float('nan'):.2e}")


if len(sys.argv) > 1 and sys.argv[1] == "--snap":
    # python3 plot_ducros.py --snap 'ducros_out/gpu_n128_r1e-3_k0_s*' 128 [out.png] [tmin] [tmax]
    g, N = sys.argv[2], sys.argv[3]
    out = sys.argv[4] if len(sys.argv) > 4 else f"ducros_out/snap_cmp_N{N}.png"
    tmin = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    tmax = float(sys.argv[6]) if len(sys.argv) > 6 else 1e9
    snap_mode(g, N, out, tmin, tmax)
else:
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
