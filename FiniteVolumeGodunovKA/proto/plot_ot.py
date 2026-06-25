#!/usr/bin/env python3
"""Render the Orszag-Tang density (saved by orszag_tang.jl) to a PNG."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

n = int(open("proto/ot_meta.txt").read().split()[0])
# Julia wrote column-major (nx,ny); read and transpose so x is horizontal.
rho = np.fromfile("proto/ot_density.bin", dtype=np.float32).reshape((n, n), order="F")

fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
im = ax.imshow(rho.T, origin="lower", extent=[0, 1, 0, 1], cmap="inferno", interpolation="bilinear")
ax.set_title(f"Orszag-Tang density, t=0.5  ({n}²)\nFiniteVolumeGodunovKA — GLM-MHD on GPU")
ax.set_xlabel("x"); ax.set_ylabel("y")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ρ")
fig.tight_layout()
fig.savefig("proto/ot_density.png")
print("wrote proto/ot_density.png")
