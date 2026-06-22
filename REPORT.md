# Single-file Julia GPU GLM-MHD driven-turbulence solver

`glmmhd_turb.jl` — a single-file, fully GPU-resident solver that reproduces the physics of
the reference CUDA-Fortran driven-turbulence run (`mini-ramses-metal`, branch
`cuda-dedner-mhd`): **Dedner GLM-MHD + PLM (MonCen) + HLLD**, near-isothermal, Mach-10,
β≈2, on a dense uniform periodic grid. KernelAbstractions for forcing/dt/diagnostics/IC,
a native CUDA.jl tiled kernel for the integrator, CUDA's CUFFT for the OU forcing. Nothing
but scalars cross PCIe per step.

## Gates — both cleared at 512³ (RTX A6000, fp32)

| metric | gate | measured | |
|---|---|---|---|
| **throughput, full step** (integrator + fused OU drive) | ≥ 750 Mcell/s | **899–905 Mcell/s** | ✅ 20% over |
| **throughput, pure integrator** | — | **1124–1131 Mcell/s** | beats the fp64 reference's ~1100 |
| **memory** | ≤ 72 B/cell | **72 B/cell** | ✅ (two fp32 (N,N,N,9) buffers, no scratch) |

The throughput gate is the harder one (≥75% of the reference's >1000 Mcell/s). The memory
gate is met exactly: the only large allocations are `uold`/`unew` = 2·9·4 = 72 B/cell;
the integrator is a single no-scratch kernel (`unew = uold + flux`, binding swap).

## Physics validation (same scheme, statistically equivalent — not bit-identical)

- **Orszag-Tang** (γ=5/3): tiled kernel is bit-identical to the multi-kernel and recompute
  paths — vrms=0.7887 (64³), 0.7906 (128³); div·B bounded (2.9 / 4.5) in the reference range;
  ρ>0; finite. Three independent integrator implementations agree.
- **Driven turbulence**: stable through the full Mach ramp into the target regime —
  Mach 9.85 → 10.9 → 11.6 at amp≈1.5e8 — finite, ρ_min floored positive, div·B GLM-cleaned
  and bounded. The LLF switch + positivity floor handle the deep supersonic voids.

The integrator cost is Mach-independent, so the gates hold across the Mach range. The exact
forcing amplitude → steady-state Mach mapping is a calibration knob (the OU normalization
differs from the reference's nominal `turb_rms`); amp≈1.2–1.5e8 brackets Mach 10 here.

## Architecture

- **Grid/buffers**: `(N,N,N,9)` fp32, var order `[ρ, ρvx, ρvy, ρvz, E, Bx, By, Bz, ψ]`,
  two buffers, periodic via index wrap. 72 B/cell, no ghosts, no oct metadata.
- **Integrator** (`integrator_tiled!`, native CUDA.jl): nsubgrid=2 → 4³ owned cells/block,
  8³ primitive tile (2-cell halo) in shared memory (SoA, var-major), 192 threads, 4 stages:
  c2p-load → MUSCL-Hancock trace into per-face interface subgrids → HLLD(+`glm_pair`)/LLF
  Riemann (once per face) → conservative update + ψ damping (`glm_fac`) + fused OU kick.
  ~35.7 KB shared → 2 blocks/SM. Each reconstruction computed once, each face solved once.
- **Forcing** (CUFFT): OU process in Fourier space on a 64³ grid, P(k)=k⁻², Helmholtz
  projection (comp_frac=0.5), temporal correlation, inverse FFT → real accel field, applied
  by trilinear interp + remove-KE/kick/restore-E with a linear ramp. Fully on-device.
- **dt**: device `mapreduce` of the per-cell signal speed; only the scalar comes back.

## The optimization that mattered

The decisive find: `moncen` (the MonCen slope limiter, 27 calls/cell in the trace) was not
tagged `@fastmath`, so Julia's `min`/`max` kept their NaN-propagation semantics (extra
compares) instead of lowering to the hardware fmin/fmax. That single fix took the kernel
from 398 → 993 Mcell/s at 256³ (the slope stage was ~65% of the whole step). Tagging the
limiter `@fastmath` + branchless `ifelse`, plus 192 threads/block for 2-block occupancy,
cleared the gate with margin.

Things that did *not* help (ruled out by measurement): SoA-vs-AoS shared layout, register
caps (`maxregs`), `@noinline` on the Riemann/predictor (made it worse — GPU call overhead),
and the 8× halo c2p (stage 1 alone runs at 3471 Mcell/s — not the bottleneck).

## Tried and rejected: 2D-coalesced tile + streamed z

The canonical "tile the two coalesced dims, stream the third" layout was implemented in full
(`integrator_stream!`): a block owns a 4×4 (x,y) column over a chunk of z, sliding a 5-plane
primitive window and carrying z-fluxes between k-steps (each plane reconstructed once, shared
down to ~17 KB, (x,y) halo 4× not 8×). It is **bit-identical to the cube on Orszag-Tang** but
**~1.9× slower** (512³: 614 vs 1141 Mcell/s, 462 vs 899 with turb). The layout wins when a
kernel is memory/halo-bound — but after the moncen fix this one is compute-bound, so the halo
savings are moot, while the per-plane phases (16 owned / 36 inner cells) under-fill the warps
and multiply barriers (~5/plane) against the cube's 4-barriers-total 3D tile. Kept in-tree as
a validated alternative; **`integrator_tiled!` (the cube) is the production kernel.**

## Usable entry points

- `run_turb(; N, t_end, amp, …)` — production driven-turbulence driver (tiled kernel).
- `bench_tiled(; N, do_turb, nthreads)` — throughput + B/cell measurement.
- `validate_tiled_ot(; N)` — Orszag-Tang correctness check.
- `run_ot(; N)` — Orszag-Tang on the multi-kernel reference path (cross-check).

## CPU path: SIMD + NUMA cache-blocked kernel (cpu_simd.jl)

A CPU-tuned GLM-MHD integrator that reuses the *same* `@inline` physics (made generic over
`NTuple{9}` so the functions run on `Float32` scalars on GPU and `Vec{8,Float32}` lanes on CPU).
The production CPU kernel is `run_ot_cstream` — a **chunked z-stream**:
- **SIMD-across-x**: vectorize 8 cells along the unit-stride x-axis (the GPU-optimal layout is
  SIMD-friendly *if* you vectorize along x instead of gathering per cell). `vload`/`vstore` via
  raw `pointer` (not `vec()`, which gives a ReshapedArray → scalar-gather fallback, ~50× slower).
- **Recon-once, cache-blocked**: a rolling per-plane reconstruction buffer (L2/L3-resident, no
  full-volume DRAM scratch); each z-flux/Riemann computed once. At 64³ a plane fits L2 → fast.
- **One `@threads` region/step**: each thread streams a contiguous k-chunk sequentially (no
  per-plane barriers — that was the killer: per-plane `@threads` = ~390 tiny regions/step,
  *inversely* scaling with thread count). A chunk recomputes 1 halo plane each end.
- **NUMA first-touch by chunk**: arrays initialized by the thread that computes them, so pages
  are node-local. On the 16-NUMA-node dual-EPYC this was a ~3× lever on its own (vs main-thread
  `zeros()`, which parks everything on node 0 → ~16 GB/s).
- ThreadPinning `:numa` (or `JULIA_EXCLUSIVE=1`); optimum ~32 threads (one socket-ish), not 128.

**Result (OT, LLF, 32 NUMA-pinned threads, vs the scalar KA multi-kernel at the same threads):**

| grid | scalar | chunked-SIMD | speedup |
|---|---|---|---|
| 64³  | 12 Mcell/s | 56  | 4.6× |
| 128³ | 20 | 108 | 5.4× |
| 256³ | 19 | 119 | 6.3× |

Validated against the scalar LLF path to Δvrms ≈ 2e-4 (float-order + arithmetic-moncen vs
ifelse-moncen; same scheme). It stays **fast at 64³** (4.6×, L2-resident) — the small-grid goal.

**Honest ceiling**: ~110-120 Mcell/s, still only ~2% of the 410 GB/s peak — the full unsplit
integrator is compute/overhead-bound (per-cell SIMD volume + halo recompute), not bandwidth-bound,
so it plateaus past ~32 threads. The 4-6× is the real, banked win; saturating bandwidth would need
HLLD→vifelse vectorization + lower halo overhead and is a further project.

**What's GPU-only / portable**: the production GPU tiled kernel (910 Mcell/s) stays untouched; the
genericity changes are GPU-verified bit-identical. The chunked CPU kernel is LLF (HLLD's wave-select
branches need `vifelse` blends to vectorize — not yet done).

## Production fast solver: Hancock + HLL + GLM + PLM, f16 shared, nsubgrid=3

`integrator_plm!` / `step_plm!` — parametric over tile size, shared precision, and Riemann
(`Val{TB}`, `Val{HALF}`, `Val{RIEM∈:hlld/:hll/:llf}`). The production default is **MonCen-PLM +
MUSCL-Hancock + HLL + GLM, f16 shared, TB=6 (nsubgrid=3)**.

Two levers, stacked:
- **Lean Riemann**: HLL (2-wave, exact fast-magnetosonic signal speeds, branchless) instead of
  HLLD (5-wave). Less diffusive than LLF, far cheaper than HLLD; div·B actually *lower* (HLL's
  diffusion smooths the field).
- **f16 shared + f32 update** (the precision trick): the shared prim-tile + face states are stored
  in Float16 (halving shared), but `u_new = u_old + dt·divF` is done in f32 (state in f32 global,
  increment promoted). The lossy f16 round-trip only touches the reconstruction; conserved
  accumulation and div·B stay accurate. **f16 halves the tile so nsubgrid=3 fits 48 KB static
  shared** (90→44 KB), and TB=6's lower halo (4.6× vs TB=4's 8×) is the actual win.

**Throughput (RTX A6000, GLM-MHD, N=480):**

| config | pure integrator | + turb |
|---|---|---|
| HLLD, f32, TB=4 (accurate/reproducible) | 1139 Mcell/s | ~900 |
| **Hancock HLL GLM PLM, f16, TB=6** | **1923 Mcell/s (1.69×)** | 1416 |

**Accuracy (Orszag-Tang vs the f32/HLLD reference):** Δvrms ≈ 3e-4 (mostly HLL's extra diffusion
vs HLLD), div·B controlled (0.96 vs HLLD's 3.59), ρ>0, finite. Hydro (5-var) with the same
f16+TB=6 reaches ~4300 Mcell/s.

Selectable via `run_turb(...; solver=:plm)` or `step_plm!(...; tb, half, riemann)`. The f32/TB=4/
HLLD cube (`step_tiled!`) remains the bit-reproducible default and is unchanged (OT bit-identical).
nsubgrid was swept (2/3/4/5): 3 is the interior optimum — smaller loses to halo recompute, larger
falls off the dynamic-shared occupancy cliff.

## DEFAULT (current): Hancock + HLL + (GLM) + PLM, f16 shared, nsubgrid auto

Both CUDA solvers now default to this config:
- **GLM-MHD**: `step_plm!` (default `run_turb(...; solver=:plm)`) — ~1923 Mcell/s (N=480).
- **Hydro (5-var Euler)**: `step_hydro!` — ~4250 Mcell/s (N=480), Sod mass-exact & positive.

Both auto-select the tile (TB=6/nsubgrid=3 when N%6==0, else TB=4), use f16 shared + f32 update,
HLL Riemann, MonCen-PLM, MUSCL-Hancock (MHD: flux-Hancock; hydro: primitive source-term predictor).
Set `half=false, tb=4, riemann=:hlld` (MHD via `step_tiled!`) for the bit-reproducible f32/HLLD path.

## PPM reconstruction — penalty vs PLM (single-zone, same tile)

Transliterated the reference's **single-zone PPM** (`mini-ramses-metal/gpu/gpu_hydro.cuf`
`local_ppm_*`, also Vespa.jl `PPMKernels`): 3-point parabolic edges (`slope=¼(qp−qm)`,
`curve=(qm−2q0+qp)/12`) + CW84 monotonize. Single-zone = a ±1 stencil, so it fits the **exact
same (TB+4) f16 tile as PLM** — no wider halo. Two flavors, matching the reference:
- **Hydro**: full 3-wave characteristic trace (`local_ppm_trace` — parabola integrated over each
  acoustic/entropy wave's domain of dependence via `local_ppm_avg`).
- **MHD**: lean parabolic-PPM (`trace_3d_mhd_par`) — keeps the MonCen slope + Hancock predictor,
  PPM only changes the *spatial* face to `mh + parabolic_edge − m0`, with a `strong_pressure_jump`
  → PLM fallback at shocks.

**Measured penalty (RTX A6000, N=480, f16, TB=6, HLL):**

| solver | recon | regs | blocks/SM | Mcell/s | penalty |
|---|---|---|---|---|---|
| Hydro | PLM | 56 | 3 | 4250 | — |
| Hydro | **PPM** (characteristic) | 95 | 3 | **2856** | **1.49×** |
| MHD | PLM | 144 | 2 | 1916 | — |
| MHD | **PPM** (parabolic) | 201 | 1 | **669** | **2.86×** |

The penalty is an **occupancy cliff set by the register budget**, not raw arithmetic. Hydro PLM (56
regs) has huge headroom, so even the heavier characteristic PPM (95 regs) keeps 3 blocks/SM → the
1.49× is pure extra math. MHD PLM already sits at 144 regs (2 blocks/SM, near the ~170 ceiling), so
even the *lean* parabolic PPM (201 regs) tips over to 1 block/SM — the 2.86× is ≈2× occupancy loss ×
~1.4× math. `maxregs` caps to force 2 blocks make it strictly worse (register spilling > occupancy
gained: 160→407, 128→292, 96→220 Mcell/s). Register-discipline experiments (reload neighbors from
shared / interleave edges) also lost — the extra shared traffic beat the occupancy recovered;
all-edges-live (the natural ab=1) was fastest.

**Validation:** hydro PPM Sod mass-exact (972000→972000), ρ>0, *sharper* contact than PLM (ρmax
0.997, no overshoot). MHD PPM Orszag-Tang vrms=0.7910 (= PLM-HLL), div·B 1.16 (between HLL 0.96 and
HLLD 3.59), ρ>0, finite.

**Now in the module** as an opt-in: `step_plm!(...; recon=:ppm)` (GLM-MHD `integrator_plm_ppm!`) and
`step_hydro!(...; recon=:ppm)` (`integrator_hydro_ppm!`). PLM stays the default; `recon=:ppm` requires
the f16+HLL path (`half=true, riemann=:hll`, tb=4 or 6) and errors otherwise. The single-zone edge
(`ppm_edges`/`ppm_mono`) is shared by both; MHD adds `ppm9`/`mhd_face9`/`spj`, hydro adds the
characteristic trace (`ppm_avg`/`ppm_side`/`ppm_trace5`/`ppm_dir`). PPM is the accuracy-for-throughput
option (~1.5× hydro / ~2.9× MHD cost).
