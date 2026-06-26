# FiniteVolumeGodunovKA.jl

Write a finite-volume Godunov solver's *physics* once, in pure Julia, and run it fast on **CPU and
CUDA** — the KernelAbstractions *philosophy* (write-once, multi-backend) specialized to FV Godunov,
with the performant backends KA lacks for this domain. You declare a **system** (conserved variables +
a handful of pure, branch-free per-cell functions over a generic element type); the library owns all
the structure — reconstruction, the Riemann combination, the conservative update, tiling, occupancy
tuning, and backend dispatch.

```julia
using FiniteVolumeGodunovKA
s  = Euler(γ = 1.4f0)
U0 = [prim2cons(s, (x < 0.5f0 ? 1f0 : 0.125f0, 0f0,0f0,0f0, x < 0.5f0 ? 1f0 : 0.1f0))
      for x in range(0.5f0/400, step = 1f0/400, length = 400)]
g  = Grid1D(s, U0; dx = 1f0/400, bc = :outflow, recon = PLM(), rsol = HLLC())
evolve!(g, 0.2f0)          # the same physics runs on every backend below
```

## The one rule

The per-cell functions (`cons2prim`, `prim2cons`, `physflux_x`, `maxspeed_x`, …) must be **branch-free
on field values** (`ifelse`/`min`/`max`, never `if x>0`). That single rule lets the *identical* source
run as `Float32` scalars on a CPU thread, `Vec{8,Float32}` lanes on a SIMD core, and `Float32` on a GPU
thread — and lets it be transpiled to CUDA-C. You write only `physflux_x`; the library rotates it for
y and z (the `vidx` declaration marks which components rotate, including multiple vectors for MHD).

## Systems

- **`Euler`** — compressible hydrodynamics (5 vars), defined entirely through the `@fvsystem` contract.
- **`GLMMHD`** — GLM-MHD (9 vars, Dedner divergence cleaning) — the same contract: two rotating vectors
  (momentum + B), a ψ-damping `source`, dynamic `ch`, and `HLLD`.

## Backends (one source, all bit-identical where the scheme matches)

| | scalar | SIMD (threaded) | CUDA | transpile→nvcc |
|---|---|---|---|---|
| **1D** | `Grid1D` | `Grid1DSoA` | `Grid1DCU` | |
| **2D** | `Grid2D` | `Grid2DSoA` | `Grid2DCU` | |
| **3D** | `Grid3D` | `Grid3DSoA` | `Grid3DCU` | `Grid3DCuMarch` |

Plus **`Grid2DCT`** (constrained transport — face-staggered B + edge EMF → machine-zero div·B) and an
untested **Metal** backend (`metal/metal_2d.jl`, for Apple hardware). Riemann solvers: `LLF`, `HLL`,
`HLLC` (Euler), `HLLD` (MHD). Reconstruction: `PLM` (MC limiter / unlimited), `PCM`.

Every fast path is **bit-identical** to its scalar reference (max |Δ| = 0) on Sod/Brio-Wu/smooth-wave
across the solvers; rotation (1, 2, or 3 axes; momentum + B) is exact; schemes are 2nd order in 1D/2D/3D.

## Performance

All numbers are on one **NVIDIA RTX A6000** (sm_86), a 3D gradient initial condition, in **Mcell/s** —
millions of cell-updates per second (one full timestep per update; higher is faster).

### The hand-tuned `.cu` reference — what we measure against

The yardstick is a set of **hand-written CUDA-C kernels** (the `mini-ramses-metal` / `march_bridge`
reference), tuned for this exact GPU over a long campaign. They represent roughly the *practical
throughput ceiling* on an A6000, and they get there with techniques the user of this library never
writes: a **fused 2.5D z-streaming march** (the whole timestep in ~one global-memory pass),
**shared-memory staging**, **f16 tiles** for the reconstruction, `--use_fast_math`, and hand-picked tile
sizes / occupancy. The reference matrix (gradient IC @480³):

| scheme (2nd order) | Euler (hydro) | GLM-MHD | CT |
|---|---:|---:|---:|
| **PLM**            | **6865** | **3175** | 1255 |
| PLM + 2 species    | 5082 | 2733 | 1011 |
| PPM                | 3995 | 2064 |  752 |
| PPM + 2 species    | 3822 | 1410 |  558 |

Think of these as "as fast as a human expert hand-coded it for this card." The whole point of this
library is to get *close to that* — for **any** system — while you write only branch-free Julia.

### What this library delivers from one `@fvsystem` stencil

GPU, 3D Euler, vs the hand-tuned reference (6865):

| backend / scheme | what it is | Euler 3D | vs `.cu` 6865 |
|---|---|---:|---:|
| `Grid3DCU` (native CUDA) | dimensional-split **2nd-order**, CUDA.jl codegen | ~2550 Mcell/s | ~37–42% |
| `evolve!` `scheme=:rk2` | 2nd-order MUSCL + SSP-RK2 (pure f32, two stages) | ~3000 Mcell/s | ~44% |
| `evolve!` `scheme=:ctu` | single-pass 2nd-order, shared-mem-tiled f16 CTU | ~4220 Mcell/s | ~61% |
| `evolve!` `scheme=:march` | single-pass 2nd-order, **streaming z-march** (large grids) | **~4750 Mcell/s** | **~69%** |
| `evolve!` `scheme=:auto` (**default**) | picks march vs tiled by grid size | **~4220–4750** | **~61–69%** |
| `Grid3DCuMarch` · `run!` | 1st-order-in-time fused **single-pass throughput demo** | **~6300 Mcell/s** | **~91%** |

Two honest readings. (1) The **transpiler emits `.cu`-class code**: its fused single-pass kernel hits
**~85–90% of the hand-tuned reference** — and the transpiled physics is **bit-identical to your Julia
functions** (`transpile_selfcheck` → `0.0`). But that kernel is 1st-order in time (it can't alternate
sweep order the way a split solver does), so it's a *throughput demonstration*, not the science path.
(2) The **science path is `evolve!`** — CFL-adaptive and genuinely 2nd-order (entropy-wave convergence
order ≈ 2, conservation to machine precision, validated as CUDA tests). Its default `:ctu` scheme is the
**shared-memory-tiled, f16-tiled single-pass CTU** kernel — the reference `.cu`'s own technique,
reproduced from the transpiler: each cell's transverse correction is computed *once* into a half-precision
shared tile (the f32 update keeps conservation exact). After a profiling-driven audit — compute each
interface flux **once** into a shared flux tile (not twice per cell), one-sided PLM reconstruction, and
an NV-aware tile size (8³ for Euler) to amortize the halo — the tiled kernel reaches **~61% of the
reference**, 2.1× over the naive single-pass and 1.4× over pure-f32 RK2 (`scheme=:rk2`, full f32).

Then the reference's actual architecture: a **streaming z-march** (`scheme=:march`) — a 2D (x,y) block
that marches through z keeping a rolling window of 5 W-planes + 3 dU-planes in shared, so each plane is
read from global **once** (over-read 3.4× → 1.6×). Reproduced generically from the transpiler, it's
validated 2nd-order and conservative. It has far fewer blocks (one per z-column), so it *underfills* small
grids but **wins at production scale** as its memory advantage compounds. A z-flux carry (each z-interface
flux is computed once and carried down the march, not recomputed by the next plane) lifts it to **64% at
384³, 69% at 512³ and climbing** (vs 54% at 256³, where the tiled kernel still wins). `scheme=:auto` (the
default) picks the march for large grids and the tiled kernel otherwise. Either way you never hand-write a
CUDA kernel, a march, or an f16 tile.

**On the CPU**, the same source runs as a SIMD-vectorized, threaded solver: ~1 Mcell/s scalar (1 core,
3D) → **~145 Mcell/s** at the threaded peak (2D, `JULIA_NUM_THREADS ≈ 8–16`). These kernels are
memory-bandwidth bound, so they peak at ~8–16 threads — **don't over-subscribe** (`-t auto` on a
128-core box is the *worst* setting).

```julia
g = Grid3DCuMarch(Euler(γ=1.4f0), U0; dx=dx)   # transpiles the stencil + nvcc-compiles + loads
evolve!(g, 0.2f0)                              # the science path: CFL-adaptive, 2nd-order (MUSCL+SSP-RK2)
run!(g, dt, 1000)                              # the fast 1st-order single-pass throughput demo
transpile_selfcheck(Euler(γ=1.4f0))            # 0.0 — emitted CUDA-C ≡ Julia physics, bit-exact
```

**Honest caveats.** Comparisons are *scheme-matched* (PLM vs PLM); a cheaper pure-1st-order kernel runs
~10,800 Mcell/s but that's not a fair comparison. The `run!` single-pass demo is 1st-order *in time* —
use `evolve!` (2nd-order, validated) for science. GLM-MHD transpiled uses LLF where the reference uses
HLLD (a Riemann mismatch). The default `:ctu` scheme uses an f16 shared tile for reconstruction (the
f32 update keeps conservation exact); `scheme=:rk2` is the pure-f32 path. Throughput varies ~±15% with
GPU thermal state, so compare runs in one process.

## Design

See `DESIGN_fvkernel.md` for the contract rationale, the rotation/Riemann design, every optimization
(threading, alternating Strang, the transpile backend) with its measured outcome, and the documented
negatives (lagged-dt, `Vec{16}` on AVX2). Run the tests: `julia --project=. -e 'using Pkg; Pkg.test()'`.
