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

## Performance (A6000, gradient IC, vs the hand-tuned reference `.cu`)

- **CPU SIMD**: ~145 Mcell/s with `JULIA_NUM_THREADS ≈ 8–16` (bandwidth-bound — don't over-subscribe).
- **CUDA (native)**: ~37–42% of the `.cu` (dimensional split + CUDA.jl codegen).
- **Transpile-to-nvcc** (`Grid3DCuMarch`): emits the `@fvsystem` stencil as CUDA-C, compiles with nvcc
  `--use_fast_math`. The transpiled physics is **bit-identical to the Julia functions**; the
  scheme-matched PLM kernel reaches **~85–90% of the hand-tuned `.cu`** — `.cu`-class speed from the same
  branch-free source. (Needs nvcc at construction time; the package loads fine without it.)

```julia
g = Grid3DCuMarch(Euler(γ=1.4f0), U0; dx=dx)   # transpiles + nvcc-compiles + loads
run!(g, dt, 1000)                              # ~.cu-class throughput
transpile_selfcheck(Euler(γ=1.4f0))            # 0.0 — transpiled C ≡ Julia physics
```

## Design

See `DESIGN_fvkernel.md` for the contract rationale, the rotation/Riemann design, every optimization
(threading, alternating Strang, the transpile backend) with its measured outcome, and the documented
negatives (lagged-dt, `Vec{16}` on AVX2). Run the tests: `julia --project=. -e 'using Pkg; Pkg.test()'`.
