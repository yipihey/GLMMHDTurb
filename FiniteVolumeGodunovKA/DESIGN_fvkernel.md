# FiniteVolumeGodunovKA — the `@fvkernel` contract

**Goal.** Write a finite-volume Godunov solver's *physics* once, in pure Julia, and run it
fast on **CPU and CUDA** (Metal a desirable bonus) — covering the full solver suite we
benchmarked: hydro / GLM-MHD / CT × PLM/PPM × ±species, with HLL/HLLC/HLLD/LLF. The library
exists because CUDA.jl/KA codegen tops out at ~60–67% of hand-written `.cu` for these kernels
(see `../JULIA_NATIVE_PERF.md`); the performance backends route *around* that wall (SIMD on
CPU, transpile-to-native on GPU) while the user authors only Julia.

This is the KernelAbstractions *philosophy* (write-once, multi-backend) specialized to FV
Godunov, with the performant backends KA lacks for this domain. The name keeps the KA signal;
the README must be explicit that it's the performant counterpart, not a thin KA wrapper.

## The split: 5 pure functions from the user, all structure from the library

A **system** is a `<: FVSystem` value; the per-cell physics are methods on the contract's
generic functions, written over a generic element type `T` and **branch-free on field values**
(`ifelse`/`min`/`max`, never `if x>0`). That single rule is what lets the identical source run
as `Float32` scalars on a CUDA thread *and* `Vec{W,Float32}` lanes on a CPU core.

```julia
@fvsystem Euler begin
    nvars = 5
    vidx  = (2, 3, 4)                 # momentum components — ROTATE under dim permutation
    @params γ = 5f0/3f0

    cons2prim(U, p)  = ...            # required: conserved → primitive
    prim2cons(W, p)  = ...            # required: primitive → conserved
    physflux_x(W, p) = ...            # required: x-direction physical flux ONLY
    maxspeed_x(W, p) = ...            # required: |u_x| + c   (CFL + LLF)
    eig_x(W, p)      = ...            # optional: (u_x, c)    (HLL/HLLC)
end
```

The last arg `p` is the params placeholder; `@fvsystem` desugars to `struct Euler <: FVSystem`
+ `@inline` methods on the contract functions (Trixi-style: idiomatic, dispatch-extensible).
GLM-MHD is the *same shape* — `@conserved`-equivalent tuple gains `B[3], ψ`, `@params` gains
`ch`, plus a ψ-damping source; no new concepts. That hydro and GLM share this contract is the
test that it's the right 80%.

### Two design wins that keep it small

1. **Rotation is the library's job.** The user writes *only* `physflux_x`. Because `vidx`
   marks the rotating vector components, the y/z sweeps permute those into the x-slot, call the
   same `physflux_x`, and permute back — one flux function, three directions, exactly as the
   `.cu` marches do.
2. **Riemann is layered.** From `physflux_x`+`maxspeed_x` the library assembles **LLF** for any
   system; with `eig_x` it adds **HLL**. **HLLC (Euler)** and **HLLD (MHD)** ship as built-ins
   keyed to the system, since they need the wave structure. `riemann=HLLC()` is a launch knob,
   never user-reimplemented.

## What the user never writes (library services)

Reconstruction (PLM/PPM + MonCen limiter), the staged shared-memory **cube** and the **march**,
the f16-tile+f32-update rule, the conservative update, halos/boundaries, the occupancy-cliff
auto-tuner (per device), and backend dispatch. `@fvkernel` is the compiler entry that fuses
*system + recon + riemann + backend* into the tiled kernel; `step!`/`evolve!` are the same call
on every backend.

## Three locked design decisions

1. **State = `NTuple{N,T}` under the hood**, named destructuring at the macro surface. Keeps the
   fast path tuple-simple (trivially SIMD-able) and the user code readable.
2. **`physflux_x` + `maxspeed_x` are the required primitive** (→ LLF/HLL for free); HLLC/HLLD are
   built-in opt-ins keyed to the system. A user may still supply a hand-written `riemann` method.
3. **CT seam reserved, not built.** v0 `@fvsystem` is cell-centered only, but the grammar
   reserves `@staggered B[3]` + an `@emf` flux→edge-EMF→curl hook so CT slots in without
   reshaping the contract. Reserve now; implement after CPU+CUDA cell-centered is solid.

## Status

Implemented and **passing** (`test/runtests.jl`):
- `@fvsystem` contract + the `Euler` system defined entirely through it.
- Library PLM (MC limiter for shocks, `:none` unlimited for smooth) + LLF/HLL/HLLC.
- A reference **CPU scalar 1D** backend (`Grid1D`, MUSCL-Hancock, periodic/outflow BCs).
- A **SIMD CPU 1D** backend (`Grid1DSoA`): SoA state, vectorized along the grid with shifted
  vector loads, `T = Vec{8,Float32}` lanes + a scalar tail — reusing the per-cell physics verbatim.
- A **CUDA 1D** backend (`Grid1DCU`): one GPU thread per cell, fused per-cell recompute, `T =
  Float32`, in-kernel BC (mod1/clamp), double-buffered device state — the same physics verbatim.
- **Sod shock tube (Float32, HLLC):** post-shock ρ=0.2655 / P=0.3031 vs exact 0.2656 / 0.3031;
  positivity + mass conservation hold.
- **Entropy-wave convergence (Float64):** L1 order 2.34 → 2.15 → 2.05 → **2.01** (nx 16→256).
  Run in Float64 from the *same* Float32-authored physics — the element-type genericity that every
  backend depends on, demonstrated end-to-end.
- **One source, three backends, all bit-identical.** SIMD ≡ scalar AND CUDA ≡ scalar — max |Δ| = 0
  on Sod and the smooth wave, across HLLC/HLL/LLF (SIMD also on a non-multiple-of-8 grid, exercising
  the tail). The element-generic physics is provably the same code on a CPU thread, a SIMD lane, and
  a GPU thread. Throughput (A6000): **CPU scalar ~9–14, CPU SIMD ~60 (4.7×–6.8×, single core), CUDA
  ~11,400 Mcell/s** (≥1M cells; ~190× the single-core SIMD).

## Roadmap (priority order)

1. ~~**SIMD-CPU backend**~~ ✅ done (single-core, ~5–7× over scalar, bit-identical). Next CPU
   increments: **threads + cache-blocking** (toward the `cpu_simd.jl` ~120 Mcell/s with NUMA), and
   `Vec{16}` on AVX-512 hosts.
2. ~~**CUDA backend**~~ ✅ done (1D, fused per-cell, bit-identical to CPU, ~11,400 Mcell/s). Open
   items: the staged shared-memory **cube** in multi-D; whether CUDA.jl plateaus below the `.cu`
   here (it didn't for this register-light fused 1D kernel) → the transpile-to-CUDA-C escape hatch
   only where needed; move CUDA to a **weakdep + package extension** so CPU-only installs stay light.
3. **Multi-D + rotation** exercised (2D Sod/Kelvin-Helmholtz), then **GLM-MHD** via the same contract.
4. **Metal** — measure the Metal.jl gap vs its bandwidth roofline before deciding native-vs-MSL.
5. **CT** through the reserved staggered/EMF seam.

Conformance lives in the parent `GLMMHDTurb` repo (OT, Sod, turbulence, div·B, the gradient-IC
benchmarks): the spec is "reproduce that matrix from one physics source on every backend."
