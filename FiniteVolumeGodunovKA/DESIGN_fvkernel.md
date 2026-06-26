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
- A **2D CPU** backend (`Grid2D`): Strang dimensional splitting (x·y·x), where the y-sweep reuses the
  per-cell physics through `_update_dir` + the rotation perm — **the user wrote only `physflux_x`**.
- **GLM-MHD** (`GLMMHD`) defined through the *same* contract: 9 vars, two rotating vectors (momentum
  AND B via `vidx = ((2,3,4),(6,7,8))`), the Dedner flux, `ch` param, and a `source` (ψ-damping) — runs
  on every backend.
- A **2D CUDA** backend (`Grid2DCU`): one thread per cell, the same Strang sweeps + source, reusing
  `_update_dir` + the perm. Bit-identical to the 2D CPU backend; runs Orszag-Tang on the GPU.
- An optional **`@source`** contract hook (operator-split), applied after each step; default identity.
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
- **Rotation is exact (the design-defining result).** A single y-sweep — the y-flux obtained purely by
  swapping the marked vector components and calling the same `physflux_x` — is **bit-identical**
  (max |Δ| = 0) to the x-sweep on the transposed problem, for **both Euler (1 vector) and GLM-MHD
  (2 vectors, momentum + B swapped together)**. The 2D Strang scheme is 2nd order (diagonal entropy
  wave, order 2.22 → 2.09). The user writes one flux function; the library does y and z.
- **The contract scales to MHD.** Going Euler → GLM-MHD was: add 4 variables, add the `ch` param, write
  `physflux_x`, and declare `vidx` as *two* triples. The rotation machinery (a compile-time permutation
  over all triples) handled the rest. Brio-Wu shock tube: stable, positive, mass-conserving, and the
  normal field Bx is preserved to **max |Bx − 0.75| = 0** — exactly.

## Documented negative — device-resident "lagged" dt (don't re-attempt naively)

Hypothesis: the per-step `dt` (a wavespeed reduction read back to the host) serializes the GPU, so
fold the wavespeed into the sweep (free — it already evaluates `maxspeed_x`) to set the *next* step's
`dt`, keep `dt`/`t` on the device, and sync only every K steps. Built it, validated it **correct**
(agrees with `:exact` to max |Δρ| ≈ 5e-4, identical div·B), and measured: it ran **~8× slower** (256²
OT t=0.5: 0.06 s exact vs 0.53 s lagged). Two lessons:

1. **The per-step host sync was never the bottleneck.** `:exact` runs 256² OT in **0.06 s** (~85 µs/step
   incl. the speed kernel + `maximum` reduction + sync). An earlier "2.6 s" figure was *compilation*,
   not runtime — always warm up before timing.
2. **Don't replace a parallel reduction with a single-address atomic.** Folding the wavespeed in via
   `CUDA.@atomic amax[1] = max(...)` makes all 65536 cells atomic-max into *one* scalar → fully
   serialized (~700 steps × 65536 atomics ≈ the 0.5 s observed). `maximum(spd)` is a proper
   hierarchical reduction and wins decisively.

If profiling ever shows the dt sync mattering at scale, revisit with a *block-hierarchical* reduction
(one atomic per block), never per-cell atomics. For now `:exact` is the right design.

## vs the hand-written `.cu` (the honest gap)

3D CUDA, gradient IC, matched grid sizes: **Euler ~36%, GLM ~25–40% of the `.cu` peak** (hydro 6865,
GLM 3175 Mcell/s). The gap is **mostly algorithm, not language**: dimensional splitting does several
global-memory passes/step vs the `.cu`'s fused single-pass march + shared-memory staging + f16 tiles;
the CUDA.jl codegen residual (~60–67% of nvcc) is a smaller factor on top. This is the deliberate trade —
~25–36% from *one* branch-free source that runs bit-identically across scalar/SIMD/CUDA × 1D/2D/3D ×
Euler/GLM/CT. The path to full `.cu` speed from one Julia source is the transpile-to-C escape hatch or
`march_bridge`, **not** the native backend (the `.cu` structures don't codegen well in CUDA.jl).

**Wins banked — pass reduction (all backends, 2D + 3D).**
- **Alternating Strang**: 3D x·y·z·y·x (5 sweeps) → 3 full-dt sweeps with order alternating per step
  (x·y·z / z·y·x); 2D x·y·x (3 sweeps) → 2 (x·y / y·x). Consecutive steps form a symmetric, 2nd-order
  pair. Same-process A/B (controls for GPU thermal throttling — which made naive cross-run numbers
  misleading): 3D step is **1.66× faster**.
- **Skip the source pass** when `has_source(s)` is false (Euler). Marginal for 3D (the source kernel is
  cheap/bandwidth-bound while the sweeps are compute-bound — removing it saves little wall-time), but
  it's free and helps relatively more in 2D; the real lever is the sweep-count reduction.

Net (3D CUDA, gradient IC, cool GPU): **Euler ~37–39%** (2541–2652 vs 6865) and **GLM-MHD ~40–42%**
(1263–1332 vs 3175) of the `.cu` — up from ~24–25%. 2nd-order convergence and backend bit-identity both
preserved (a single `step!` is x·y·z on every backend; `evolve` alternates).
The remaining gap to the `.cu` is the fused single-pass march + shared-memory staging + f16 — structures
that don't codegen well in CUDA.jl — so closing it needs the transpile-to-C / `march_bridge` path.

## The performance backend — transpile-to-CUDA-C (v0 PROVEN)

The point of the project: full `.cu`-class speed from ONE `@fvsystem` stencil. `transpile/transpile.jl`
is a working v0 for `Euler`:
- A small **Julia-Expr → CUDA-C transpiler** walks the `@fvsystem` physics Exprs (cons2prim, prim2cons,
  physflux_x, maxspeed_x) and emits `__host__ __device__` C (handles +,-,*,/, inv/sqrt/abs/min/max/
  ifelse, tuple destructuring/return, param access, unicode→ascii names).
- nvcc `--use_fast_math` → `.so`, run over `CuArray`s via the `march_bridge` `ccall` mechanism.
- **The transpiled C physics is BIT-IDENTICAL to the Julia functions** (max|Δ| = 0 over 2000 random
  states) — the part that proves the transpiler.
- A fused single-pass **PLM MUSCL-Hancock** nvcc kernel (2nd-order in *space*) reaches **85–90% of the
  hand-tuned `.cu` 6865** (5236–6428 Mcell/s, 256–480³) — vs the native CUDA.jl backend's ~37%. This is
  the *throughput demonstration* that the transpiler emits `.cu`-class code. **But it is 1st-order in
  *time*** (see "Making it a real solver" below) — a fused kernel does all directions simultaneously
  (Lie splitting) and can't alternate sweep order, so it is a benchmark, not the science path.

So the two-backend design is fully realized: **portable native (~37%, bit-identical on scalar/SIMD/CUDA
× 1D/2D/3D, runs everywhere) + transpile-to-nvcc (~90% of the `.cu`, NVIDIA), from one `@fvsystem`
stencil.** The last ~10% to the `.cu` is its hand-tuned 2.5D march + shared-memory staging + f16 — a
fixed, system-agnostic C-template optimization that would lift every transpiled system at once.

**GENERALIZED (v1) — works for ANY `@fvsystem` system.** `@fvsystem` now emits `_fvmeta(sys)` (the
stencil captured as data: nvars, vidx, params, physics Exprs). The transpiler reads it and handles
arbitrary **params** (→ a `PRM[]` array), **NVARS**, the **`vidx` rotation** (generated `swap_y`/`swap_z`
over all vector triples), and **inter-physics calls** (GLM `maxspeed_x`→`fastspeed_x`). The
reconstruction/Riemann are NVARS-generic fixed C. Validated on BOTH from one pipeline:
- **Euler** (5 var, 1 param, 1 triple): physics bit-identical, PLM **85–90% of `.cu`**.
- **GLM-MHD** (9 var, 3 params, 2 triples): physics bit-identical (max|Δ|=0), PLM **104–107% of `.cu` GLM
  3175** (faster because it's LLF vs the `.cu`'s HLLD — a Riemann mismatch, like the 1st-order upper
  bound; the point is it *generalizes* and hits `.cu`-class).

So the transpile backend is general: write any system in `@fvsystem`, get `.cu`-class CUDA from the same
source. Remaining: HLLD in the transpiled kernel (fair GLM compare); the march/f16 C template; CT
transpile (staggered structure).

## Making it a real solver — CFL timestepping + 2nd-order, validated

The `Grid3DCuMarch` v0 was a fixed-`dt` throughput demo. Promoting it to a usable solver surfaced a
real scheme finding and a fix:

- **On-device CFL timestep.** A `k_speed` kernel writes per-cell `(sx+sy+sz)` (the summed directional
  signal speeds — the unsplit-CFL quantity); the host reduces the max and sets `dt = cfl·dx/max`.
  `evolve!(g, tend)` integrates to a physical end time, recomputing `dt` every `dtevery` steps.
- **The convergence test caught a 1st-order scheme.** On a smooth entropy wave (exact solution known),
  the fused single-pass kernel converged at **order ~1.1**, while the scalar backend — same MC limiter,
  same problem — hit **~2.0**. So it wasn't the limiter: the fused kernel is **1st-order in time**
  (simultaneous directions = Lie splitting; a fused kernel *can't* alternate sweep order, which is how
  the split backends recover 2nd order).
- **Fix: MUSCL + SSP-RK2.** Added a Hancock-free PLM+LLF flux operator integrated with SSP-RK2 (Heun) —
  textbook 2nd-order in space *and* time. `evolve!` uses it. Validated: entropy-wave convergence
  **order ≈ 1.9**, mass & energy conserved to **machine zero** (periodic), and it agrees with the scalar
  2nd-order backend to **below** each one's own discretization error. These are now CUDA-guarded tests.
- **The cost, honestly.** RK2 is two stages → ~2× the single-pass demo: the 2nd-order solver runs
  **~2580 Mcell/s (~38% of the `.cu`)**, on par with the native 2nd-order CUDA backend. The reference
  reaches 2nd order in *one* pass via a transverse **CTU** predictor; porting CTU is the route to a
  single-pass 2nd-order transpile kernel (recovering most of the 2× toward the ~85–90% the demo shows).

Net: `run!` = the 1st-order-in-time single-pass **benchmark** (~85–90% of `.cu`); `evolve!` = the
**validated 2nd-order science solver** (~38%, CFL-adaptive). Both from the same `@fvsystem` stencil.

### Single-pass 2nd-order (CTU) — implemented, and the measurement that matters

To recover single-pass speed *with* 2nd-order, added `run_ctu!`: the unsplit MUSCL-Hancock scheme with
the **transverse predictor** (each cell's 6 PLM face states are evolved dt/2 by the *full* all-direction
flux divergence — the transverse terms the fused kernel dropped). Generic over `@fvsystem`: `physflux_x`
rotates to y/z via `swap_y`/`swap_z`, reconstruction is per-component (rotation-free). **Validated
genuinely 2nd-order** (entropy-wave order 2.0–2.6, conservation machine-zero).

**The naive single-pass is compute-bound — measuring proved it.** `run_ctu!` runs **~2000 Mcell/s —
*slower* than RK2's ~2980**: with no shared memory, each output cell recomputes the full transverse
prediction of all 6 neighbors (~7× redundant arithmetic, 168 registers per ptxas), outweighing the one
global pass it saves. The hypothesis that the A6000 L2 would amortize the ~25-cell stencil was wrong —
the bottleneck is arithmetic, not bandwidth.

**The fix — shared-memory tiling + f16, the reference's own technique (`run_ctus!`).** Three phases per
block: (0) load the primitive tile + 2-cell halo into shared; (1) compute each cell's transverse `dU`
**once** into a shared `dU` tile (+1 halo); (2) each output cell builds its 6 predicted faces from the
shared tiles (cheap PLM recon on the fly + the stored `dU`), Riemann, update. ptxas: 72 regs, no spill,
but 35 KB shared → only **2 blocks/SM = 33% occupancy**. So tile in **f16** (the `.cu`'s f16-tile trick),
halving shared to ~18 KB → 3 blocks/SM (50%); regs rise to 80, still no spill. Conservation stays
*exact* — the interface flux is computed identically from both sides (same shared f16 values) and the
update is f32. Validated 2nd-order (order 2.0–2.6, `Δmass`~1e-7, matches the naive scheme to the f16
floor ~2e-4). Result: **~3300 Mcell/s = ~48% of the `.cu`, 1.6× over naive and 1.17× over pure-f32 RK2**
— now `evolve!`'s default (`scheme=:ctu`; `:rk2` keeps the pure-f32 path). `__launch_bounds__(256,4)` to
force 64 regs / 4 blocks made it *worse* (spills) — the "reduce structurally, never cap registers" lesson
re-confirmed.

**High-effort audit: 48% → 61%.** ptxas + a structural read found three wins. (1) **Flux-once**: phase 2
computed every interface flux *twice* (cell `i`'s `Fxp` is cell `i+1`'s `Fxm`). Restructured to compute
each interface **once** into a shared flux tile (`Fs`, +1 halo) and have each cell accumulate the
divergence — halving the Riemann/`predicted_face` work. This *also* dropped registers **80 → 48** (an
`acc[NV]` accumulator instead of holding 8 flux arrays live), removing the register pressure. → ~3750
Mcell/s (55%). (2) **One-sided reconstruction**: `predicted_face` computed both PLM faces but used one;
`recon_one` does just the needed side. (3) **NV-aware tile**: with registers now cheap (48), grow the
z-tile to **8³ for Euler** (NV≤5) to amortize the halo (Ws over-read 4.5×→3.4×) and lift occupancy to 2
blocks/SM = 66%; GLM (NV=9) keeps 8×8×4 (the transpiler emits `TBZ` per system, since 8³ at NV=9 needs
~98 KB shared). → **~4220 Mcell/s = 61% of the `.cu`, 2.1× over naive, 1.4× over RK2** — 2nd-order and
conservation preserved throughout. The remaining 61→91% is the reference's rolling z-stream march (holds
only a few z-slabs of shared per output cell, vs our whole-tile-in-shared), full-f16, and a leaner flux —
the "long campaign" structure, diminishing returns from one transpiled stencil.

## Roadmap (priority order)

1. ~~**SIMD-CPU backend**~~ ✅ done (single-core, ~5–7× over scalar, bit-identical). Next CPU
   increments: **threads + cache-blocking** (toward the `cpu_simd.jl` ~120 Mcell/s with NUMA), and
   `Vec{16}` on AVX-512 hosts.
2. ~~**CUDA backend**~~ ✅ done (1D, fused per-cell, bit-identical to CPU, ~11,400 Mcell/s). Open
   items: the staged shared-memory **cube** in multi-D; whether CUDA.jl plateaus below the `.cu`
   here (it didn't for this register-light fused 1D kernel) → the transpile-to-CUDA-C escape hatch
   only where needed; move CUDA to a **weakdep + package extension** so CPU-only installs stay light.
3. ~~**Multi-D + rotation**~~ ✅ 2D CPU (`Grid2D`) via Strang splitting; rotation bit-exact, 2nd order.
4. ~~**GLM-MHD via the contract**~~ ✅ 9 vars, two-vector rotation, Brio-Wu validated, ψ-damping
   `source` hook, **2D CUDA** backend (`Grid2DCU`, bit-identical to CPU), Orszag-Tang on GPU (256² to
   t=0.5 in **0.06 s** post-warmup; 512² in 0.34 s; stable, controlled div·B), and **dynamic `ch`** =
   global max fast speed each step (the `fastspeed_x` + `prestep` contract hooks; OT div·B 2.34 → 2.06),
   and an **HLLD** built-in (Miyoshi-Kusano, keyed to `GLMMHD`): branch-free, stable, positive,
   conservative, Bx-exact, rotation bit-exact. Open MHD items: CT for exact div·B.
   - *HLLD debugging notes (for future MHD solvers):* the L=R→physflux consistency check + bit-exact
     rotation isotropy localize bugs fast. Two real ones found: the transverse star formula is valid
     only when `dK = ρ(S−u)(S−Sₘ) − Bₓ² > 0` (else fall back to the un-rotated limit, not `1/dK`); and
     the star **energy** convective term is `(S−u)·E` (E is a density), NOT `(S−u)·ρ·E` — the spurious
     `ρ` blew up the low-density state. Guard `sqrt(ρ*)` (computed branch-free, even where unselected).
5. ~~**2D SIMD** backend~~ ✅ `Grid2DSoA`: SoA flat storage (x contiguous), vectorized along x for
   both sweeps (x-sweep shifted x-loads; y-sweep aligned x-blocks at rows j±1/j±2), reusing
   `_update_dir` + perm with `Vec{8}`. Bit-identical to scalar 2D (Euler/HLLC + GLM/HLLD), ~6.6× faster
   single-core.
6. ~~**3D**~~ ✅ `Grid3D` (scalar), **`Grid3DSoA` (SIMD)**, **`Grid3DCU` (CUDA)`**: symmetric Strang
   x·y·z·y·x; the z-sweep uses `dirperm(s,N,3)` — the rotation machinery generalized to all 3 axes with
   **no new code**. z-rotation isotropy bit-exact, 3D convergence 2nd order (2.19/2.14); SIMD & CUDA
   bit-identical to scalar (Euler/HLLC + GLM/HLLD). Throughput @128³: scalar 0.9, SIMD 6.7, CUDA 1703.
   Remaining: `Vec{16}` on AVX-512; **CT**; **Metal**.
7. ~~**CPU threads**~~ ✅ chunk-capped `Threads.@threads` over rows (2D) / z-planes (3D) in the SIMD
   backends; bit-identical at any thread count. **Peak ~145 Mcell/s at 16 threads** (12× over
   single-thread, beats the `cpu_simd.jl` 120 target). KEY FINDING: these kernels are
   **memory-bandwidth-bound**, so they peak at ~8–16 threads and **over-subscription hurts** (64→71,
   128→collapses on small grids). The chunk-cap (`≥ _MINROWS` lines/task) prevents the worst
   small-grid footgun, but the right knob is `JULIA_NUM_THREADS ≈ 8–16` — do NOT use `-t auto` (128)
   for these. **Cache-blocking** (sweep-fusion over tiles) is a further refinement, not yet needed
   since threading already exceeds the target; deferred.
8. ~~**`Vec{16}`**~~ ✅ measured — a *documented negative* on this host. The lane width `_W` is a single
   tunable const. On the AMD EPYC 7763 (Zen 3, **AVX2 only, no AVX-512**), `Vec{16}` is *worse* than
   `Vec{8}` (87–98 vs 111–145 Mcell/s) because it lowers to 2× emulated 256-bit ops + register
   pressure. Kept `_W = 8`; documented that `_W = 16` is for real AVX-512 hosts (Zen 4 / Intel
   Skylake-X+), which we can't test here.
9. ~~**CT** (constrained transport)~~ ✅ `Grid2DCT` (scalar, 2D, planar B): face-staggered B advanced
   by the curl of an edge EMF (built from the Godunov magnetic fluxes) → **div·B at machine zero**.
   Orszag-Tang: IC div·B = 0 exactly; after evolution max|div·B| = 2.9e-4 (Float32 roundoff × 1/dx)
   vs the GLM cleaning's ~2.0 — ~7000× smaller. Uses the GLMMHD physics with ch=0 as the ideal-MHD
   flux. Remaining CT: the `@staggered`/`@emf` CONTRACT seam (user-definable CT systems), PLM, 3D, GPU.
10. **Metal** — write the backend (analogous to CUDA); CANNOT be compiled/tested on this Linux/NVIDIA
    host (Metal.jl is macOS-only). Measure the Metal.jl gap on Apple hardware.
11. **Packaging:** move CUDA to a weakdep + extension (CPU-only installs stay light).

Conformance lives in the parent `GLMMHDTurb` repo (OT, Sod, turbulence, div·B, the gradient-IC
benchmarks): the spec is "reproduce that matrix from one physics source on every backend."
