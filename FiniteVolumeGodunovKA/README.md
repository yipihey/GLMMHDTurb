# FiniteVolumeGodunovKA.jl

Write a finite-volume Godunov solver's *physics* once, in pure Julia, and run it fast on **CPU
and CUDA** (Metal a desirable bonus). You declare a **system** — conserved variables plus a
handful of pure, branch-free per-cell functions over a generic element type — and the library
owns all the structure: reconstruction, the Riemann combination, the conservative update,
tiling, occupancy tuning, and backend dispatch.

This is the KernelAbstractions *philosophy* (write-once, multi-backend) specialized to FV
Godunov, with the performant backends KA lacks for this domain (SIMD on CPU, transpile-to-native
on GPU — routing around the CUDA.jl codegen wall documented in `../JULIA_NATIVE_PERF.md`).

```julia
using FiniteVolumeGodunovKA
s  = Euler(γ = 1.4f0)
U0 = [prim2cons(s, (x < 0.5f0 ? 1f0 : 0.125f0, 0f0,0f0,0f0, x < 0.5f0 ? 1f0 : 0.1f0))
      for x in range(0.5f0/400, step = 1f0/400, length = 400)]
g  = Grid1D(s, U0; dx = 1f0/400, bc = :outflow, recon = PLM(), rsol = HLLC())
evolve!(g, 0.2f0)        # same call on every backend
```

**Status: v0.** The `@fvsystem` contract, library PLM + LLF/HLL/HLLC/HLLD, `Euler` and `GLMMHD`
systems, and backends spanning **1D/2D/3D × scalar/SIMD/CUDA**: `Grid1D`, `Grid1DSoA` (SIMD),
`Grid1DCU` (CUDA), `Grid2D`, `Grid2DSoA` (SIMD), `Grid2DCU` (CUDA), `Grid3D`. The same branch-free
physics source runs **bit-identically** on a CPU thread, a SIMD lane, and a GPU thread (max |Δ| = 0 vs
the scalar backend across HLLC/HLL/LLF/HLLD); rotation (1, 2, or 3 axes; momentum + B) is exact, and the
schemes are 2nd order in 1D/2D/3D.
Throughput on an A6000: CPU scalar ~9–14, CPU SIMD ~60 (single core), **CUDA ~11,400 Mcell/s**.
Convergence runs in Float64 from the same Float32 physics (2nd order). A **2D backend** (`Grid2D`,
Strang splitting) validates the rotation design: the y-flux — obtained by rotating the marked vector
components and calling the same `physflux_x` the user wrote — is **bit-identical** to the x-flux.

**GLM-MHD lives in the same contract.** Going from `Euler` (5 vars, momentum) to `GLMMHD` (9 vars,
momentum *and* B) was: add variables, add the `ch` param, write `physflux_x`, and declare `vidx` as two
vector triples — the library's rotation handled y/z automatically. Brio-Wu shock tube is stable and
preserves the normal field exactly; the two-vector rotation isotropy is bit-exact. GLM-MHD also has
a ψ-damping `source`, runs on the GPU (Orszag-Tang, `Grid2DCU`), and sets the cleaning speed `ch`
dynamically to the global max fast speed each step (via the `fastspeed_x`/`prestep` hooks). See
`DESIGN_fvkernel.md` for the contract and roadmap (next: 2D SIMD, HLLD, Metal/CT).

See `DESIGN_fvkernel.md` for the contract, the rotation/Riemann design wins, the three locked
decisions, and the roadmap. Run the tests with `julia --project=. -e 'using Pkg; Pkg.test()'`.
