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

**Status: v0.** The `@fvsystem` contract, library PLM + LLF/HLL/HLLC, a reference CPU (scalar, 1D)
backend, and a **SIMD CPU backend** (`Grid1DSoA`, vectorized along the grid, `Vec{8,Float32}`
lanes) — with the `Euler` system defined through the contract. Sod and 2nd-order entropy-wave
convergence pass; convergence runs in Float64 from the same Float32-authored physics (element-type
genericity). The SIMD backend is **bit-identical** to the scalar one and runs **~5–7× faster**
single-core. See the roadmap in `DESIGN_fvkernel.md` (next: threads/cache-blocking, then CUDA).

See `DESIGN_fvkernel.md` for the contract, the rotation/Riemann design wins, the three locked
decisions, and the roadmap. Run the tests with `julia --project=. -e 'using Pkg; Pkg.test()'`.
