# Examples & validators

Runnable scripts used during development — physics demos, numerical validators, and benchmarks.
They are not part of the package API. Run from the package root, e.g.

```bash
julia -O3 --project=. examples/orszag_tang.jl
```

Most need a CUDA GPU (and `nvcc` for the transpile benchmarks); the CPU validators do not.

## Physics demos
- `orszag_tang.jl` — GLM-MHD Orszag-Tang vortex on the GPU; writes a density field, render with `plot_ot.py`.
- `ct2d.jl` — 2D constrained-transport MHD (machine-zero div·B).

## Numerical validators (correctness)
- `validate_2d.jl`, `validate_3d.jl` — rotation isotropy + convergence order in 2D/3D.
- `validate_3d_backends.jl`, `validate_simd2d.jl` — backend bit-identity (scalar ≡ SIMD ≡ CUDA).
- `validate_mhd.jl` — GLM-MHD (Brio-Wu) + rotation.
- `validate_hlld.jl` — the HLLD MHD Riemann solver.
- `validate_march.jl` — the transpile `Grid3DCuMarch` solver (convergence, conservation, scalar agreement).

## Benchmarks
- `bench_vs_cu.jl` — transpile-to-nvcc throughput vs the hand-tuned `.cu` reference.
- `bench_threads.jl` — CPU SIMD threading scaling.
- `compare_passes.jl` — same-process A/B for the pass-reduction study (avoids GPU thermal confounds).
