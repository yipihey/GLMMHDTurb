# Julia-native GPU performance — closing the gap to the hand-written `.cu` (branch `julia-native-perf`)

**Goal.** Make the pure-Julia kernels (CUDA.jl-native + KernelAbstractions in `glmmhd_turb.jl` /
`ct_mhd.jl`) come as close as possible to the `march_bridge/cu/spike_*.cu` reference throughput, using
**only Julia code** (no `.cu`). CUDA.jl-native is the parity vehicle; KA is best-effort. All numbers on
the **gradient IC** (`ph=0.001·mod(i,911)`, matching the `.cu` matrix — a uniform IC inflates ~20–40%
via MonCen early-return) @480³ on the A6000.

## Baseline (before tuning) vs the `.cu` targets

| solver / recon | Julia Mcell/s | Julia regs | `.cu` target | Julia % | `.cu` regs |
|---|---:|---:|---:|---:|---:|
| Hydro PLM (`integrator_hydro!` cube) | 3972 | 56 (3 blk) | 6865 (march) | 58% | 64 (4 blk) |
| Hydro PPM (`integrator_hydro_ppm!`) | 2575 | 95 (3 blk) | 3995 | 64% | ~95 |
| GLM PLM (`integrator_plm!`) | 1837 | 144 (2 blk) | 3175 | 58% | 128 (2 blk) |
| GLM PPM (`integrator_plm_ppm!`) | 656 | 203 (1 blk) | 2064 | 32% | 128 (2 blk) |
| CT (`ct_mhd.jl` 7-kernel) | 205 | — | 1255 (fused march) | 16% | 64–133 |

Diagnosis: GLM PLM and hydro PLM are at the **same occupancy tier** as the `.cu` (2 / 3–4 blocks), so
the ~58% gap is **per-instruction codegen** (CUDA.jl emits more/slower ops, no `--use_fast_math` SFU
approximations). GLM PPM is the codegen **occupancy** gap (203 regs → 1 block vs `.cu` 128 → 2). CT is
**structural** — the Julia is a 7-kernel multi-kernel whose global flux-array round-trips (Fx/Fy/Fz are
(N,N,N,8) each) are the 205 ceiling, not codegen.

## Lever 1 — CUDA.jl `fastmath=true` (the `--use_fast_math` equivalent) ✅

CUDA.jl 5.11's `@cuda fastmath=true` enables LLVM fast-math + the SFU approximate `rsqrt`/recip — the
global flag the per-expression `@fastmath` macro does not provide. Applied to the production
`integrator_plm!` / `integrator_hydro!` / `integrator_hydro_ppm!` launches:

| kernel | before | after | regs (before→after) |
|---|---:|---:|---|
| **GLM PLM** | 1837 | **2068** (58%→**65%**) | 144 → 121 |
| Hydro PLM | 3972 | 4118 (58%→60%) | 56 → 64 |
| Hydro PPM | 2575 | 2633 | 95 → 95 |

Correctness unchanged (Orszag-Tang vrms 0.7903, div·B cleaned). **NOT** applied to GLM PPM — it
*regressed* 656→556 there (the SFU change interacts badly with the 201-reg/1-block PPM); that cell's fix
is structural register reduction (Lever 2). Note: unlike nvcc (where fast-math crossed an occupancy tier,
80→64 regs), in CUDA.jl it's a modest per-instruction win that does **not** cross tiers.

## Next steps (prioritized)

1. **CT structural port (biggest gap, 205→target).** Port the `.cu` fused staged f16 CT tile
   (`spike_ct2.cu` structure: load prim+faceB f16 tile → phase-A flux → shared → phase-B hydro update +
   edge-EMF + face-B, f32 update base) into a single CUDA.jl `@cuda` kernel, reusing the `ct_mhd.jl`
   NTuple helpers (`cellprim`/`dflux`/`hll`/`hanc1d`). Eliminates the global flux round-trips. Expect
   205 → several hundred+ (structure dominates; codegen caps the rest). Validate div·B machine-zero vs
   the 7-kernel.
2. **GLM PPM register reduction (203→toward 128, 1→2 blocks).** @fastmath-coverage audit on `ppm9`/
   `ppm_edges`/`spj`; reduce live state; the goal is 2 blocks.
3. **Hydro structural** — the `.cu` hydro is a *march* (4 blocks); the Julia cube is 3 blocks. Either
   lever the cube further or fix the Julia light march (`integrator_hydro_lmarch!`, 1153, warp
   under-fill) toward a full z-stream.
4. **Fill the matrix** — species (CMA) + PPM for each solver; KA best-effort.

## Honest expectation

CUDA.jl register allocation is structurally ~1.3–1.5× nvcc's and fast-math is a smaller lever in Julia,
so exact parity is unlikely. Realistic landing: hydro/GLM PLM ~65% → ~75–85%; CT the big mover
(structural) from 16% toward ~50–70%.
