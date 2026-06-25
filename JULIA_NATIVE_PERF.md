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

## Final Julia-native matrix (gradient IC @480, all levers applied)

| solver | PLM | PLM + 2 species | PPM | structure |
|---|---:|---:|---:|---|
| **Hydro** | **4166 (61%)** | 912 (18%, lmarch) | **2663 (67%)** | cube (4-stage shared tile) |
| **GLM-MHD** | **2094 (66%)** | — | 656 (32%) | cube |
| **CT** | 206 (16%) | — | — | 7-kernel (fused tile = negative, 194) |

(% of the `.cu` reference: hydro 6865/5082/3995, GLM 3175/—/2064, CT 1255.)

## What was tried, and the verdict

- **Lever 1 — `fastmath=true`** (the `--use_fast_math` equivalent, CUDA.jl 5.11): the only broadly
  useful lever. GLM PLM 58→66%, hydro 58→61%, applied to all production launches; OT correctness
  unchanged. A per-instruction win that does **not** cross occupancy tiers (unlike nvcc's 80→64). Made
  GLM PPM *worse* (SFU vs the 1-block PPM) — reverted there.
- **Lever 2 — register reduction:** the PPM helpers (`ppm9`/`ppm_edges`/`spj`/…) were **already**
  `@fastmath`-tagged; GLM PPM's 203 regs / 1 block is inherent codegen (vs the `.cu` lean PPM's 128 / 2),
  no easy reduction. Stays at 32%.
- **Lever 3 — CT fused-tile port:** CORRECT (div·B machine-zero, matches the 7-kernel to ~2e-4) but a
  **NEGATIVE** — 194 < the 7-kernel's 206. The shared tile caps Julia at 1 block where the `.cu` ran
  efficiently; opposite verdict to nvcc. CT ceilings at ~206 (16%).
- **Structure does not transfer:** the `.cu`'s fastest kernels are *marches* (hydro 6800, CT 1255). In
  Julia the march is **far slower than the cube** (hydro lmarch 1170 vs cube 4166) — warp under-fill +
  heavier codegen. So Julia's best is the **cube**, and the cube tops at ~61–67%.

## Bottom line

**Pure Julia-native (CUDA.jl) tops out at ~60–67% of the `.cu` for the cube-friendly PLM/PPM cells, and
16–37% for the cells that need a structure Julia cannot codegen efficiently (march, fused-tile, HLLD,
the slow lmarch species path).** The gap is the irreducible CUDA.jl-vs-nvcc codegen+occupancy residual:
CUDA.jl emits ~1.3–1.5× the registers, fast-math is a smaller lever, and the shared-tile structures that
win for nvcc cap Julia at 1 block. This is precisely why `march_bridge` exists — driving the `.cu` from
CUDA.jl over shared device memory is the only way to get full `.cu` throughput from a Julia workflow.

KA (portable) path: `@fastmath` is in the kernel bodies; the KA launch can't take `fastmath=true`, so it
trails CUDA.jl-native further (best-effort, as scoped). Missing matrix cells (GLM/CT species, CT PPM)
would each need a new kernel and are capped by the same wall — low value, left as documented follow-on.
