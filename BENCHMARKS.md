# GLMMHDTurb — benchmark record

Hardware: RTX A6000 (sm_86, 768 GB/s, 48 GB). All GPU runs fp32 state + f16 shared tile
(f16 + f32-update), 192 threads/block, HLL Riemann unless noted. Throughput = N³·nsteps / wall.
BW% = throughput · 72 B/cell (MHD, 9 var) or 40 B/cell (hydro, 5 var) / 768 GB/s.

## Reconstruction: PPM vs PLM penalty (N=480, TB=6, f16, HLL)

Single-zone PPM (3-pt parabolic edges + CW84 monotonize, ±1 stencil → same tile as PLM).
Hydro = full 3-wave characteristic trace; MHD = lean parabolic-PPM (`mh+edge−m0`, shock→PLM fallback).

| solver | recon | regs | blocks/SM | Mcell/s | penalty vs PLM |
|---|---|---|---|---:|---:|
| Hydro (5-var Euler) | PLM (MonCen)        |  56 | 3 | 4250 | —     |
| Hydro (5-var Euler) | PPM (characteristic) |  95 | 3 | 2856 | 1.49× |
| GLM-MHD (9-var)     | PLM (MonCen)        | 144 | 2 | 1916 | —     |
| GLM-MHD (9-var)     | PPM (parabolic)     | 201 | 1 |  669 | 2.86× |

Penalty mechanism = register-budget occupancy cliff, not arithmetic. Hydro PLM (56 regs) keeps
3 blocks/SM even at PPM's 95 regs → 1.49× is pure math. MHD PLM (144 regs) is already near the
~170-reg ab2 ceiling, so PPM's 201 regs tips to 1 block/SM → 2.86× ≈ 2× occupancy × 1.4× math.

### MHD PPM — maxregs sweep (forcing occupancy makes it worse: spill > occupancy gained)

| maxregs | blocks/SM | Mcell/s | penalty |
|---|---|---:|---:|
| 0 (natural, 201) | 1 | 669 | 2.86× |
| 160 | 2 | 407 | 4.71× |
| 128 | 2 | 292 | 6.56× |
|  96 | 2 | 220 | 8.69× |

## Fastest solver: nvcc light line-march, driven from Julia (`march_bridge/`)

The throughput ceiling on this A6000. A hand-written CUDA 2.5D light line-march (transverse-free
1D-Hancock + HLL + Dedner GLM, f16 shared tile) compiled with `--use_fast_math` and driven from
CUDA.jl over shared device memory — nvcc's codegen with the Julia test rig. See `march_bridge/`.

| solver | config | regs | blocks/SM | N=480 Mcell/s | vs cube |
|---|---|---|---|---:|---:|
| **GLM-MHD march** | HLL, f16, fast-math | 128 | 2 | **~3100** | 1.6× (`step_plm!` 1923) |
| **Hydro march**   | HLL, f16, fast-math |  64 | 4 | **~6800** | — (5.9× Julia-native lmarch 1153) |

Zero bridge overhead (driven ≥ standalone; timed loop is pure kernel launches). `--use_fast_math`
is load-bearing: it frees the registers (hydro 80→64 → the 4-block tile; not a driver/ptxas-version
effect — confirmed across ptxas 11.8/12.9/13.3). MHD is register+shared-bound at 2 blocks (38 KB f16
tile mandatory: fp32 9-var = 76 KB > 48 KB) and only ~29% of peak BW, so it is the A6000 ceiling for
the scheme; H200 (more regs, 228 KB shared) would reach 3+ blocks.

Correctness of the fastest MHD path: matched-IC Orszag-Tang vs the cube agrees to ~1% in all physical
fields (the slowly-accumulating transverse term), vrms to 0.2%; div·B drift ~5e-5, GLM cleaning active
(`ot_validate.jl`). Turbulent driving stable to Mach~6 at CFL≤0.4, but with ~10× looser ∇·B than the
cube — the transverse-free scheme's tradeoff (`turb_robustness.jl`). So ~3100 is the fastest
correct-to-1% MHD throughput; the cube (~1920) remains the ∇·B-tight reference.

## CT-MHD in the prototype (`ct_mhd.jl`) — div·B = 0 by construction

Lean constrained transport: 8 reals/cell (5 cell-centered conserved + 3 face-staggered B; no ψ,
no redundant 6-component face storage → fits 512³ where production CT OOMs at 22/cell). The
cell-centered Godunov is the GLM light-march machinery verbatim (PLM MonCen + transverse-free
1D-Hancock + HLL, fp32); only the induction step is new — face-B updated by curl of Balsara-Spicer
edge-EMFs. Validated on Orszag-Tang N=128: initial div·B **exactly 0** (face-B from a vector
potential), vrms 0.78 @t=0.21 (matches the GLM cube's 0.79).

| solver | form | N=192/256 Mcell/s | div·B | notes |
|---|---|---:|---|---|
| CT-MHD `ct_mhd.jl` | Julia, 7-kernel, fp32 | **205** | ~1e-4 (fp32 floor, bounded) | > production CT ~160 already |
| (production CT, mini-ramses) | CUDA fused, full-step | ~160 (256³) | machine-0 (f64 EMF) | OOMs at 512³ |

### Fused `.cu` CT (`march_bridge/cu/spike_ct.cu`) — correct, but the fusion is register-bound

Built the single-kernel fused CT (load prim+faceB tile once; each owned thread updates hydro from
6 face HLL fluxes AND its 3 face-B from curl of edge-EMFs, recomputing the magnetic fluxes from the
tile — recompute-determinism ⇒ div·B preserved at block seams with no shared flux store). Validated
vs `ct_mhd.jl` on OT N=128: **div·B 7.4e-5 = the Julia 8.4e-5** (machine-zero, fp32 floor, across
block boundaries), fields agree to ~5% (fast-math + predictor).

| CT variant | form | Mcell/s | regs/blk | div·B |
|---|---|---:|---|---|
| `ct_mhd.jl` | Julia, 7-kernel | **205** | — | ~1e-4 |
| `spike_ct.cu` | .cu fused recompute | **61** | 255 / 1 blk | ~7e-5 |

The fused-*recompute* CT is register-bound (255 regs → 1 block) because each thread recomputes
~20 Riemann solves (every edge-EMF re-derives 4 face fluxes) — the GLM cube>march lesson. The fix
is **staging**, below.

### ★ STAGED f16 CT (`spike_ct2.cu`) — the hand-tuned A6000 rewrite: ~1200 Mcell/s

Compute each face flux **once** into a shared **f16** flux tile (phase A); then hydro update + edge
EMF + face-B all **read** it (phase B). Two keys: (1) staging drops registers 255→**64** (the sweet
spot) — each phase has tiny live state; (2) the update base (hydro U *and* face-B) is read **f32
from global**, the f16 tile is reconstruction-input only — this is the GLM "f16-tile + f32-update"
lesson, and it's what keeps div·B machine-zero (f16-rounding the face-B base broke div·B → 1.15;
f32 base → 9e-5). fast-math holds it at 64 regs. NG=3 halo (CT's EMF reaches one cell past hydro).

| CT variant | Mcell/s @480 | regs | div·B | vs production CT (160) |
|---|---:|---|---|---:|
| fused recompute (`spike_ct.cu`) | 61 | 255 / 1 blk | 7e-5 | 0.4× |
| Julia multi-kernel (`ct_mhd.jl`) | 205 | — | 1e-4 | 1.3× |
| **staged f16 (`spike_ct2.cu`)** | **1206** | **64 / 1 blk** | **9e-5 = machine-0** | **7.5×** |

Validated on OT N=128: div·B 9.3e-5 = the Julia 8.4e-5; fields match Julia to ~4% (f16 tile).
Tile sweep: 8×8×4 wins (1294 @288); all 64 regs, shared-limited to **1 block** (66 KB > the 50 KB
for 2 blocks). **Revised conclusion (my "~300–700, 5× gap intrinsic" was WRONG):** staged f16 CT is
**~63% of the GLM cube (1923) and ~39% of the GLM march (3100)** — and it gives **exact
(machine-zero) div·B** where GLM only cleans it. The production CT's 8× deficit was layout +
orchestration, NOT the scheme. Headroom: dropping to 2 blocks (shared < 50 KB via face-B-from-global
or mag-only flux storage) should reach ~1800–2000.

## Production solver throughput (prior sessions, for context)

| solver | config | N=480 pure | N=480 +turb | N=512 (gate) |
|---|---|---:|---:|---:|
| GLM-MHD `step_tiled!` | HLLD, f32, TB4 (bit-repro) | 1139 | ~900 | 1124–1141 |
| GLM-MHD `step_plm!`   | HLL, f16, TB6 (default)    | 1916–1923 | 1416 | — |
| Hydro `step_hydro!`   | HLL, f16, TB6 (default)    | 4250 | — | — |

Memory gate: 72 B/cell (two fp32 (N,N,N,9) buffers, no scratch). Throughput gate ≥750 Mcell/s @512³ — met.

## Validation (correctness, not throughput)

| test | solver | result |
|---|---|---|
| Sod (N=120)        | Hydro PPM | mass-exact 972000→972000, ρ>0, ρmax 0.997 (sharper contact than PLM, no overshoot) |
| Orszag-Tang (N=120)| MHD PPM   | vrms 0.7910 (= PLM-HLL), div·B 1.16 (HLL 0.96 < this < HLLD 3.59), ρmin 0.0835, finite |
| Orszag-Tang        | MHD PLM/HLLD | bit-identical across 3 integrator paths, vrms 0.7887 |

PPM is wired into the module as an opt-in: `step_plm!(...; recon=:ppm)` / `step_hydro!(...; recon=:ppm)`
(f16+HLL path only; PLM stays the default). Numbers above reproduced through the public API.
