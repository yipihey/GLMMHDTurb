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

The 205 is **traffic-bound** (7 global passes, fluxes spilled to DRAM) — *not* the ceiling. The
fused f16-tile path (flux tile in shared, EMF computed in-tile, fast-math) is exactly what lifted
GLM from multi-kernel to the ~3100 march; the same applies here. **Finding: a lean fused f16 CT
should be GLM-march-class, not the 5× slower the production CT showed — that 5× was layout +
orchestration (22-var, multi-pass), not the scheme.** div·B is fp32-roundoff-bounded (~1e-3 over
2000 steps); f64 EMF accumulation pins it to machine-0 (as the production CT does). Next step: the
fused `.cu` CT kernel via the march_bridge.

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
