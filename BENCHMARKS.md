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
