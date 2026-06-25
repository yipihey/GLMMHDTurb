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

**2-block tuning (`spike_ct3.cu`): ~1500 Mcell/s.** Drop the 3 face-B slots from the prim tile and
read normal-B from global in the flux routine → shared 66→48 KB at OZ=3 → **2 blocks/SM**.

| variant | Mcell/s @480 | regs | blocks | note |
|---|---:|---|---|---|
| `spike_ct2.cu` store-full, faceB-in-tile | 1206 | 64 | 1 | any even N |
| **`spike_ct3.cu` faceB-from-global, OZ=3** | **1501** | 74 | **2** | needs N%3==0 |
| `spike_ct4.cu` mag-only + recompute hydro | 492 | 128 | 2 | recompute blows regs — negative |

The mag-only/recompute path is a trap (the cube>march lesson once more: 6 live recomputed fluxes →
128 regs). Staged 3D-tile CT (`spike_ct3.cu`) = 1501 Mcell/s @480.

### ★★ 2.5D z-STREAMING CT MARCH (`spike_ctm.cu`) — the NEW BEST: ~1560 Mcell/s @480

The streaming win *does* transfer to CT after all — once the periodic-z pipeline is solved.
**Each block owns a full z-column (no z block-seam), streaming z through a 5-plane prim ring +
3-plane mag-flux ring (f16); inline hydro, face-B updated with lag 2.** The blocker — plane 0's
EMF needs plane NZ−1's magnetic flux (computed last in a forward sweep) — is solved by **priming
the ring with the wrap planes** before the sweep (`loadp(-3..-1)` + computing `magflux(-1)`). That
priming *is* the "permanent periodic-copy buffer" idea: it converts the wrap-dependency into a
clean linear sweep, no extra global memory. (Two bugs found en route: missing priming → garbage
z-stencil; and the `magflux` loop guarded `L≥1` so it never computed the wrap plane's flux →
EX/EY read uninitialized → NaN. Both fixed.)

Tile sweep @480 (all 64–133 regs, div·B 7.6e-5 = Julia, validated tile-independent):

| tile (z-streamed) | shmem | blocks | Mcell/s |
|---|---|---|---:|
| 16×12 / 24×8 | 44–47 KB | 2 | **~1560** |
| 32×8 | 60 KB | 1 | 1398 |
| 16×16 | 54 KB | 1 | 1377 |
| 8×8 | 22 KB | 4* | 1115 (only 64 threads) |

### CT march — full PLM/PPM × ±species matrix (`spike_ctm.cu` flags `-DPPM`, `-DNSP=2`)

All measured @480³, best 2-block tile, validated (div·B 7–8e-5 machine-zero, species mass conserved
to 1e-7, **Σ X_i = 1 exact**, no overshoot):

| CT @480 (best 2-block tile) | PLM | PPM (lean parabolic-edge) |
|---|---:|---:|
| **no species** | **1575** (16×12) | **757** (24×8) |
| **+2 species (CMA)** | **1213** (16×10) | **565** (16×10) |

Findings: (1) species via CMA (ride the mass flux, store fraction in the f16 tile, conserved rho*X
global) cost −23% PLM (1575→1213) — entirely the +2-var tile crossing 50 KB → 1 block, *recovered* to
2 blocks with a compact 16×10 tile (`__launch_bounds__` at 160 threads also caps regs 168→128). Σ X_i
stays exactly 1 (both species ride the same mass flux). (2) The **lean parabolic-edge PPM stays at 2
blocks** (only +3–6 regs: 133→136) and is ~2× slower purely on *compute* (the parabola+monotonize per
var in the inline recompute) — it does NOT hit the GLM-PPM 1-block cliff (669), validating the
playbook's "parabolic-edge PPM is the viable MHD PPM". Further species compaction (uint16-log / f16
global, as in `spike_25d.cu -DU16SP`) would shrink the −23% further.

**Final hand-tuned CT: ~1575 Mcell/s @480 (2.5D march, PLM, no species)** — **9.75× the production CT (160)**,
**~81% of the GLM cube (1923)**, **~50% of the GLM march (3100)** — with **exact (machine-zero)
div·B** where GLM only cleans it. **Verdict: the production CT's ~10× deficit was entirely layout +
orchestration; a hand-tuned f16 CT is GLM-cube-class with a hard div·B=0. The streaming structure
DOES carry to CT** (the periodic-wrap pipeline just needs ring priming — credit: user's
ghost-buffer insight). Ceiling now set by 128–133 regs + dual-ring shared (2 blocks); H200 (more
regs/shared) is the next lever.

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

## Complete reference matrix — 3 solvers × PLM/PPM × ±2 species (@480³, A6000, gradient IC)

All measured on the **same gradient IC** (the `ph=0.001·mod(i,911)` pattern the standalone spikes use,
so the MonCen limiter does real work). NB a *uniform* IC lets MonCen early-return (`dl·dr≤0→0`) and
inflates throughput ~20–40% (worst for PPM) — the earlier CT numbers (1206–1575) were uniform-IC; the
representative gradient values below are the honest ones. Best 2-block tile per cell.

| solver | PLM no-sp | PLM +2sp | PPM no-sp | PPM +2sp | reference kernel |
|---|---:|---:|---:|---:|---|
| **Hydro** (5-var) | **6865** | 5082 | 3995 | 3822 | `spike_25d.cu` (`HANCOCK1D`/`PPM1D`, `SCALARS+CMA`) |
| **GLM-MHD** (9-var) | **3175** | 2733 | 2064 | 1410 | `spike_mhd.cu` (`-DPPM`, `-DNSP=2`) |
| **CT-MHD** (8-var) | **1255** | 1011 | 752 | 558 | `spike_ctm.cu` (`-DPPM`, `-DNSP=2`) |

Register counts (GLM, `cuobjdump -res-usage`): PLM 128, **PPM 128 (no inflation!)**, PLM+2sp 128,
PPM+2sp 177. **The lean parabolic-edge PPM does NOT inflate registers** — its ~1.5–1.7× penalty is
*pure compute* (parabola + monotonize per var), staying at the same occupancy. This is the decisive
contrast with the old **full-characteristic** MHD-PPM (201 regs → 1 block → 669, a 4.7× cliff): the
viable MHD/CT PPM is the parabolic-edge form. Species via CMA cost −12% (hydro) to −33% (GLM+PPM),
mostly the extra data dropping a block; Σ X_i = 1 exact, species mass conserved to ~1e-7, div·B
unaffected. Hydro PPM+sp barely costs over PPM (−4%: hydro PPM is compute-bound, the +2 vars don't
drop a block). All cells validated (Orszag-Tang / conservation).
