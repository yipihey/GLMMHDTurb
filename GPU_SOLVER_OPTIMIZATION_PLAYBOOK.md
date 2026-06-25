# GPU finite-volume MHD/hydro solver optimization — a playbook for future agents

**Audience.** An agent (or human) asked to make a cell-centered finite-volume hydro / MHD
solver as fast as possible on a given GPU, possibly on hardware different from the RTX A6000
this work targeted. This document is the *methodology* plus the worked A6000 results, so you
can re-derive the optimal kernel for your hardware rather than copy numbers that won't transfer.

**The three reference solvers** (all in `march_bridge/`, driven from Julia via `MarchBridge.jl`,
cross-validated against the fp32 Julia prototypes `glmmhd_turb.jl` / `ct_mhd.jl`):

| solver | scheme | best @480³, A6000, PLM, no species | reference kernel |
|---|---|---:|---|
| **Hydro** (5-var Euler) | transverse-free 1D-Hancock + HLL | **~6800 Mcell/s** | `cu/spike_25d.cu -DHANCOCK1D` |
| **GLM-MHD** (9-var, Dedner) | transverse-free 1D-Hancock + HLL + GLM | **~3100 Mcell/s** | `cu/spike_mhd.cu` |
| **CT-MHD** (8-var, constrained transport) | 2.5D z-march + HLL + edge-EMF | **~1560 Mcell/s** | `cu/spike_ctm.cu` |

All three keep state fp32 in global memory, reconstruct from an **f16 shared tile**, build with
`--use_fast_math`, and are bit-validated to ~1–5% against a fp32 reference (Orszag-Tang / driven
turbulence). CT holds div·B at machine-zero (fp32 floor); GLM cleans div·B to a small bounded level.

> **The single most important idea in this document:** on this class of kernel the limiter is almost
> never raw FLOPs or even raw bandwidth — it is the **occupancy cliff** set by registers and shared
> memory per SM. Every win below is ultimately a way to land the kernel on a higher occupancy tier
> (more thread-blocks resident per SM) without losing correctness. Optimize the cliff, not the math.

---

## 1. Characterize the GPU first (the occupancy-cliff model)

Before touching the kernel, write down for your GPU (values shown for the **A6000, sm_86**):

| quantity | A6000 | why it matters |
|---|---|---|
| 32-bit registers / SM | 65536 | sets `blocks = floor(65536 / (threads × regs/thread))` |
| shared memory / SM | 100 KB dynamic (48 KB static opt-out) | sets `blocks = floor(shmem_SM / shmem/block)` |
| max threads / SM | 1536 | sets `blocks = floor(1536 / threads/block)` |
| max blocks / SM | 16 | rarely binding |
| DRAM bandwidth | ~768 GB/s | sets the roofline ceiling |
| fp64 : fp32 throughput | 1 : 64 | makes fp64 accumulation expensive → drives the f16/block-FP tricks |

**Resident blocks/SM = the MINIMUM of the three `floor(...)` above.** Throughput scales roughly with
(resident warps) until latency is hidden, so crossing from N to N+1 blocks is a step change. Build
the **cliff table** for your launch (here, threads = 256):

| regs/thread | blocks/SM (A6000, 256 thr) |
|---|---|
| ≤ 64 | 4 |
| 65–85 | 3 |
| 86–128 | 2 |
| 129–255 | 1 |

The exact thresholds move with thread count: `regs_max(B) = floor(65536 / (256 × B))`. **Recompute
this table for your GPU and thread count — it governs every decision below.** On an H200 (more
registers and 228 KB shared) the thresholds are looser, so the same kernel lands a tier or two higher.

---

## 2. Roofline the kernel (classify the bottleneck)

Compute two numbers, then classify:

1. **Bandwidth ceiling** = `peak_BW / essential_traffic`. Essential traffic = `NV × bytes_per_var ×
   2 (load+store)` per cell. fp32 global: hydro 40 B/cell → 19.2 Gcell/s; GLM 72 → 10.7; CT 64 → 12.0.
2. **Memfloor (measured)** = run the kernel with the flux/Riemann compute **#ifdef'd out** (load tile,
   do the trivial `u_new = u_old`, store). This is the load/store + barrier + tile-traffic floor with
   zero physics. (`-DMEMFLOOR` in the spikes.)

Then:
- throughput ≈ memfloor and memfloor ≈ BW-ceiling → **bandwidth-bound** → attack *traffic* (compact storage).
- throughput ≈ memfloor but memfloor ≪ BW-ceiling → **memfloor/structure-bound** → attack the tile load + barriers.
- throughput ≪ memfloor → **compute/occupancy-bound** → attack *registers and occupancy* (the usual case for MHD/CT).

A6000 readings: hydro 6800 vs memfloor ~7650 vs BW 19200 → ~89% of memfloor, **near its structural
ceiling**. GLM 3100 at **29%** of BW, CT 1560 at **13%** of BW → both deeply **occupancy-bound** (the
EMF/MHD flux is register-heavy). This is why "make GLM/CT use less bandwidth" does almost nothing and
"make GLM/CT use fewer registers" is everything.

> Pitfall: a *stale* memfloor binary (e.g. an earlier flag combination that failed to relink and left
> an old full binary) reads as "compute is free / memfloor ≈ full". Rebuild from the lowest changed
> object/module **down**, and verify register/shared usage with `cuobjdump -res-usage` /
> `cudaFuncGetAttributes` — never trust a number you didn't relink for.

---

## 3. Strategy catalog (what worked, why, when) — ranked by impact

**S1. The transverse-free light scheme (biggest structural win).** A faithful unsplit MUSCL-Hancock
godunov holds the entire per-cell pipeline live (3 directional slopes + transverse predictor + 6
Riemann working sets) → **227 regs → 1 block**. Replacing the *transverse* Hancock predictor with a
*normal-only 1-D Hancock half-step* (uses only the direction's own slope; still 2nd order in space
**and** time) collapses that to **64 regs → 4 blocks** for hydro. Validated: matches the full-transverse
scheme to ~1% on Orszag-Tang (the dropped term accumulates slowly) and is stable to CFL 1.0. *When:*
always try first for a light kernel; it is the difference between register-bound and memfloor-bound.
*Caveat:* the transverse term is what tightens ∇·B and positivity under strong MHD turbulence — the
light scheme drives ρ to the floor / lets ∇·B grow ~10× more than the cube under hard driving. Use the
light scheme for throughput/exploration; keep a transverse/PPM variant for ∇·B-critical science.

**S2. f16 shared tile + f32 update base.** Store the reconstruction tile (prims, face-B, fluxes) in
**Float16** — it halves shared memory, which is often what lets the tile fit at 2–4 blocks, *and*
halves shared traffic. But: the **conservative update base must be read fp32 from global**
(`u_new[g] = u_old_global_fp32[g] + dt·divF`), with the f16 tile used *only* as reconstruction input.
f16-rounding the *old state* before adding the flux divergence **breaks conservation and (for CT)
breaks div·B** — we measured div·B jump from 9e-5 to **1.15** purely from reading the old face-B out
of the f16 tile instead of fp32 global. *When:* any tiled kernel; it is nearly free accuracy-wise if
you respect the f32-update-base rule (~1–5% field error vs fp32 from the reconstruction inputs).

**S3. Staging (compute-once-to-shared), not recompute.** When neighboring threads need each other's
fluxes (CT's edge-EMF reads 4 face fluxes per edge; the hydro update reads 6 face fluxes per cell), the
naive fused kernel **recomputes** them → ~20 Riemann solves/cell live → **255 regs → 1 block → slow**.
Instead **stage**: phase A computes each face flux once into a shared flux tile; phase B reads it for
the update/EMF. Each phase has tiny live state → **64 regs**. This single change took fused CT from
**61 → 1206 Mcell/s**. *When:* any kernel with cross-thread flux reuse. The staged "cube" structure is
register-optimal for *heavy* kernels; the fused march only wins for *light* kernels (S1).

**S4. `--use_fast_math` (free register relief).** Approximate reciprocal/sqrt in the HLL wave-speed and
`1/ρ` paths free ~16 registers — enough to cross a whole occupancy tier (hydro **80 regs/3 blocks/5300
→ 64/4/6800**). This is **not** a driver or compiler-version effect (we confirmed identical register
counts across ptxas 11.8/12.9/13.3); it is the flag. Accuracy cost is ~1 ULP, far below the scheme's
truncation error. *When:* always, for these solvers. Validate conservation once to confirm.

**S5. The 2.5D z-streaming march (eliminates the z-halo).** A 3-D tile with NG ghost cells reprocesses
the z-halo for every block (an 8³-owned, NG=3 tile is ~20% volume-efficient). Streaming z through a
rolling plane ring makes each plane's flux computed ~once with no z-halo redundancy, at good thread
counts. *When:* the z-direction is uniform/periodic and the tile is halo-dominated. **Periodicity
subtlety:** if the scheme couples planes (CT's EMF does; pure hydro does not), a forward sweep hits a
wrap dependency — plane 0 needs the last plane's flux. **Fix: prime the ring with the wrap planes
before the sweep** (load planes −3..−1 = NZ−3..NZ−1, and *compute their fluxes too*). This converts the
wrap into a clean linear sweep with no extra global memory. This is what made CT's march work (1560,
the current best CT) after it first appeared not to transfer.

**S6. Compact global storage (for bandwidth-bound kernels).** Halve global traffic with f16 state, or
uint16-log for quantities spanning many decades (chemical species over 30 dex: uint16-log10 gives
~0.1%/ULP). Conservation over a huge dynamic range is preserved with **block-floating-point**: store an
fp32 per-block mean/scale + f16 local deviations (multiplicative rescale for positive ρ/E; additive
deficit-distribution for signed momentum; Kahan/f64 at the cheap cross-thread reduction). *When:*
bandwidth-bound (hydro, species-heavy). Modest for compute-bound GLM/CT (f16 state gave hydro only +11%
→ it is half compute-bound even at 4 blocks).

**S7. CMA species (passive scalars ride the mass flux).** Advect species as `F_species = F_mass ·
X_upwind` instead of a Riemann solve per species — conserves Σ X_i exactly, compute is count-
independent (nearly free). Store the major species linear (exact Σ=1 via *derive-last*: store K−1, set
the Kth = 1−Σ); use uint16-log only for *isolated trace* tracers (log reconstruction overshoots at a
sharp trace-vs-major front — a known-hard 2nd-order multispecies problem). *When:* a handful of advected
species/colors. The cost is almost entirely the **extra data → occupancy drop**, not compute.

---

## 4. Trap catalog (symptom → cause → fix)

| symptom | cause | fix |
|---|---|---|
| 255 regs, 1 block, slow fused kernel | recompute of neighbor fluxes (every edge/cell re-derives) | **stage** to shared (S3) |
| forcing `maxregs` down → *slower* | spill to local memory > occupancy gained | never cap regs; restructure instead |
| heavy fused godunov = 1 block | full pipeline live (227 regs) | light scheme (S1) or stage (S3) |
| div·B 1.15 / mass drifts with f16 tile | f16-rounded the **update base** | read update base fp32 from global (S2) |
| NaN at boundary planes in a z-march | periodic wrap dependency unprimed | prime the ring with wrap planes **and compute their fluxes** (S5) |
| "compute is free / memfloor ≈ full" | stale/mis-relinked memfloor binary | rebuild bottom-up; verify with `cuobjdump -res-usage` |
| 80 regs/3 blocks where 64/4 expected | missing `--use_fast_math` | add it (S4); it's the flag, not the toolchain |
| Julia kernel 5× slower than the C twin | CUDA.jl emits ~1.5× the registers of nvcc | use Julia for algorithm/accuracy; drive the nvcc kernel over shared device memory (the `march_bridge` pattern) |
| recompute "to save shared" backfires | 6 live recomputed fluxes blow regs to 128 (the mag-only CT variant) | store-full in shared beats recompute here |

---

## 5. Headroom think-through (which solvers can still improve, and how)

Target matrix, @480³ A6000, Mcell/s. **✓ = measured this work; ~ = projected** (with the lever):

| solver | PLM, no species | PLM, +2 species | PPM, no species | PPM, +2 species |
|---|---:|---:|---:|---:|
| **Hydro** | **6800 ✓** | ~5300 ✓(U16SP) | ~2900 ✓(cube-PPM) | ~2400 ~ |
| **GLM-MHD** | **3100 ✓** | ~2300 ~ | 669 ✓ | ~550 ~ |
| **CT-MHD** | **1575 ✓** | **1213 ✓** | **757 ✓** | **565 ✓** |

**Hydro — near its ceiling; smallest headroom.** At 4 blocks / 64 regs / 89% of memfloor it is
structurally optimal for PLM. The only PLM lever is compact global storage (f16 state measured +11% →
~7400), which confirms it is *half* compute-bound even at 4 blocks. **Best realistic PLM hydro ~7400.**
PPM keeps 3 blocks (56→95 regs) so its ~1.49× penalty is pure arithmetic (full characteristic trace);
a *lighter* parabolic-only PPM (spatial parabola + the existing 1-D Hancock predictor, no characteristic
projection) should cut that toward ~1.2× → ~5600. Species: the −22% (U16SP) is the 7-var tile dropping
4→3 blocks; *derive-last linear* major species + uint16-log trace minimizes the footprint.

**GLM-MHD — occupancy-stuck at 2 blocks on A6000; modest headroom.** 128 regs and a 38 KB f16 tile both
pin it at 2 blocks; getting to 3 needs ≤85 regs *and* ≤33 KB, both very hard for a 9-var flux + GLM.
f16 global barely helps (29% of BW). **Best realistic A6000 PLM GLM ~3100–3400** (register micro-opt).
PPM is brutal (201 regs → 1 block → 669, a 2.86× cliff): viable only as the lean parabolic-PPM, and even
that wants 2 blocks it cannot get on A6000. **GLM's real lever is hardware** — on an H200 the 2→3 block
relaxation alone is ~1.5×, and PPM becomes affordable.

**CT-MHD — newest, the most headroom (~+20–30% on A6000).** 1560 at 2 blocks / 128–133 regs. Two
untried levers: (a) a **full-flux ring** (store hydro+mag flux per plane → the update reads instead of
the current lag-recompute of lower-face fluxes; trades shared for compute — worth measuring at OX≈12–16
to stay at 2 blocks); (b) **register reduction to 3 blocks** (≤85 regs) by splitting the EMF into a
second light pass or shrinking live state. **Best realistic A6000 PLM CT ~1800–2000.** PPM and species
for CT are now **built and measured** (`-DPPM`, `-DNSP=2` on `spike_ctm.cu`): species via CMA cost −23%
(1575→1213, the +2-var tile; recovered to 2 blocks with a compact 16×10 tile — Σ X_i=1 *exact*, div·B
unaffected, species mass conserved to 1e-7); the **lean parabolic-edge PPM stays at 2 blocks** (only
+3–6 regs) and is ~2× slower purely on *compute* (757), NOT an occupancy cliff — confirming the playbook
prediction that a parabolic-edge PPM avoids the GLM-PPM 1-block collapse. Further species compaction
(uint16-log / f16 global) would shrink the tile cost below the −23%.

**Cross-cutting:** all three live on the same cliff. Hydro is light enough for 4 blocks (so it is
near-BW); GLM and CT are heavy enough to be stuck at 2 (so they are register-bound). The species and
PPM axes both act by *adding live state/shared* → dropping an occupancy tier; the mitigations are always
"add the minimum data (compact storage, CMA, derive-last) and the minimum live state (lean PPM, stage)."

---

## 6. Porting guide — how the optimal config changes on new hardware

Recompute the cliff table (§1) for the target, then:

- **More registers/SM** (H200, MI300): the GLM/CT 2-block cliff relaxes → 3+ blocks → ~1.5× *for free*,
  and the light scheme (S1) matters less (you have register headroom for the transverse term / PPM).
  Re-evaluate whether the *faithful* transverse scheme (better ∇·B) is now affordable.
- **More shared/SM** (H200: 228 KB vs 100 KB): bigger tiles, the **f16-tile mandate relaxes** (fp32
  9-var tiles may fit → drop the f16 reconstruction error), and `nsubgrid=3` (lower halo fraction)
  becomes easy. Re-sweep tile size against the new shared budget.
- **Better fp64** (1:2 on data-center parts): fp64 accumulation/reductions are cheap → the f16/block-FP
  conservation gymnastics (S6) are less necessary; you may keep state fp32/fp64 and skip the tricks.
- **Lower BW-per-FLOP**: the compute-bound solvers (GLM/CT) get *relatively* better; traffic tricks
  (S6) matter less, occupancy still rules.
- **Different warp/threadblock granularity** (AMD: 64-wide wavefronts): re-derive thread counts so the
  tile is a whole number of wavefronts; the cliff math is identical in form.

**The invariant across all hardware:** roofline → classify → pick the structure (light scheme + staging
+ f16-tile-or-not) that lands the best occupancy tier → tune the tile to the binding constraint → apply
fast-math/compact-storage → validate. The *numbers* change; the *method* does not.

---

## 7. Reference-implementation index

CUDA-C spikes (`mini-ramses-metal/gpu/`, vendored into `march_bridge/cu/`), built with
`-O3 -arch=sm_86 --use_fast_math -DAS_LIB`, driven via `MarchBridge.jl` (device-pointer ABI; build
scripts `march_bridge/build_{march,mhd,ct}.sh`):

- **`spike_25d.cu`** — hydro 2.5D march. Default `-DHANCOCK1D` (light scheme). Flags:
  `HANCOCK`/`HANCOCK1D`/`PPM1D` (recon), `DONOR` (1-ghost donor seam), `LLF_FALLBACK`, `SCALARS`+`CMA`+
  `U16SP` (2 species), `F16STATE`/`U16STATE` (compact global), `MEMFLOOR`, `OX/OY/THREADS`.
- **`spike_mhd.cu`** — GLM-MHD 9-var march. `march_set_glm(ch,fac)`, `march_set_gamma`. f16 tile
  mandatory (fp32 9-var > 48 KB). PLM only currently.
- **`spike_ctm.cu`** — **CT-MHD 2.5D z-march, the fastest CT (~1560).** Prim ring (5) + mag-flux ring
  (3), inline hydro, lag-2 face-B, ring-priming for periodicity. Tile `OX×OY` (best 16×12 / 24×8).
- **`spike_ct2.cu`** — CT staged 3D tile, store-full flux, face-B-in-tile (1206, any even N).
- **`spike_ct3.cu`** — CT staged 3D tile, face-B-from-global, OZ=3 → 2 blocks (1501, needs N%3==0).
- **`spike_ct.cu`** (fused recompute, 61) and **`spike_ct4.cu`** (mag-only recompute, 492) — **kept as
  documented negatives** (the recompute register-blowup trap).

Julia prototypes (algorithm + accuracy reference; codegen-bound, ~1.5× the regs of nvcc → run the
spikes for throughput): `glmmhd_turb.jl` (`step_plm!`/`step_hydro!` with `recon=:ppm`,
`step_hydro_lmarch_sp!` = light hydro + 2 CMA species, `run_turb`/`run_ot`), `ct_mhd.jl` (`step!`,
lean 8-var CT, the div·B=0 ground truth). Numbers and the full derivation log: `BENCHMARKS.md`.

---

## 8. Validation checklist (do not ship an unvalidated fast kernel)

1. **Conservation**: mass/energy drift per step (f16-tile kernels ~1e-4; flag if larger → check the
   f32-update-base rule, S2).
2. **div·B** (MHD): GLM cleaned + bounded; CT machine-zero (fp32 floor ~1e-4, *bounded/saturating* — a
   real telescoping bug grows it unboundedly). Measure ∇·B with the **upper-minus-lower face stagger**
   (`(b[i+1]−b[i])/dx`), not lower-minus-previous.
3. **A reference test to ~1%**: Orszag-Tang (MHD), Sod (hydro), or driven turbulence (vrms, density-PDF
   width σ_s, the shell-binned power spectrum) vs an independent fp32/fp64 implementation.
4. **Robustness**: the light/transverse-free schemes survive strong driving (finite, positive) but with
   looser ∇·B — characterize the regime before claiming a science-ready win.
5. **Occupancy actuals**: confirm regs/shared/blocks with `cudaFuncGetAttributes` /
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor` — the cliff math is only as good as the real numbers.
