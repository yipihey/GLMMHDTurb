# march_bridge — nvcc kernel speed, Julia test rig

Drive the hand-written CUDA 2.5D-march godunov kernel
(`mini-ramses-metal/gpu/spike_25d.cu`, light `HANCOCK1D` scheme) directly from
CUDA.jl, sharing device memory. You get nvcc throughput **and** the Julia
harness (CUFFT spectra, conservation checks) on the same `CuArray`s.

## Why

The Julia-native `integrator_hydro_lmarch!` is codegen-bound, not
algorithm-bound: CUDA.jl emits ~96 registers for the same light scheme where
nvcc emits ~80, dropping A6000 occupancy from 3 to 2 blocks/SM. Forcing the
register count down (`maxregs`) only spills and is strictly worse. Net: the
Julia kernel runs ~4–5× slower than the identical algorithm through nvcc.

Measured on an RTX A6000 (sm_86), 480³, HANCOCK1D, nvcc cuda-13.1,
`--use_fast_math` (the build_march.sh default):

| kernel                          | regs | blocks/SM | Mcell/s |
|---------------------------------|------|-----------|---------|
| nvcc march (this bridge)        | 64   | 4         | ~6800   |
| Julia-native `lmarch!`          | 96   | 2         | ~1155   |

→ **~5.9×**, with zero bridge overhead (the timed loop is pure kernel launches
over shared device memory; host transfer only at setup; mass drift −4e-5).

> The `--use_fast_math` flag is load-bearing: without it the kernel is 80 regs /
> 3 blocks / ~5300 — one register over the A6000's 64-reg/4-block cliff. Fast-math
> lets ptxas use approximate reciprocal/sqrt in the HLL sound-speed and 1/rho path,
> freeing 16 registers and the 4th block. This is *not* a driver or compiler-version
> effect: the IEEE-accurate source compiles to 80 regs on ptxas 11.8, 12.9 and 13.3
> alike (11.8 is the worst at 86). `NO_FASTMATH=1 bash build_march.sh` builds the
> accurate 3-block variant for comparison.

## GLM-MHD (9-var)

`spike_mhd.cu` is the MHD twin: 9-var Dedner GLM-MHD (rho,mx,my,mz,E,Bx,By,Bz,psi),
same transverse-free 1D-Hancock march + HLL + divergence cleaning, physics
transliterated faithfully from `glmmhd_turb.jl`. Build with `build_mhd.sh`; the
bridge auto-detects `NV==9` and exposes `set_glm(m, ch, glmfac)`.

A6000, 480³, `--use_fast_math`:

| kernel                        | regs | blocks/SM | Mcell/s |
|-------------------------------|------|-----------|---------|
| nvcc GLM-MHD march (bridge)   | 128  | 2         | ~3100   |
| Julia-native cube `step_plm!` | 144  | 2         | ~1920   |

→ **~1.6×**. Validated driving it from Julia: mass drift ~5e-5, energy ~1e-4,
finite, and **div·B is actively cleaned** (max|div·B| 30→3.4 over 55 steps, psi
bounded). Register-bound at 2 blocks (both regs and the 38 KB f16 tile cap it);
fp32 traffic is only ~29% of peak BW, so this is the A6000 ceiling for the scheme.

### Turbulence robustness (`turb_robustness.jl`)

Driving the nvcc MHD march with the project's OU forcing (the cube's own forcing +
positivity floor + CFL, applied in Julia on the shared CuArrays) and ramping the
amplitude. N=192, matched `courant=0.4`, vs the validated cube `step_plm!`:

| amp  | march: Mach / max\|div·B\| / ρmin / finite | cube: Mach / max\|div·B\| / ρmin / finite |
|------|--------------------------------------------|-------------------------------------------|
| 3e5  | 1.26 / 1e3  / floor / ✓ | 0.85 / 19  / 0.12  / ✓ |
| 1e6  | 3.86 / 2e4  / floor / ✓ | 1.53 / 66  / 0.02  / ✓ |
| 3e6  | 3.59 / 2e2  / floor / ✓ | 0.53 / 3   / 0.43  / ✓ |
| 1e7  | 6.33 / 4e2  / floor / ✓ | 3.16 / 2e2 / 4e-4  / ✓ |

Findings, stated honestly:
- **Stability:** at CFL ≤ 0.4 the march never NaNs, even at Mach ~6. (At CFL 0.7
  both the march and the cube blow up under strong MHD driving — the "cube NaNs
  above Mach 4" note was a CFL=0.7 effect, not intrinsic.)
- **Quality:** the march develops 1–3 orders larger ∇·B and drives ρ to the floor
  (voids + B-spikes), where the cube keeps ∇·B bounded and density physical. The
  transverse Hancock predictor the light scheme drops is exactly what controls the
  ∇·B constraint and positivity in MHD. **The hydro robustness win does NOT carry
  over to MHD** — for science needing tight ∇·B, the cube's predictor matters.

So the right split for MHD: the bridge march is the **throughput** path (~1.6×,
survives hard driving) for exploration/scaling; the cube is the **accuracy/∇·B**
path. Run `turb_robustness.jl` to reproduce (set `ENV["N"]`).

## Build

```bash
bash build_march.sh 480            # hydro  -> libmarch480.so (N div by 32 in x, 8 in y)
bash build_mhd.sh   480            # GLM-MHD -> libmhd480.so
NVCC=/path/to/nvcc bash build_march.sh 256   # override toolkit / grid
```

## Use

```julia
using CUDA
include("MarchBridge.jl"); using .MarchBridge
m = MarchBridge.open_lib("libmarch480.so")          # m.NV==5, m.N==480
q = [CUDA.zeros(Float32, m.N, m.N, m.N) for _ in 1:m.NV]   # SoA: rho,mx,my,mz,E
o = [similar(x) for x in q]
# ... fill q with an initial conserved state ...
MarchBridge.set_dtdx(m, 0.02f0)
# MHD lib only: MarchBridge.set_glm(m, ch, glmfac)
ms = MarchBridge.run!(m, q, o, 30)   # 30 periodic steps; result left in q
# now run CUFFT power spectrum / conservation diagnostics on q directly
```

State is fp32 conserved `[rho, mx, my, mz, E]` (hydro) or the 9-var GLM-MHD set,
periodic, fixed `dt` (`set_dtdx`).
The `.so` is a build artifact (gitignored) — rebuild with `build_march.sh`.
