#!/usr/bin/env bash
# Build the STAGED f16 constrained-transport MHD kernel (spike_ct2.cu) as a shared library.
# Staged: compute each face flux ONCE into a shared f16 flux tile, then EMF+update read it
# (f32 update base from global — div·B stays machine-zero, the GLM f16-tile+f32-update lesson).
# ~1200 Mcell/s @480, 64 regs. spike_ct.cu (recompute, 61 Mcell/s) kept as the validated-negative.
#   ./build_ct.sh [N] [OX OY OZ]
set -euo pipefail
N="${1:-480}"; OX="${2:-8}"; OY="${3:-8}"; OZ="${4:-4}"; OUT="$(dirname "$0")"
SPIKE="${SPIKE_CU:-$OUT/cu/spike_ct2.cu}"; ARCH="${ARCH:-sm_86}"; NVCC="${NVCC:-}"
if [ -z "$NVCC" ]; then for c in /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/bin/nvcc /usr/local/cuda*/bin/nvcc; do [ -x "$c" ] && NVCC="$c" && break; done; fi
[ -x "$NVCC" ] || { echo "no nvcc"; exit 1; }
echo "build: N=$N tile=${OX}x${OY}x${OZ} f16 -> $OUT/libct${N}.so"
"$NVCC" -O3 -arch="$ARCH" --use_fast_math -DAS_LIB -DTILET=__half -DOX=$OX -DOY=$OY -DOZ=$OZ \
  -DNX=$N -DNY=$N -DNZ=$N --shared -Xcompiler -fPIC -o "$OUT/libct${N}.so" "$SPIKE"
echo ok
