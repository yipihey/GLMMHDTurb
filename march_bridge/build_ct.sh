#!/usr/bin/env bash
# Build the hand-tuned f16 constrained-transport MHD kernel as a shared library.
#   ./build_ct.sh [N] [OX OY OZ] [src]
# Default = spike_ct3.cu (face-B-from-global, 2 blocks/SM at OZ=3 on N%3==0 grids): ~1500 Mcell/s @480.
# For N not divisible by 3, use spike_ct2.cu (face-B-in-tile, 1 block, any even N): ~1200 Mcell/s.
#   SRC=cu/spike_ct2.cu ./build_ct.sh 256 8 8 4
# Both: staged (each face flux computed once -> shared f16 flux tile), f32 update base from global
# (div·B machine-zero), --use_fast_math, 64-74 regs. spike_ct.cu (recompute, 61) = validated-negative.
set -euo pipefail
N="${1:-480}"; OX="${2:-8}"; OY="${3:-8}"; OZ="${4:-3}"; OUT="$(dirname "$0")"
SPIKE="${SRC:+$OUT/$SRC}"; SPIKE="${SPIKE:-$OUT/cu/spike_ct3.cu}"
ARCH="${ARCH:-sm_86}"; NVCC="${NVCC:-}"
if [ -z "$NVCC" ]; then for c in /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/bin/nvcc /usr/local/cuda*/bin/nvcc; do [ -x "$c" ] && NVCC="$c" && break; done; fi
[ -x "$NVCC" ] || { echo "no nvcc"; exit 1; }
echo "build: $(basename "$SPIKE") N=$N tile=${OX}x${OY}x${OZ} f16 -> $OUT/libct${N}.so"
"$NVCC" -O3 -arch="$ARCH" --use_fast_math -DAS_LIB -DTILET=__half -DOX=$OX -DOY=$OY -DOZ=$OZ \
  -DNX=$N -DNY=$N -DNZ=$N --shared -Xcompiler -fPIC -o "$OUT/libct${N}.so" "$SPIKE"
echo ok
