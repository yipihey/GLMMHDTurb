#!/usr/bin/env bash
# Build a hand-tuned f16 constrained-transport MHD kernel as a shared library.
#   ./build_ct.sh [N] [OX OY [OZ]] [SRC=cu/...]
# Default = spike_ctm.cu, the 2.5D z-STREAMING march (NEW BEST: ~1560 Mcell/s @480, tile 24x8 or
# 16x12, 2 blocks, machine-zero div·B). Periodicity solved by priming the wrap planes into the ring
# (loadp(-3..-1) + computing magflux(-1)) — the "permanent periodic-copy buffer" trick.
#   Alternatives: SRC=cu/spike_ct3.cu (3D tile, 1501, OZ=3) ; cu/spike_ct2.cu (3D, 1206, any even N).
set -euo pipefail
N="${1:-480}"; OX="${2:-24}"; OY="${3:-8}"; OZ="${4:-1}"; OUT="$(dirname "$0")"
SPIKE="${SRC:+$OUT/$SRC}"; SPIKE="${SPIKE:-$OUT/cu/spike_ctm.cu}"
ARCH="${ARCH:-sm_86}"; NVCC="${NVCC:-}"
if [ -z "$NVCC" ]; then for c in /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/bin/nvcc /usr/local/cuda*/bin/nvcc; do [ -x "$c" ] && NVCC="$c" && break; done; fi
[ -x "$NVCC" ] || { echo "no nvcc"; exit 1; }
echo "build: $(basename "$SPIKE") N=$N tile=${OX}x${OY}x${OZ} f16 -> $OUT/libct${N}.so"
"$NVCC" -O3 -arch="$ARCH" --use_fast_math -DAS_LIB ${TILET:+-DTILET=$TILET} -DOX=$OX -DOY=$OY -DOZ=$OZ \
  -DNX=$N -DNY=$N -DNZ=$N --shared -Xcompiler -fPIC -o "$OUT/libct${N}.so" "$SPIKE"
echo ok
