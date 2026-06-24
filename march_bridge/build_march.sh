#!/usr/bin/env bash
# Build the nvcc 2.5D-march kernel (mini-ramses-metal spike_25d.cu) as a shared library
# callable from Julia/CUDA.jl over shared device memory. See MarchBridge.jl.
#
#   ./build_march.sh [N] [outdir]
#
# N defaults to 480 (must be divisible by OX=32 in x and OY=8 in y). The light HANCOCK1D
# scheme (-DHANCOCK1D: transverse-free 2nd-order, the register-light reference) is selected;
# fp32-conserved SoA state (rho,mx,my,mz,E) matching CUDA.jl plane arrays.
#
# --use_fast_math is ESSENTIAL here, not cosmetic: it lets ptxas use approximate reciprocal/
# sqrt in the HLL sound-speed and 1/rho path, dropping the kernel 80->64 registers, which
# restores the 4th block/SM on the A6000 (3->4 blocks) and lifts throughput ~5300->6600
# Mcell/s. Without it the kernel is one register over the 64-reg/4-block cliff. (The 80-vs-86
# reg spread across ptxas 11.8/12.9/13.3 is <1 block; the fast-math flag is the real lever.)
# Set NO_FASTMATH=1 to build the IEEE-accurate (slower, 3-block) variant for an accuracy check.
set -euo pipefail
N="${1:-480}"
OUT="${2:-$(dirname "$0")}"
SPIKE="${SPIKE_CU:-$(dirname "$0")/cu/spike_25d.cu}"
ARCH="${ARCH:-sm_86}"

# locate an nvcc (prefer the NVHPC one the project builds with)
NVCC="${NVCC:-}"
if [ -z "$NVCC" ]; then
  for c in /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/bin/nvcc /usr/local/cuda*/bin/nvcc; do
    [ -x "$c" ] && NVCC="$c" && break
  done
fi
[ -x "$NVCC" ] || { echo "no nvcc found; set NVCC=..." >&2; exit 1; }

FASTMATH="--use_fast_math"
[ -n "${NO_FASTMATH:-}" ] && FASTMATH=""
LIB="$OUT/libmarch${N}.so"
echo "nvcc: $NVCC"
echo "build: N=${N} arch=${ARCH} fastmath='${FASTMATH:-off}' -> $LIB"
"$NVCC" -O3 -arch="$ARCH" $FASTMATH -DAS_LIB -DHANCOCK1D \
  -DNX="$N" -DNY="$N" -DNZ="$N" \
  --shared -Xcompiler -fPIC -o "$LIB" "$SPIKE"
echo "ok: $(ls -la "$LIB")"
