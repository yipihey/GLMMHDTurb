#!/usr/bin/env bash
# Build the nvcc 2.5D GLM-MHD light line-march (mini-ramses-metal spike_mhd.cu) as a shared
# library callable from Julia/CUDA.jl. 9-var GLM-MHD (rho,mx,my,mz,E,Bx,By,Bz,psi), transverse-
# free 1D-Hancock + HLL + Dedner divergence cleaning; physics transliterated from glmmhd_turb.jl.
# See MarchBridge.jl (open_lib/set_dtdx/set_glm/run!). f16 shared tile is mandatory (fp32 9-var
# tile is 76KB > 48KB). --use_fast_math frees registers in the fast-speed/HLL sqrt+recip path.
#
#   ./build_mhd.sh [N] [outdir]
set -euo pipefail
N="${1:-480}"
OUT="${2:-$(dirname "$0")}"
SPIKE="${SPIKE_CU:-$(dirname "$0")/cu/spike_mhd.cu}"
ARCH="${ARCH:-sm_86}"
NVCC="${NVCC:-}"
if [ -z "$NVCC" ]; then
  for c in /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/bin/nvcc /usr/local/cuda*/bin/nvcc; do
    [ -x "$c" ] && NVCC="$c" && break
  done
fi
[ -x "$NVCC" ] || { echo "no nvcc found; set NVCC=..." >&2; exit 1; }
FASTMATH="--use_fast_math"
[ -n "${NO_FASTMATH:-}" ] && FASTMATH=""
LIB="$OUT/libmhd${N}.so"
echo "nvcc: $NVCC"
echo "build: N=${N} arch=${ARCH} fastmath='${FASTMATH:-off}' -> $LIB"
"$NVCC" -O3 -arch="$ARCH" $FASTMATH -DAS_LIB \
  -DNX="$N" -DNY="$N" -DNZ="$N" \
  --shared -Xcompiler -fPIC -o "$LIB" "$SPIKE"
echo "ok: $(ls -la "$LIB")"
