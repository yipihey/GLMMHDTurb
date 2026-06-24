#!/usr/bin/env bash
# Build the fused constrained-transport MHD kernel (spike_ct.cu) as a shared library.
# CORRECTNESS-validated (div·B machine-zero, matches ct_mhd.jl); PERF is WIP (register-bound).
#   ./build_ct.sh [N] [outdir]
set -euo pipefail
N="${1:-128}"; OUT="${2:-$(dirname "$0")}"; SPIKE="${SPIKE_CU:-$(dirname "$0")/cu/spike_ct.cu}"
ARCH="${ARCH:-sm_86}"; NVCC="${NVCC:-}"
if [ -z "$NVCC" ]; then for c in /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/bin/nvcc /usr/local/cuda*/bin/nvcc; do [ -x "$c" ] && NVCC="$c" && break; done; fi
[ -x "$NVCC" ] || { echo "no nvcc"; exit 1; }
echo "build: N=$N -> $OUT/libct${N}.so"
"$NVCC" -O3 -arch="$ARCH" --use_fast_math -DAS_LIB -DNX="$N" -DNY="$N" -DNZ="$N" \
  --shared -Xcompiler -fPIC -o "$OUT/libct${N}.so" "$SPIKE"
echo "ok"
