#!/bin/bash
# 64^3 driven GLM-MHD turbulence: HLL vs HLLD vs Ducros (widened + threshold), 2 forcing seeds.
cd "$(dirname "$0")"
export AMP=3e6 NSTEPMAX=40000 TAVG0=0.4 NSNAP=7
for seed in 1 2; do
  for cfg in hll hlld hll_wide_thr; do
    out=ducros_out/n64_s$seed; mkdir -p $out
    echo "=== seed=$seed cfg=$cfg start $(date +%T)"
    SEED=$seed OUTDIR=$out julia --project=. -t 4 test_ducros.jl turb 64 1.0 $cfg > $out/$cfg.log 2>&1
    grep -E "^TURB|->|NON" $out/$cfg.log
  done
done
echo "=== ALL DONE $(date +%T)"
