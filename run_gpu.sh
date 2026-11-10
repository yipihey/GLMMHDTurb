#!/bin/bash
# GPU driven GLM-MHD turbulence: HLL vs HLLD vs Ducros (widened + threshold), several forcing seeds.
#   N=128 T_END=1.5 SEEDS="1 2 3" [SOLVER=plm] ./run_gpu.sh   (SOLVER=plm: fast f16 tile, out dir gpu_plm_*)
# SMALLR=1e-3 default: at the 1e-6 Params floor HLL itself floors at Mach>=2 and dt collapses (see log).
# Forcing is seeded from SEED on both backends, so configs see identical forcing per seed.
cd "$(dirname "$0")"
N=${N:-128}; T_END=${T_END:-1.5}; SEEDS=${SEEDS:-"1 2 3"}; CFGS=${CFGS:-hll,hlld,hll_wide_thr}
export SMALLR=${SMALLR:-1e-3} FKMAX=${FKMAX:-0}
export BACKEND=gpu AMP=${AMP:-3e6} NSTEPMAX=${NSTEPMAX:-400000} TAVG0=${TAVG0:-0.5} NSNAP=${NSNAP:-11}
for seed in $SEEDS; do
  out=ducros_out/gpu${SOLVER:+_$SOLVER}_n${N}_r${SMALLR}_k${FKMAX}_s$seed; mkdir -p $out
  echo "=== N=$N seed=$seed cfgs=$CFGS start $(date +%T)"
  SEED=$seed OUTDIR=$out julia --project=. test_ducros.jl turb $N $T_END $CFGS > $out/run.log 2>&1
  grep -E "^TURB|->|NON" $out/run.log
done
echo "=== ALL DONE $(date +%T)"
