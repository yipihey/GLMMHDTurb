# Time to solution on the production fast path (f16 plm tile), driven Mach-4 turbulence.
#   julia --project=. bench_tts.jl [N=192] [t_end=1.0] [seeds=1,2,3] [cfgs=hll,hll_wide_thr,hlld]
# Whole-run wall time (incl. compute_dt, forcing, launch overhead), no snapshot I/O.
include("glmmhd_turb.jl"); using .GLMMHDTurb; const G = GLMMHDTurb
using Printf, Statistics
N     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 192
TEND  = length(ARGS) >= 2 ? parse(Float32, ARGS[2]) : 1.0f0
SEEDS = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ",")) : [1,2,3]
CFGS  = length(ARGS) >= 4 ? Symbol.(split(ARGS[4], ",")) : [:hll, :hll_wide_thr, :hlld]
kw(c) = c === :hll ? (riemann=:hll,) : c === :hlld ? (riemann=:hlld,) :
        c === :hll_wide_thr ? (riemann=:hll, ducros=true, phimin=0.1f0, thr=0.2f0, widen=true) : error("cfg $c")
# warm-up compile
for c in CFGS; G.run_turb(N=N, t_end=1f-3, amp=3f6, backend=:gpu, solver=:plm, verbose=false, seed=1; kw(c)...); end
res = Dict{Symbol,Vector{Tuple{Float64,Int,Bool}}}()
for s in SEEDS, c in CFGS
    t0 = time()
    d, _ = G.run_turb(N=N, t_end=TEND, amp=3f6, backend=:gpu, solver=:plm, verbose=false, seed=s,
                      nstepmax=2_000_000, return_state=true; kw(c)...)
    w = time() - t0
    # (step counts are in the TURB lines printed by run_turb)
    push!(get!(res, c, Tuple{Float64,Int,Bool}[]), (w, 0, d.finite))
    @printf("TTS N=%d seed=%d %-13s wall=%7.1fs finite=%s Mach=%.2f\n", N, s, c, w, d.finite, d.mach); flush(stdout)
end
println()
for c in CFGS
    r = res[c]; ok = [x[1] for x in r if x[3]]
    @printf("%-13s finite %d/%d   mean wall %.1fs (finite runs)\n", c, count(x->x[3], r), length(r), isempty(ok) ? NaN : mean(ok))
end
