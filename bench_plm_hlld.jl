# Benchmark the fast step_plm! path: HLL vs HLLD at f16/TB=6 and TB=4 (+ the f32/TB4 HLLD reference).
#   julia --project=. bench_plm_hlld.jl [N=384] [nsteps=30]
# State: Orszag-Tang 3D (non-degenerate MHD everywhere, so HLLD takes its real branches; a uniform
# IC would flatter it). Reports regs/thread, static smem/block, active blocks/SM, Mcell/s, and an
# accuracy check of each config vs the f32/TB4/HLLD reference after a few OT steps.
include("glmmhd_turb.jl"); using .GLMMHDTurb; const G = GLMMHDTurb
using CUDA, KernelAbstractions, Printf, Statistics
N  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 384
NS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 30
# (tb, half, riemann, duc): duc 0 = plain, 1 = hll_wide_thr with the global ducros_kernel! sensor
# prepass (timed INCLUDING the prepass), 2 = hll_wide_thr with the sensor fused into the tile.
CFGS = [(6,true,:hll,0), (6,true,:hlld,0), (6,true,:hll,1), (6,true,:hll,2), (6,true,:hlld,2),
        (4,true,:hll,0), (4,true,:hlld,0), (4,true,:hll,2), (4,false,:hlld,0)]
sen = CUDA.zeros(Float32, N,N,N)
DUCKW = (phimin=0.1f0, thr=0.2f0, widen=true)
be = CUDABackend()
p = G.Params(N=N, gamma=5f0/3f0, cs0=1f0, switch_llf_dmin=0.01f0, switch_llf_pmin=0.01f0/1.01f0)
u0 = CUDA.zeros(Float32, N,N,N,G.NV); G.ic_ot_kernel!(be)(u0, N, p.gamma; ndrange=(N,N,N)); KA = KernelAbstractions; KA.synchronize(be)
ct = CUDA.zeros(Float32, N,N,N); afield = CUDA.zeros(Float32, 3,G.TURB_GS,G.TURB_GS,G.TURB_GS)
dt = G.compute_dt(u0, ct, p, be)
dev = CUDA.device(); nsm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
function kinfo(tb, half, riem, duc, nth)
    pfl = G.pfloor(p); dx = G.dxof(p); ch = p.courant*dx/dt/3f0*p.glm_ch_scale
    k = @cuda launch=false G.integrator_plm!(Val(tb),Val(half),Val(riem), u0,u0,afield, p.gamma,p.smallr,pfl,p.smallc,
        dt,dx,ch,1f0, p.switch_llf_dmin,p.switch_llf_pmin, N,p.boxlen,1f0,p.turb_min_rho,false,
        duc == 1 ? sen : afield, 0.1f0, 0.2f0, true, 0.05f0, Val(duc))
    regs = CUDA.registers(k); smem = CUDA.memory(k).shared
    nb = CUDA.active_blocks(k.fun, nth)
    return regs, smem, nb
end
ref = nothing
println("N=$N nsteps=$NS  (SMs=$nsm)")
@printf("%-18s %4s %6s %8s %9s %9s %10s %12s\n","config","thr","regs","smem[B]","blk/SM","warps/SM","Mcell/s","max|drho| vs ref")
results = []
for nth in (192, 128)
  for (tb,half,riem,duc) in CFGS
    regs, smem, nb = kinfo(tb,half,riem,duc,nth)
    kw = duc == 0 ? (;) : (; ducros=true, sen=sen, fused=(duc == 2), DUCKW...)
    u1 = copy(u0); u2 = similar(u0)
    for s in 1:3; G.step_plm!(u1,u2,afield,p,dt,0f0; tb=tb,half=half,riemann=riem,nthreads=nth,kw...); u1,u2=u2,u1; end
    CUDA.synchronize()
    el = CUDA.@elapsed for s in 1:NS
        G.step_plm!(u1,u2,afield,p,dt,0f0; tb=tb,half=half,riemann=riem,nthreads=nth,kw...); u1,u2=u2,u1
    end
    mc = N^3*NS/el/1e6
    # accuracy: 10 steps from OT IC vs f32/TB4/HLLD
    ua = copy(u0); ub = similar(u0)
    for s in 1:10; G.step_plm!(ua,ub,afield,p,dt,0f0; tb=tb,half=half,riemann=riem,nthreads=nth,kw...); ua,ub=ub,ua; end
    r = Array(@view ua[:,:,:,1])
    if (tb,half,riem,duc) == (4,false,:hlld,0) && nth == 192; global ref = r; end
    err = ref === nothing ? NaN : maximum(abs.(r .- ref))
    push!(results, (tb,half,riem,nth,regs,smem,nb,mc,err,r,duc))
    @printf("tb=%d %s %-15s %4d %6d %8d %9d %9d %10.1f %12.3e\n", tb, half ? "f16" : "f32", string(riem, ("", "+duc_pre", "+duc_fused")[duc+1]), nth, regs, smem, nb, nb*cld(nth,32), mc, err)
  end
end
# errors vs ref for the 192 rows computed before ref existed
println("\naccuracy vs f32/TB4/HLLD (10 OT steps, max|Δρ|, ρ~O(1)):")
for r in results
    r[4] == 192 && @printf("  tb=%d %s %-15s  %.3e\n", r[1], r[2] ? "f16" : "f32", string(r[3], ("", "+duc_pre", "+duc_fused")[r[11]+1]), maximum(abs.(r[10] .- ref)))
end
