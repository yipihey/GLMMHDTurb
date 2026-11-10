# Parity of the Ducros-scaled flux in the fast f16 plm tile vs the KA multi-kernel path.
#   julia --project=. test_plm_ducros.jl [N=96] [nsteps=40]
# 3D Orszag-Tang, identical fixed dt. Pass criterion: |plm_duc - multi_duc| is the same size as
# the plain f16-tile error |plm_hll - multi_hll|, and both << the Ducros effect |multi_duc - multi_hll|.
include("glmmhd_turb.jl"); using .GLMMHDTurb; const G = GLMMHDTurb
using CUDA, KernelAbstractions, Printf, Statistics
const KA = KernelAbstractions
N  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 96
NS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 40
be = CUDABackend()
p = G.Params(N=N, gamma=5f0/3f0, cs0=1f0, switch_llf_dmin=0.01f0, switch_llf_pmin=0.01f0/1.01f0)
u0 = CUDA.zeros(Float32, N,N,N,G.NV); G.ic_ot_kernel!(be)(u0, N, p.gamma; ndrange=(N,N,N)); KA.synchronize(be)
ct = CUDA.zeros(Float32, N,N,N); af = CUDA.zeros(Float32, 3,G.TURB_GS,G.TURB_GS,G.TURB_GS)
dt = G.compute_dt(u0, ct, p, be)
duc = (phimin=0.1f0, thr=0.2f0, widen=true)
function run(kind)
    u1 = copy(u0); u2 = similar(u0); sen = CUDA.zeros(Float32, N,N,N)
    rec = CUDA.zeros(Float32, N,N,N,36); flx = CUDA.zeros(Float32, N,N,N,27)
    for s in 1:NS
        if kind === :multi_hll
            G.step_multi!(u1,u2,rec,flx,ct,af,p,dt,0f0,be; riemann=:hll)
        elseif kind === :multi_duc
            G.step_multi!(u1,u2,rec,flx,ct,af,p,dt,0f0,be; riemann=:hll, ducros=true, sen=sen, duc...)
        elseif kind === :plm_hll
            G.step_plm!(u1,u2,af,p,dt,0f0; tb=6, half=true, riemann=:hll)
        elseif kind === :plm_duc
            G.step_plm!(u1,u2,af,p,dt,0f0; tb=6, half=true, riemann=:hll, ducros=true, sen=sen, duc...)
        elseif kind === :plm_duc_tb4
            G.step_plm!(u1,u2,af,p,dt,0f0; tb=4, half=true, riemann=:hll, ducros=true, sen=sen, duc...)
        elseif kind === :plm32_hll
            G.step_plm!(u1,u2,af,p,dt,0f0; tb=4, half=false, riemann=:hll)
        elseif kind === :plm32_duc
            G.step_plm!(u1,u2,af,p,dt,0f0; tb=4, half=false, riemann=:hll, ducros=true, sen=sen, duc...)
        end
        u1, u2 = u2, u1
    end
    CUDA.synchronize(); return Array(u1)
end
R = Dict(k => run(k) for k in (:multi_hll, :multi_duc, :plm_hll, :plm_duc, :plm_duc_tb4, :plm32_hll, :plm32_duc))
d(a,b,v) = (x = R[a][:,:,:,v] .- R[b][:,:,:,v]; (maximum(abs.(x)), sqrt(mean(abs2, x))))
println("OT N=$N, $NS steps, dt=$dt")
for (v,nm) in ((1,"rho"),(5,"E"),(6,"Bx"))
    @printf("%-3s  f16 tile err |plm_hll-multi_hll| max %.2e rms %.2e | port err |plm_duc-multi_duc| max %.2e rms %.2e | Ducros effect |multi_duc-multi_hll| max %.2e rms %.2e | tb4 vs tb6 duc max %.2e\n",
            nm, d(:plm_hll,:multi_hll,v)..., d(:plm_duc,:multi_duc,v)..., d(:multi_duc,:multi_hll,v)..., d(:plm_duc_tb4,:plm_duc,v)[1])
end
for (v,nm) in ((1,"rho"),(5,"E"),(6,"Bx"))
    @printf("%-3s  f32 tile: |plm32_hll-multi_hll| max %.2e | |plm32_duc-multi_duc| max %.2e   (f32 port check)\n",
            nm, d(:plm32_hll,:multi_hll,v)[1], d(:plm32_duc,:multi_duc,v)[1])
end
println("finite: ", all(all(isfinite, R[k]) for k in keys(R)))
