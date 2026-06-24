# ot_validate.jl — matched-IC Orszag-Tang cross-validation: nvcc GLM-MHD march (bridge) vs
# the validated cube `step_plm!`. IDENTICAL IC (the same ic_ot_kernel! state copied into both
# layouts), IDENTICAL fixed dt sequence and ch/glmfac, both HLL + f16 tile, no forcing. The
# ONLY difference is the reconstruction predictor: cube = full transverse Hancock, march =
# transverse-free 1D Hancock. So the field difference isolates exactly the transverse term.
#
#   bash build_mhd.sh 192 march_bridge
#   julia --project=.. march_bridge/ot_validate.jl
using CUDA, Printf
include(joinpath(@__DIR__, "..", "glmmhd_turb.jl")); using .GLMMHDTurb
include(joinpath(@__DIR__, "MarchBridge.jl"));        using .MarchBridge
const M = GLMMHDTurb

relerr(a, b) = Float64(sqrt(sum((a .- b).^2)) / max(sqrt(sum(b.^2)), 1f-30))

function ot_compare(N; nsteps=120, cfl=0.4f0, libdir=@__DIR__)
    be = CUDA.CUDABackend()
    p = M.Params(N=N, gamma=5f0/3f0, cs0=1f0, courant=cfl)
    dx = M.dxof(p); ct = CUDA.zeros(Float32,N,N,N)
    # shared OT initial condition
    u1 = CUDA.zeros(Float32,N,N,N,M.NV); u2 = similar(u1)
    M.ic_ot_kernel!(be)(u1, N, p.gamma; ndrange=(N,N,N)); CUDA.synchronize()
    # march SoA planes, copied bit-identical from the cube IC
    m = MarchBridge.open_lib(joinpath(libdir,"libmhd$(N).so")); @assert m.NV==9
    MarchBridge.set_gamma(m, p.gamma)
    q = [CUDA.zeros(Float32,N,N,N) for _ in 1:9]; o = [similar(x) for x in q]
    for v in 1:9; q[v] .= @view u1[:,:,:,v]; o[v] .= q[v]; end
    # ONE fixed dt from the IC, used by both for every step (isolates the scheme, not dt-control)
    dt = M.compute_dt(u1, ct, p, be)
    ch = p.courant*dx/dt/Float32(M.NDIM)*p.glm_ch_scale
    glmfac = Float32(exp(-(ch/(p.glm_cp_coef*p.boxlen))*dt))
    MarchBridge.set_dtdx(m, dt/dx); MarchBridge.set_glm(m, ch, glmfac)
    @printf("OT N=%d^3  gamma=5/3  fixed dt=%.3e  ch=%.3f  %d steps (no forcing)\n", N, dt, ch, nsteps)
    for s in 1:nsteps
        M.step_plm!(u1, u2, q[1], p, dt, 0f0; do_turb=false, riemann=:hll)  # afield unused (do_turb=false)
        u1, u2 = u2, u1
        MarchBridge.run!(m, q, o, 1)
    end
    CUDA.synchronize()
    # compare final fields (cube u1 vs march q)
    names = ("rho","mx","my","mz","E","Bx","By","Bz","psi")
    @printf("  per-field  relL2     max|Δ|   (march vs cube; mz/Bz/psi≈0 in 2D OT → relL2 ill-defined)\n")
    for v in 1:9
        d = q[v] .- @view u1[:,:,:,v]
        @printf("    %-4s %.3e  %.3e\n", names[v], relerr(q[v], @view u1[:,:,:,v]), Float64(maximum(abs.(d))))
    end
    # scalar diagnostics each
    function diag(getρ,getmx,getmy,getmz,getbx,getby,getbz)
        ρ=getρ; vx=getmx./ρ; vy=getmy./ρ; vz=getmz./ρ
        vrms=Float64(sqrt(sum(vx.^2 .+vy.^2 .+vz.^2)/N^3))
        db=(circshift(getbx,(-1,0,0)).-circshift(getbx,(1,0,0)).+circshift(getby,(0,-1,0)).-circshift(getby,(0,1,0)).+
            circshift(getbz,(0,0,-1)).-circshift(getbz,(0,0,1)))./(2f0*dx)
        (vrms=vrms, divB=Float64(maximum(abs.(db))), rhomin=Float64(minimum(ρ)))
    end
    dc = diag(u1[:,:,:,1],u1[:,:,:,2],u1[:,:,:,3],u1[:,:,:,4],u1[:,:,:,6],u1[:,:,:,7],u1[:,:,:,8])
    dm = diag(q[1],q[2],q[3],q[4],q[6],q[7],q[8])
    @printf("  CUBE : vrms=%.4f divBmax=%.3e rhomin=%.4f\n", dc.vrms, dc.divB, dc.rhomin)
    @printf("  MARCH: vrms=%.4f divBmax=%.3e rhomin=%.4f\n", dm.vrms, dm.divB, dm.rhomin)
    MarchBridge.close_lib(m)
end

if abspath(PROGRAM_FILE) == @__FILE__
    N = haskey(ENV,"N") ? parse(Int, ENV["N"]) : 192
    for ns in (40, 120, 300)
        ot_compare(N; nsteps=ns)
        println()
    end
end
