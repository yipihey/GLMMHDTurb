# turb_robustness.jl — drive the nvcc GLM-MHD march (spike_mhd.cu, via MarchBridge) under the
# project's OU turbulence forcing and check robustness as the forcing amplitude (Mach) is ramped.
#
# The C march does the pure GLM-MHD flux update + Dedner psi-damping over shared device memory;
# this harness adds, in Julia, on the SAME CuArrays: (1) a CFL-adaptive dt each step, (2) the
# positivity floor and (3) the OU driving (remove-KE / kick / restore-E), exactly mirroring the
# fused cube kernel (glmmhd_turb.jl update_kernel!). So it is the cube's turbulence test rig
# wrapped around nvcc's kernel — best of both: nvcc throughput + the validated forcing/diagnostics.
#
#   bash build_mhd.sh 192 .                                   # -> libmhd192.so
#   julia --project=.. march_bridge/turb_robustness.jl        # ramp sweep
using CUDA, Printf
include(joinpath(@__DIR__, "..", "glmmhd_turb.jl")); using .GLMMHDTurb
include(joinpath(@__DIR__, "MarchBridge.jl"));        using .MarchBridge
const M = GLMMHDTurb

# OU driving + positivity floor on the bridge's SoA planes (mirrors update_kernel! lines 448-475).
function force_kernel!(q1,q2,q3,q4,q5,q6,q7,q8, afield, dt, dx, boxlen, ramp,
                       smallr, smallc, turb_min_rho, pfl, gamma, N)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if i <= N*N*N
        @inbounds begin
            k = (i-1) ÷ (N*N); j = ((i-1) ÷ N) % N; ii = (i-1) % N      # 0-based
            I, J, K = ii+1, j+1, k+1
            rho = q1[I,J,K]; mx=q2[I,J,K]; my=q3[I,J,K]; mz=q4[I,J,K]; E=q5[I,J,K]
            bx=q6[I,J,K]; by=q7[I,J,K]; bz=q8[I,J,K]
            if rho < smallr                                              # static floor (void cell)
                emag = 0.5f0*(bx*bx+by*by+bz*bz)
                rho=smallr; mx=0f0; my=0f0; mz=0f0; E=pfl/(gamma-1f0)+emag
            end
            if rho >= turb_min_rho
                ax,ay,az = M.turb_interp(afield, ii, j, k, dx, boxlen, ramp)  # 0-based like the kernel
                rhom = max(rho, smallr); e = E; fl = rho*smallc*smallc
                e = max(e - 0.5f0*mx*mx/rhom, fl); e = max(e - 0.5f0*my*my/rhom, fl); e = max(e - 0.5f0*mz*mz/rhom, fl)
                mx = mx + rhom*ax*dt; my = my + rhom*ay*dt; mz = mz + rhom*az*dt
                e = max(e + 0.5f0*mx*mx/rhom, fl); e = max(e + 0.5f0*my*my/rhom, fl); e = max(e + 0.5f0*mz*mz/rhom, fl)
                E = e
            end
            q1[I,J,K]=rho; q2[I,J,K]=mx; q3[I,J,K]=my; q4[I,J,K]=mz; q5[I,J,K]=E
        end
    end
    return
end

# SoA diagnostics (no (N,N,N,NV) assembly): CFL dt, Mach, max|divB|, rhomin, finiteness.
function soa_primitives(q, gamma)
    rho = max.(q[1], 1f-30)
    vx = q[2]./rho; vy = q[3]./rho; vz = q[4]./rho
    b2 = q[6].^2 .+ q[7].^2 .+ q[8].^2
    p  = max.((gamma-1f0).*(q[5] .- 0.5f0.*rho.*(vx.^2 .+ vy.^2 .+ vz.^2) .- 0.5f0.*b2), 1f-30)
    rho, vx, vy, vz, p, b2
end
function max_signal_speed(q, gamma)
    rho, vx, vy, vz, p, b2 = soa_primitives(q, gamma)
    cf = sqrt.((gamma.*p .+ b2)./rho)                 # fast-speed upper bound (cs^2 + vA^2)
    Float64(maximum(sqrt.(vx.^2 .+ vy.^2 .+ vz.^2) .+ cf))
end
function maxdivB(bx,by,bz,dx)
    db = (circshift(bx,(-1,0,0)).-circshift(bx,(1,0,0)).+
          circshift(by,(0,-1,0)).-circshift(by,(0,1,0)).+
          circshift(bz,(0,0,-1)).-circshift(bz,(0,0,1)))./(2f0*dx)
    Float64(maximum(abs.(db)))
end

function run_one(N, amp; t_end=0.6f0, nstepmax=4000, libdir=@__DIR__, verbose=true,
                 courant=0.4f0, diag=10)
    lib = joinpath(libdir, "libmhd$(N).so")
    isfile(lib) || error("missing $lib — run: bash build_mhd.sh $N .")
    m = MarchBridge.open_lib(lib); @assert m.NV == 9
    be = CUDA.CUDABackend()
    p = M.Params(N=N, turb_rms=amp, turb_min_rho=1f-5, courant=courant)
    dx = M.dxof(p); pfl = M.pfloor(p)
    q = [CUDA.zeros(Float32,N,N,N) for _ in 1:9]; o = [similar(x) for x in q]
    q[1] .= p.rho0; q[5] .= (p.rho0*p.cs0^2/p.gamma)/(p.gamma-1f0) + 0.5f0*p.b0^2; q[6] .= p.b0  # uniform IC, Bx0=b0
    for v in 1:9; o[v] .= q[v]; end
    f = M.Forcing(p, be; amp=amp); M.ou_advance!(f, p, be)
    nb = cld(N^3, 256)
    t = 0f0; step = 0; worst_div = 0.0; worst_mach = 0.0; rhomin = 1.0; ok = true
    while t < t_end && step < nstepmax
        while t >= f.next_time; M.ou_advance!(f, p, be); end
        smax = max_signal_speed(q, p.gamma)
        dt = Float32(p.courant * dx / smax); dt = min(dt, f.next_time - t, t_end - t)
        ch = p.courant*dx/dt/Float32(M.NDIM)*p.glm_ch_scale
        glmfac = p.glm_cp_coef > 0 ? Float32(exp(-(ch/(p.glm_cp_coef*p.boxlen))*dt)) : 1f0
        MarchBridge.set_dtdx(m, dt/dx); MarchBridge.set_glm(m, ch, glmfac)
        MarchBridge.run!(m, q, o, 1)
        ramp = min(t/p.turb_T, 1f0)
        @cuda threads=256 blocks=nb force_kernel!(q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],
            f.afield, dt, dx, p.boxlen, ramp, p.smallr, p.smallc, p.turb_min_rho, pfl, p.gamma, N)
        t += dt; step += 1
        if isnan(dt) || !isfinite(Float64(sum(q[5])))   # cheap per-step blowup guard
            @printf("    BLEW UP at step=%d t=%.4f dt=%.2e\n", step, t, dt); ok=false; break
        end
        if step % diag == 0 || t >= t_end
            rho, vx, vy, vz, pr, b2 = soa_primitives(q, p.gamma)
            mach = Float64(sqrt(sum(vx.^2 .+ vy.^2 .+ vz.^2)/N^3) / p.cs0)
            dB = maxdivB(q[6],q[7],q[8],dx); rm = Float64(minimum(q[1]))
            fin = all(isfinite, (mach, dB, rm)) && Bool(all(isfinite.(q[1])))
            worst_div = max(worst_div, dB); worst_mach = max(worst_mach, mach); rhomin = min(rhomin, rm)
            ok &= fin
            verbose && @printf("    step=%4d t=%.3f dt=%.2e Mach=%.2f divBmax=%.2e rhomin=%.2e finite=%s\n",
                               step, t, dt, mach, dB, rm, fin)
            fin || break
        end
    end
    MarchBridge.close_lib(m)
    (steps=step, t=t, peak_mach=worst_mach, worst_divB=worst_div, rhomin=rhomin, finite=ok)
end

if abspath(PROGRAM_FILE) == @__FILE__
    N = haskey(ENV,"N") ? parse(Int, ENV["N"]) : 192
    @printf("GLM-MHD turbulence robustness — nvcc march (bridge) + OU forcing, N=%d^3\n", N)
    for amp in (1f6, 3f6, 1f7, 3f7)
        @printf("--- amp=%.0e ---\n", amp)
        r = run_one(N, amp; verbose=true)
        @printf(">>> amp=%.0e: steps=%d t=%.3f peakMach=%.2f worst|divB|=%.2e rhomin=%.2e finite=%s\n\n",
                amp, r.steps, r.t, r.peak_mach, r.worst_divB, r.rhomin, r.finite)
    end
end
