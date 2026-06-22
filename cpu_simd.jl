# CPU-tuned, SIMD-across-x, fused single-pass GLM-MHD integrator (prototype).
# Vectorizes W cells along the unit-stride x dimension into Vec{W,Float32} lanes, reusing
# the SAME @inline physics as the GPU kernels (now generic over NTuple{9} of scalars or Vecs).
# Layout: a 2-cell x-ghost padding (NP=N+4, N, N, 9) so x-vector loads never wrap; y/z are
# periodic via modular row indices. LLF Riemann (branchless -> vectorizes; HLLD's wave-select
# branches would need vifelse blends). Threaded over k, NUMA-pinned by the driver.
include(joinpath(@__DIR__, "glmmhd_turb.jl"))
using .GLMMHDTurb
const G = GLMMHDTurb
using SIMD, Base.Threads, Printf

const W  = 8
const VF = Vec{W,Float32}
const XG = 2                              # x-ghost width each side

@inline lini(x,j,k,v,NP,N) = x + NP*((j-1) + N*((k-1) + N*(v-1)))   # linear idx into (NP,N,N,9)
# vload/vstore via raw pointer = a true contiguous vector load (vec()/reshape gives a
# ReshapedArray that SIMD.jl falls back to a scalar gather on — correct but ~50x slower).
@inline loadc_v(pad, x0, j, k, NP, N) = ntuple(v -> vload(VF, pointer(pad, lini(x0,j,k,v,NP,N))), 9)

# Vec-friendly MonCen (arithmetic, no ifelse-mask): the module's moncen uses `ifelse`, which
# is fast on the GPU but has no SIMD-mask method. copysign/sign/min/max all vectorize. (This
# arithmetic form is ~2.7x slower than ifelse ON THE GPU, so it lives here, not in the module.)
@inline function moncen_v(dl, dr)
    dc = 0.5f0*(dl + dr)
    mag = min(2.0f0*min(abs(dl), abs(dr)), abs(dc))
    return copysign(mag, dc) * max(sign(dl*dr), zero(dl))
end
@inline prim_slope_v(L, M, R) = ntuple(i -> 0.5f0*moncen_v(M[i]-L[i], R[i]-M[i]), Val(9))

@inline function recon_v(pad, x0, j, jm, jp, k, km, kp, NP, N,
                         gamma::Float32, smallr::Float32, pfl::Float32, dtdx::Float32)
    m0  = G.cons2prim(loadc_v(pad, x0,   j, k, NP,N), gamma, smallr, pfl)
    mxl = G.cons2prim(loadc_v(pad, x0-1, j, k, NP,N), gamma, smallr, pfl)
    mxr = G.cons2prim(loadc_v(pad, x0+1, j, k, NP,N), gamma, smallr, pfl)
    myl = G.cons2prim(loadc_v(pad, x0, jm, k, NP,N), gamma, smallr, pfl)
    myr = G.cons2prim(loadc_v(pad, x0, jp, k, NP,N), gamma, smallr, pfl)
    mzl = G.cons2prim(loadc_v(pad, x0, j, km, NP,N), gamma, smallr, pfl)
    mzr = G.cons2prim(loadc_v(pad, x0, j, kp, NP,N), gamma, smallr, pfl)
    sx = prim_slope_v(mxl,m0,mxr); sy = prim_slope_v(myl,m0,myr); sz = prim_slope_v(mzl,m0,mzr)
    uh = G.hancock(m0, sx, sy, sz, dtdx, gamma)
    mh = G.cons2prim(uh, gamma, smallr, pfl)
    bad = (mh[1] <= smallr) | (mh[5] <= pfl)
    # NB: bind to a NEW name — reassigning `mh` while it's captured by the closure below
    # boxes it (heap alloc per cell, ~50x slowdown). mhf is fresh, so no capture+mutate.
    mhf = ntuple(v -> vifelse(bad, m0[v], mh[v]), Val(9))
    return mhf, sx, sy, sz
end

# LLF directional flux on Vec lanes (rotate -> GLM clean -> LLF -> rotate back), interface floored.
@inline function llf_dir_v(Lq, Rq, dir::Int, gamma::Float32, ch::Float32, smallr::Float32, pfl::Float32)
    L0 = G.rot_to(Lq, dir); R0 = G.rot_to(Rq, dir)
    L = (max(L0[1],smallr), L0[2],L0[3],L0[4], max(L0[5],pfl), L0[6],L0[7],L0[8],L0[9])
    R = (max(R0[1],smallr), R0[2],R0[3],R0[4], max(R0[5],pfl), R0[6],R0[7],R0[8],R0[9])
    bns, psis = G.glm_pair(L[6], R[6], L[9], R[9], ch)
    Lc = (L[1],L[2],L[3],L[4],L[5],bns,L[7],L[8],L[9])
    Rc = (R[1],R[2],R[3],R[4],R[5],bns,R[7],R[8],R[9])
    f = G.llf_x(Lc, Rc, gamma, ch, psis, ch*ch*bns)
    return G.rot_flux_from(f, dir)
end

function simd_step!(unew, uold, p::G.Params, dt::Float32)
    N = p.N; NP = N + 2XG; dx = G.dxof(p); pfl = G.pfloor(p); dtdx = dt/dx
    gamma = p.gamma; smallr = p.smallr; smallc = p.smallc; glm_fac = 1.0f0
    ch = p.courant*dx/dt/Float32(G.NDIM)*p.glm_ch_scale
    if p.glm_cp_coef > 0; glm_fac = exp(-(ch*ch/(p.glm_cp_coef*p.boxlen*ch))*dt); end
    nx = N ÷ W
    @threads for k in 1:N
        km = k==1 ? N : k-1; kp = k==N ? 1 : k+1
        @inbounds for j in 1:N
            jm = j==1 ? N : j-1; jp = j==N ? 1 : j+1
            for xc in 0:nx-1
                x0 = XG + 1 + xc*W                       # owned x start in padded coords
                mhc,sxc,syc,szc = recon_v(uold, x0,   j,jm,jp,k,km,kp,NP,N,gamma,smallr,pfl,dtdx)
                mhxp,sxxp,_,_   = recon_v(uold, x0+1, j,jm,jp,k,km,kp,NP,N,gamma,smallr,pfl,dtdx)
                mhxm,sxxm,_,_   = recon_v(uold, x0-1, j,jm,jp,k,km,kp,NP,N,gamma,smallr,pfl,dtdx)
                Fhx = llf_dir_v(G.padd(mhc,sxc,1.0f0),  G.padd(mhxp,sxxp,-1.0f0), 1, gamma,ch,smallr,pfl)
                Flx = llf_dir_v(G.padd(mhxm,sxxm,1.0f0), G.padd(mhc,sxc,-1.0f0),  1, gamma,ch,smallr,pfl)
                mhyp,_,syyp,_ = recon_v(uold, x0, jp,j,jp2(jp,N),k,km,kp,NP,N,gamma,smallr,pfl,dtdx)
                mhym,_,syym,_ = recon_v(uold, x0, jm,jm2(jm,N),j,k,km,kp,NP,N,gamma,smallr,pfl,dtdx)
                Fhy = llf_dir_v(G.padd(mhc,syc,1.0f0),  G.padd(mhyp,syyp,-1.0f0), 2, gamma,ch,smallr,pfl)
                Fly = llf_dir_v(G.padd(mhym,syym,1.0f0), G.padd(mhc,syc,-1.0f0),  2, gamma,ch,smallr,pfl)
                mhzp,_,_,szzp = recon_v(uold, x0, j,jm,jp,kp,k,kp2(kp,N),NP,N,gamma,smallr,pfl,dtdx)
                mhzm,_,_,szzm = recon_v(uold, x0, j,jm,jp,km,km2(km,N),k,NP,N,gamma,smallr,pfl,dtdx)
                Fhz = llf_dir_v(G.padd(mhc,szc,1.0f0),  G.padd(mhzp,szzp,-1.0f0), 3, gamma,ch,smallr,pfl)
                Flz = llf_dir_v(G.padd(mhzm,szzm,1.0f0), G.padd(mhc,szc,-1.0f0),  3, gamma,ch,smallr,pfl)
                u0 = loadc_v(uold, x0, j, k, NP, N)
                r1 = ntuple(v -> u0[v] + dtdx*((Flx[v]-Fhx[v]) + (Fly[v]-Fhy[v]) + (Flz[v]-Fhz[v])), 9)
                emag = 0.5f0*(r1[6]*r1[6] + r1[7]*r1[7] + r1[8]*r1[8])
                bad = r1[1] < smallr
                fl5 = pfl/(gamma-1.0f0) + emag; z = zero(VF)
                r = (vifelse(bad, VF(smallr), r1[1]), vifelse(bad,z,r1[2]), vifelse(bad,z,r1[3]),
                     vifelse(bad,z,r1[4]), vifelse(bad, fl5, r1[5]), r1[6], r1[7], r1[8], r1[9]*glm_fac)
                for v in 1:9
                    vstore(r[v], pointer(unew, lini(x0,j,k,v,NP,N)))
                end
            end
        end
    end
    return nothing
end
@inline pad_c(u) = u
@inline jp2(jp,N) = jp==N ? 1 : jp+1
@inline jm2(jm,N) = jm==1 ? N : jm-1
@inline kp2(kp,N) = kp==N ? 1 : kp+1
@inline km2(km,N) = km==1 ? N : km-1

# refresh the 2-cell x-ghosts (periodic) on a padded (NP,N,N,9) array
function refresh_xghost!(pad, N)
    NP = N + 2XG
    @inbounds @threads for k in 1:N
        for v in 1:9, j in 1:N, g in 1:XG
            pad[g, j, k, v]          = pad[N+g, j, k, v]      # low ghost <- high interior
            pad[N+XG+g, j, k, v]     = pad[XG+g, j, k, v]     # high ghost <- low interior
        end
    end
end

# Orszag-Tang on the SIMD kernel; returns diagnostics (compared against the GPU/CPU reference).
function run_ot_simd(; N=128, t_end=0.2f0, nsteps=0)
    @assert N % W == 0
    NP = N + 2XG
    p = G.Params(N=N, gamma=5f0/3f0, cs0=1f0)
    be = G.KA.CPU()
    # build IC on the unpadded grid via the KA kernel, then copy into the padded buffer
    u0 = zeros(Float32, N,N,N,9)
    G.ic_ot_kernel!(be)(u0, N, p.gamma; ndrange=(N,N,N)); G.KA.synchronize(be)
    pad1 = zeros(Float32, NP,N,N,9); pad2 = zeros(Float32, NP,N,N,9)
    @views pad1[XG+1:XG+N, :, :, :] .= u0
    ct = zeros(Float32, N,N,N)
    t = 0f0; step = 0; tt = 0.0
    refresh_xghost!(pad1, N)
    simd_step!(pad2, pad1, p, 1f-3)   # warmup
    while (nsteps==0 ? t < t_end : step < nsteps)
        # dt from the interior (reuse the KA ctot on the unpadded view)
        @views ct .= 0f0
        @views uint = pad1[XG+1:XG+N, :, :, :]
        uflat = Array(uint)
        G.ctot_kernel!(be)(ct, uflat, p.gamma, p.smallr, G.pfloor(p); ndrange=(N,N,N)); G.KA.synchronize(be)
        dt = p.courant*G.dxof(p)/maximum(ct)
        nsteps==0 && (dt = min(dt, t_end - t))
        t0 = time(); simd_step!(pad2, pad1, p, dt); refresh_xghost!(pad2, N); tt += time()-t0
        pad1, pad2 = pad2, pad1
        t += dt; step += 1
    end
    uint = Array(@view pad1[XG+1:XG+N, :, :, :])
    d = G.diagnostics(uint, ct, p, be)
    mc = N^3*step/tt/1e6
    @printf("SIMD-OT N=%d threads=%d: steps=%d t=%.4f vrms=%.4f divB=%.3e finite=%s  %.1f Mcell/s\n",
            N, Threads.nthreads(), step, t, d.vrms, d.divbmax, d.finite, mc)
    return d, mc
end

# ============================================================================
# Recon-ONCE, SIMD-across-x, 3-pass path (recon->rec, flux->flx, update). Mirrors the KA
# multi-kernel but vectorized; the scratch makes each reconstruction/Riemann happen once,
# so the freed compute lets it become bandwidth-bound (the fused-recompute above can't).
# Arrays padded by 1 cell in x (all stencils are +-1); y/z periodic via modular rows.
# ============================================================================
const XM = 1                                  # x-ghost for the multi-pass arrays
@inline linm(x,j,k,c,NP,N) = x + NP*((j-1) + N*((k-1) + N*(c-1)))   # (NP,N,N,NC) linear idx
@inline load9(arr, x0, j, k, off, NP, N) = ntuple(v -> vload(VF, pointer(arr, linm(x0,j,k,off+v,NP,N))), 9)

function simd_recon!(rec, u, p::G.Params, dtdx::Float32)
    N = p.N; NP = N + 2XM; gamma = p.gamma; smallr = p.smallr; pfl = G.pfloor(p); nx = N ÷ W
    @threads for k in 1:N
        km = k==1 ? N : k-1; kp = k==N ? 1 : k+1
        @inbounds for j in 1:N
            jm = j==1 ? N : j-1; jp = j==N ? 1 : j+1
            for xc in 0:nx-1
                x0 = XM + 1 + xc*W
                m0  = G.cons2prim(load9(u,x0,  j,k,0,NP,N), gamma,smallr,pfl)
                mxl = G.cons2prim(load9(u,x0-1,j,k,0,NP,N), gamma,smallr,pfl)
                mxr = G.cons2prim(load9(u,x0+1,j,k,0,NP,N), gamma,smallr,pfl)
                myl = G.cons2prim(load9(u,x0,jm,k,0,NP,N), gamma,smallr,pfl)
                myr = G.cons2prim(load9(u,x0,jp,k,0,NP,N), gamma,smallr,pfl)
                mzl = G.cons2prim(load9(u,x0,j,km,0,NP,N), gamma,smallr,pfl)
                mzr = G.cons2prim(load9(u,x0,j,kp,0,NP,N), gamma,smallr,pfl)
                sx = prim_slope_v(mxl,m0,mxr); sy = prim_slope_v(myl,m0,myr); sz = prim_slope_v(mzl,m0,mzr)
                mh = G.cons2prim(G.hancock(m0,sx,sy,sz,dtdx,gamma), gamma,smallr,pfl)
                bad = (mh[1] <= smallr) | (mh[5] <= pfl)
                mhf = ntuple(v -> vifelse(bad, m0[v], mh[v]), Val(9))
                @inbounds for v in 1:9
                    vstore(mhf[v], pointer(rec, linm(x0,j,k,v,NP,N)))
                    vstore(sx[v],  pointer(rec, linm(x0,j,k,9+v,NP,N)))
                    vstore(sy[v],  pointer(rec, linm(x0,j,k,18+v,NP,N)))
                    vstore(sz[v],  pointer(rec, linm(x0,j,k,27+v,NP,N)))
                end
            end
        end
    end
end

function simd_flux!(flx, rec, p::G.Params, dt::Float32)
    N = p.N; NP = N + 2XM; gamma = p.gamma; smallr = p.smallr; pfl = G.pfloor(p)
    dx = G.dxof(p); ch = p.courant*dx/dt/Float32(G.NDIM)*p.glm_ch_scale; nx = N ÷ W
    @threads for k in 1:N
        kp = k==N ? 1 : k+1
        @inbounds for j in 1:N
            jp = j==N ? 1 : j+1
            for xc in 0:nx-1
                x0 = XM + 1 + xc*W
                mh0 = load9(rec,x0,j,k,0,NP,N)
                Lx = G.padd(mh0, load9(rec,x0,j,k,9,NP,N), 1.0f0)
                Rx = G.padd(load9(rec,x0+1,j,k,0,NP,N), load9(rec,x0+1,j,k,9,NP,N), -1.0f0)
                Fx = llf_dir_v(Lx, Rx, 1, gamma, ch, smallr, pfl)
                Ly = G.padd(mh0, load9(rec,x0,j,k,18,NP,N), 1.0f0)
                Ry = G.padd(load9(rec,x0,jp,k,0,NP,N), load9(rec,x0,jp,k,18,NP,N), -1.0f0)
                Fy = llf_dir_v(Ly, Ry, 2, gamma, ch, smallr, pfl)
                Lz = G.padd(mh0, load9(rec,x0,j,k,27,NP,N), 1.0f0)
                Rz = G.padd(load9(rec,x0,j,kp,0,NP,N), load9(rec,x0,j,kp,27,NP,N), -1.0f0)
                Fz = llf_dir_v(Lz, Rz, 3, gamma, ch, smallr, pfl)
                @inbounds for v in 1:9
                    vstore(Fx[v], pointer(flx, linm(x0,j,k,v,NP,N)))
                    vstore(Fy[v], pointer(flx, linm(x0,j,k,9+v,NP,N)))
                    vstore(Fz[v], pointer(flx, linm(x0,j,k,18+v,NP,N)))
                end
            end
        end
    end
end

function simd_update!(unew, uold, flx, p::G.Params, dt::Float32)
    N = p.N; NP = N + 2XM; gamma = p.gamma; smallr = p.smallr; pfl = G.pfloor(p)
    dx = G.dxof(p); dtdx = dt/dx; glm_fac = 1.0f0; nx = N ÷ W
    ch = p.courant*dx/dt/Float32(G.NDIM)*p.glm_ch_scale
    if p.glm_cp_coef > 0; glm_fac = exp(-(ch*ch/(p.glm_cp_coef*p.boxlen*ch))*dt); end
    @threads for k in 1:N
        km = k==1 ? N : k-1
        @inbounds for j in 1:N
            jm = j==1 ? N : j-1
            for xc in 0:nx-1
                x0 = XM + 1 + xc*W
                u0 = load9(uold,x0,j,k,0,NP,N)
                Fhx = load9(flx,x0,j,k,0,NP,N);   Flx = load9(flx,x0-1,j,k,0,NP,N)
                Fhy = load9(flx,x0,j,k,9,NP,N);   Fly = load9(flx,x0,jm,k,9,NP,N)
                Fhz = load9(flx,x0,j,k,18,NP,N);  Flz = load9(flx,x0,j,km,18,NP,N)
                r1 = ntuple(v -> u0[v] + dtdx*((Flx[v]-Fhx[v]) + (Fly[v]-Fhy[v]) + (Flz[v]-Fhz[v])), Val(9))
                emag = 0.5f0*(r1[6]*r1[6] + r1[7]*r1[7] + r1[8]*r1[8])
                bad = r1[1] < smallr; z = zero(VF); fl5 = pfl/(gamma-1.0f0) + emag
                r = (vifelse(bad,VF(smallr),r1[1]), vifelse(bad,z,r1[2]), vifelse(bad,z,r1[3]),
                     vifelse(bad,z,r1[4]), vifelse(bad,fl5,r1[5]), r1[6], r1[7], r1[8], r1[9]*glm_fac)
                @inbounds for v in 1:9
                    vstore(r[v], pointer(unew, linm(x0,j,k,v,NP,N)))
                end
            end
        end
    end
end

# refresh 1-cell x-ghost (periodic) for an (NP,N,N,NC) array
function refresh_xg!(a, N, NC)
    NP = N + 2XM
    @inbounds @threads for k in 1:N
        for c in 1:NC, j in 1:N
            a[1, j, k, c]    = a[N+1, j, k, c]
            a[N+2, j, k, c]  = a[2, j, k, c]
        end
    end
end

# NUMA first-touch alloc: each thread writes the k-planes it will later compute (same @threads
# partition), so pages land on that thread's node -> local-bandwidth access (~3x vs main-thread
# zeros() which puts everything on node 0). The single biggest CPU lever on this 16-NUMA box.
function numa_zeros(NP, N, NC)
    a = Array{Float32}(undef, NP, N, N, NC)
    @threads for k in 1:N
        @inbounds for c in 1:NC, j in 1:N, x in 1:NP; a[x,j,k,c] = 0f0; end
    end
    return a
end

function run_ot_simd_mp(; N=128, nsteps=40)
    @assert N % W == 0
    NP = N + 2XM; p = G.Params(N=N, gamma=5f0/3f0, cs0=1f0); be = G.KA.CPU()
    u0 = zeros(Float32, N,N,N,9); G.ic_ot_kernel!(be)(u0,N,p.gamma;ndrange=(N,N,N)); G.KA.synchronize(be)
    u1 = numa_zeros(NP,N,9); u2 = numa_zeros(NP,N,9)
    rec = numa_zeros(NP,N,36); flx = numa_zeros(NP,N,27); ct = zeros(Float32,N,N,N)
    @threads for k in 1:N          # first-touch the IC copy too (keep u1 NUMA-local)
        @inbounds for v in 1:9, j in 1:N, x in 1:N; u1[XM+x,j,k,v] = u0[x,j,k,v]; end
    end
    dt = 1f-3
    onestep!(a,b) = (refresh_xg!(a,N,9); simd_recon!(rec,a,p,dt/G.dxof(p)); refresh_xg!(rec,N,36);
                     simd_flux!(flx,rec,p,dt); refresh_xg!(flx,N,27); simd_update!(b,a,flx,p,dt))
    onestep!(u1,u2)  # warmup
    t=0f0; step=0; tt=0.0
    while step < nsteps
        t0=time(); onestep!(u1,u2); tt += time()-t0
        u1,u2 = u2,u1; step += 1
    end
    uint = Array(@view u1[XM+1:XM+N,:,:,:]); d = G.diagnostics(uint, ct, p, be)
    mc = N^3*step/tt/1e6; gbs = mc*1e6*153*4/1e9
    @printf("SIMD-MP-OT N=%d thr=%d: steps=%d vrms=%.4f divB=%.3e finite=%s  %.1f Mcell/s (~%.0f GB/s, %.0f%% of 410)\n",
            N, Threads.nthreads(), step, d.vrms, d.divbmax, d.finite, mc, gbs, 100gbs/410)
    return d, mc
end

# ============================================================================
# Cache-blocked recon-ONCE CPU z-stream: reconstruct one z-plane at a time into an
# L3-resident plane buffer (NOT a full-volume DRAM scratch), carry z-fluxes between planes
# (1-plane-lagged update), SIMD-across-x, NUMA-first-touched, periodic via inline mod (no
# ghost passes except 1 cheap x-column/plane). DRAM traffic ~= read u + write unew only.
# Mirrors the GPU integrator_stream!. LLF Riemann. Plane buffers are (NP,N,*) with 1 x-ghost.
# ============================================================================
@inline loadu(u, x, j, kz, NP, N) = ntuple(v -> vload(VF, pointer(u, x + NP*((j-1)+N*((kz-1)+N*(v-1))))), 9)
@inline loadp(prc, x, j, off, NP, N) = ntuple(v -> vload(VF, pointer(prc, x + NP*((j-1)+N*(off+v-1)))), 9)

# Reconstruct plane kk (0-based; z-neighbors via mod) into prc=(NP,N,36): mh,sx,sy,sz. SIMD-x.
function recon_plane!(prc, u, kk, p::G.Params, dtdx::Float32)
    N = p.N; NP = N + 2XM; gamma = p.gamma; smallr = p.smallr; pfl = G.pfloor(p); nx = N ÷ W
    kc = mod(kk,N)+1; kdn = mod(kk-1,N)+1; kup = mod(kk+1,N)+1
    @threads for j in 1:N
        jm = j==1 ? N : j-1; jp = j==N ? 1 : j+1
        @inbounds for xc in 0:nx-1
            x0 = XM + 1 + xc*W
            m0  = G.cons2prim(loadu(u,x0,  j,kc,NP,N), gamma,smallr,pfl)
            mxl = G.cons2prim(loadu(u,x0-1,j,kc,NP,N), gamma,smallr,pfl)
            mxr = G.cons2prim(loadu(u,x0+1,j,kc,NP,N), gamma,smallr,pfl)
            myl = G.cons2prim(loadu(u,x0,jm,kc,NP,N), gamma,smallr,pfl)
            myr = G.cons2prim(loadu(u,x0,jp,kc,NP,N), gamma,smallr,pfl)
            mzl = G.cons2prim(loadu(u,x0,j,kdn,NP,N), gamma,smallr,pfl)
            mzr = G.cons2prim(loadu(u,x0,j,kup,NP,N), gamma,smallr,pfl)
            sx = prim_slope_v(mxl,m0,mxr); sy = prim_slope_v(myl,m0,myr); sz = prim_slope_v(mzl,m0,mzr)
            mh = G.cons2prim(G.hancock(m0,sx,sy,sz,dtdx,gamma), gamma,smallr,pfl)
            bad = (mh[1] <= smallr) | (mh[5] <= pfl)
            mhf = ntuple(v -> vifelse(bad, m0[v], mh[v]), Val(9))
            base = NP*(j-1)
            @inbounds for v in 1:9
                vstore(mhf[v], pointer(prc, x0 + base + NP*N*(v-1)))
                vstore(sx[v],  pointer(prc, x0 + base + NP*N*(9+v-1)))
                vstore(sy[v],  pointer(prc, x0 + base + NP*N*(18+v-1)))
                vstore(sz[v],  pointer(prc, x0 + base + NP*N*(27+v-1)))
            end
        end
    end
    # refresh BOTH x-ghost columns (periodic): high for the i+1 face, low for the i-1 face
    @threads for j in 1:N
        @inbounds for c in 1:36
            prc[N+2, j, c] = prc[2, j, c]      # high ghost <- first owned
            prc[1,   j, c] = prc[N+1, j, c]    # low  ghost <- last owned
        end
    end
end

# In-plane x,y Riemann -> xy flux-divergence (XY); z-face states Mz=mh-sz, Pz=mh+sz. SIMD-x.
function faces_plane!(Mz, Pz, XY, prc, p::G.Params, ch::Float32)
    N = p.N; NP = N + 2XM; gamma = p.gamma; smallr = p.smallr; pfl = G.pfloor(p); nx = N ÷ W
    @threads for j in 1:N
        jp = j==N ? 1 : j+1
        @inbounds for xc in 0:nx-1
            x0 = XM + 1 + xc*W
            mh0 = loadp(prc,x0,j,0,NP,N); sxc = loadp(prc,x0,j,9,NP,N)
            syc = loadp(prc,x0,j,18,NP,N); szc = loadp(prc,x0,j,27,NP,N)
            Lx = G.padd(mh0, sxc, 1.0f0)
            Rx = G.padd(loadp(prc,x0+1,j,0,NP,N), loadp(prc,x0+1,j,9,NP,N), -1.0f0)
            Fxh = llf_dir_v(Lx, Rx, 1, gamma, ch, smallr, pfl)
            # low x face: L = +x edge of cell x0-1, R = -x edge of cell x0
            Lxl = G.padd(loadp(prc,x0-1,j,0,NP,N), loadp(prc,x0-1,j,9,NP,N), 1.0f0)
            Fxl = llf_dir_v(Lxl, G.padd(mh0,sxc,-1.0f0), 1, gamma, ch, smallr, pfl)
            Ly = G.padd(mh0, syc, 1.0f0)
            Ry = G.padd(loadp(prc,x0,jp,0,NP,N), loadp(prc,x0,jp,18,NP,N), -1.0f0)
            Fyh = llf_dir_v(Ly, Ry, 2, gamma, ch, smallr, pfl)
            jm = j==1 ? N : j-1
            Lyl = G.padd(loadp(prc,x0,jm,0,NP,N), loadp(prc,x0,jm,18,NP,N), 1.0f0)
            Fyl = llf_dir_v(Lyl, G.padd(mh0,syc,-1.0f0), 2, gamma, ch, smallr, pfl)
            xy = ntuple(v -> (Fxl[v]-Fxh[v]) + (Fyl[v]-Fyh[v]), Val(9))
            mz = G.padd(mh0, szc, -1.0f0); pz = G.padd(mh0, szc, 1.0f0)
            base = NP*(j-1)
            @inbounds for v in 1:9
                vstore(xy[v], pointer(XY, x0 + base + NP*N*(v-1)))
                vstore(mz[v], pointer(Mz, x0 + base + NP*N*(v-1)))
                vstore(pz[v], pointer(Pz, x0 + base + NP*N*(v-1)))
            end
        end
    end
end

# z-flux at face (kk-1/2)=riemann(prevPz, currMz); if doupd, update plane kupd. SIMD-x.
function fz_update!(unew, uold, kupd, doupd::Bool, prevPz, Mz, prevXY, prevFz, Fz,
                    p::G.Params, dtdx::Float32, ch::Float32, glm_fac::Float32)
    N = p.N; NP = N + 2XM; gamma = p.gamma; smallr = p.smallr; pfl = G.pfloor(p); nx = N ÷ W
    kz = mod(kupd,N)+1
    @threads for j in 1:N
        @inbounds for xc in 0:nx-1
            x0 = XM + 1 + xc*W; base = NP*(j-1)
            pz = ntuple(v -> vload(VF, pointer(prevPz, x0+base+NP*N*(v-1))), 9)
            mz = ntuple(v -> vload(VF, pointer(Mz,     x0+base+NP*N*(v-1))), 9)
            fz = llf_dir_v(pz, mz, 3, gamma, ch, smallr, pfl)
            @inbounds for v in 1:9; vstore(fz[v], pointer(Fz, x0+base+NP*N*(v-1))); end
            if doupd
                u0 = loadu(uold, x0, j, kz, NP, N)
                xy  = ntuple(v -> vload(VF, pointer(prevXY, x0+base+NP*N*(v-1))), 9)
                flo = ntuple(v -> vload(VF, pointer(prevFz, x0+base+NP*N*(v-1))), 9)
                r1 = ntuple(v -> u0[v] + dtdx*(xy[v] + (flo[v]-fz[v])), Val(9))
                emag = 0.5f0*(r1[6]*r1[6]+r1[7]*r1[7]+r1[8]*r1[8]); z=zero(VF); fl5=pfl/(gamma-1.0f0)+emag
                bad = r1[1] < smallr
                r = (vifelse(bad,VF(smallr),r1[1]), vifelse(bad,z,r1[2]), vifelse(bad,z,r1[3]),
                     vifelse(bad,z,r1[4]), vifelse(bad,fl5,r1[5]), r1[6], r1[7], r1[8], r1[9]*glm_fac)
                @inbounds for v in 1:9
                    vstore(r[v], pointer(unew, x0 + NP*((j-1)+N*((kz-1)+N*(v-1)))))
                end
            end
        end
    end
end

# One full z-stream step (top-level so the buffer ping-pong swaps are LOCAL vars, not a boxed
# closure capture). Buffers are scratch, re-seeded each step, so nothing carries across calls.
function stream_onestep!(ub, ua, prc, Mz, Pz, XY, Fz, prevPz, prevXY, prevFz,
                         p::G.Params, dtdx::Float32, ch::Float32, glm_fac::Float32)
    N = p.N
    @threads for k in 1:N            # periodic x-ghost of uold (1 col each side)
        @inbounds for v in 1:9, j in 1:N; ua[1,j,k,v]=ua[N+1,j,k,v]; ua[N+2,j,k,v]=ua[2,j,k,v]; end
    end
    recon_plane!(prc, ua, N-1, p, dtdx); faces_plane!(Mz, Pz, XY, prc, p, ch)
    prevPz, Pz = Pz, prevPz          # seed: prevPz holds plane N-1's +z face
    for kk in 0:N
        recon_plane!(prc, ua, kk, p, dtdx); faces_plane!(Mz, Pz, XY, prc, p, ch)
        fz_update!(ub, ua, kk-1, kk>=1, prevPz, Mz, prevXY, prevFz, Fz, p, dtdx, ch, glm_fac)
        prevPz, Pz = Pz, prevPz; prevXY, XY = XY, prevXY; prevFz, Fz = Fz, prevFz
    end
    return nothing
end

function run_ot_stream(; N=128, nsteps=50)
    @assert N % W == 0
    NP = N + 2XM; p = G.Params(N=N, gamma=5f0/3f0, cs0=1f0); be = G.KA.CPU()
    u0 = zeros(Float32, N,N,N,9); G.ic_ot_kernel!(be)(u0,N,p.gamma;ndrange=(N,N,N)); G.KA.synchronize(be)
    u1 = numa_zeros(NP,N,9); u2 = numa_zeros(NP,N,9); ct = zeros(Float32,N,N,N)
    @threads for k in 1:N
        @inbounds for v in 1:9, j in 1:N, x in 1:N; u1[XM+x,j,k,v]=u0[x,j,k,v]; end
    end
    pz9() = (a=Array{Float32}(undef,NP,N,9); @threads for j in 1:N; @inbounds for c in 1:9,x in 1:NP; a[x,j,c]=0f0; end; end; a)
    prc = (a=Array{Float32}(undef,NP,N,36); @threads for j in 1:N; @inbounds for c in 1:36,x in 1:NP; a[x,j,c]=0f0; end; end; a)
    Mz=pz9(); Pz=pz9(); XY=pz9(); Fz=pz9(); prevPz=pz9(); prevXY=pz9(); prevFz=pz9()
    dt = 1f-3; dx = G.dxof(p); dtdx = dt/dx
    ch = p.courant*dx/dt/Float32(G.NDIM)*p.glm_ch_scale
    glm_fac = p.glm_cp_coef>0 ? exp(-(ch*ch/(p.glm_cp_coef*p.boxlen*ch))*dt) : 1.0f0
    stream_onestep!(u2, u1, prc, Mz, Pz, XY, Fz, prevPz, prevXY, prevFz, p, dtdx, ch, glm_fac)  # warmup
    t0=0.0; for s in 1:nsteps; tt=time(); stream_onestep!(u2, u1, prc, Mz, Pz, XY, Fz, prevPz, prevXY, prevFz, p, dtdx, ch, glm_fac); t0+=time()-tt; u1,u2=u2,u1; end
    uint = Array(@view u1[XM+1:XM+N,:,:,:]); d = G.diagnostics(uint, ct, p, be)
    mc = N^3*nsteps/t0/1e6; gbs = mc*1e6*18*4/1e9
    @printf("STREAM-OT N=%d thr=%d: steps=%d vrms=%.4f divB=%.3e finite=%s  %.1f Mcell/s (~%.0f GB/s eff, %.0f%% of 410)\n",
            N, Threads.nthreads(), nsteps, d.vrms, d.divbmax, d.finite, mc, gbs, 100gbs/410)
    return d, mc
end

# ============================================================================
# CHUNKED z-stream: ONE @threads region per step; each thread streams a contiguous k-chunk
# sequentially (recon-once, SIMD-across-x), with thread-local plane buffers. No per-plane
# barriers (the z-stream's granularity trap). A chunk recomputes 1 halo plane each end.
# At 64^3 a plane's recon buffer (0.6 MB) fits L2 -> fast on small grids. Chunk count is
# adaptive (>=8 planes/chunk) so small grids use fewer, fatter chunks.
# ============================================================================
struct ChunkBufs
    recC::Array{Float32,3}; cMz::Array{Float32,3}; cPz::Array{Float32,3}; cXY::Array{Float32,3}
    pPz::Array{Float32,3};  pXY::Array{Float32,3}; pFz::Array{Float32,3};  Fz::Array{Float32,3}
end
@inline crange(c, nchunks, N) = (div((c-1)*N, nchunks), div(c*N, nchunks)-1)   # 0-based [c0,c1]

# reconstruct one plane kk (0-based) into recC=(NP,N,36); sequential (parallelism is at chunk level)
function recon1!(recC, u, kk::Int, p::G.Params, dtdx::Float32)
    N = p.N; NP = N + 2XM; gamma = p.gamma; smallr = p.smallr; pfl = G.pfloor(p); nx = N ÷ W
    kc = mod(kk,N)+1; kdn = mod(kk-1,N)+1; kup = mod(kk+1,N)+1
    @inbounds for j in 1:N
        jm = j==1 ? N : j-1; jp = j==N ? 1 : j+1
        for xc in 0:nx-1
            x0 = XM + 1 + xc*W
            m0  = G.cons2prim(loadu(u,x0,  j,kc,NP,N), gamma,smallr,pfl)
            mxl = G.cons2prim(loadu(u,x0-1,j,kc,NP,N), gamma,smallr,pfl)
            mxr = G.cons2prim(loadu(u,x0+1,j,kc,NP,N), gamma,smallr,pfl)
            myl = G.cons2prim(loadu(u,x0,jm,kc,NP,N), gamma,smallr,pfl)
            myr = G.cons2prim(loadu(u,x0,jp,kc,NP,N), gamma,smallr,pfl)
            mzl = G.cons2prim(loadu(u,x0,j,kdn,NP,N), gamma,smallr,pfl)
            mzr = G.cons2prim(loadu(u,x0,j,kup,NP,N), gamma,smallr,pfl)
            sx = prim_slope_v(mxl,m0,mxr); sy = prim_slope_v(myl,m0,myr); sz = prim_slope_v(mzl,m0,mzr)
            mh = G.cons2prim(G.hancock(m0,sx,sy,sz,dtdx,gamma), gamma,smallr,pfl)
            bad = (mh[1] <= smallr) | (mh[5] <= pfl)
            mhf = ntuple(v -> vifelse(bad, m0[v], mh[v]), Val(9))
            base = NP*(j-1)
            for v in 1:9
                vstore(mhf[v], pointer(recC, x0+base+NP*N*(v-1)))
                vstore(sx[v],  pointer(recC, x0+base+NP*N*(9+v-1)))
                vstore(sy[v],  pointer(recC, x0+base+NP*N*(18+v-1)))
                vstore(sz[v],  pointer(recC, x0+base+NP*N*(27+v-1)))
            end
        end
        for c in 1:36; recC[1,j,c]=recC[N+1,j,c]; recC[N+2,j,c]=recC[2,j,c]; end
    end
end

function faces1!(cMz, cPz, cXY, recC, p::G.Params, ch::Float32)
    N = p.N; NP = N + 2XM; gamma = p.gamma; smallr = p.smallr; pfl = G.pfloor(p); nx = N ÷ W
    @inbounds for j in 1:N
        jp = j==N ? 1 : j+1; jm = j==1 ? N : j-1
        for xc in 0:nx-1
            x0 = XM + 1 + xc*W
            mh0 = loadp(recC,x0,j,0,NP,N); sxc = loadp(recC,x0,j,9,NP,N)
            syc = loadp(recC,x0,j,18,NP,N); szc = loadp(recC,x0,j,27,NP,N)
            Fxh = llf_dir_v(G.padd(mh0,sxc,1.0f0), G.padd(loadp(recC,x0+1,j,0,NP,N),loadp(recC,x0+1,j,9,NP,N),-1.0f0), 1, gamma,ch,smallr,pfl)
            Fxl = llf_dir_v(G.padd(loadp(recC,x0-1,j,0,NP,N),loadp(recC,x0-1,j,9,NP,N),1.0f0), G.padd(mh0,sxc,-1.0f0), 1, gamma,ch,smallr,pfl)
            Fyh = llf_dir_v(G.padd(mh0,syc,1.0f0), G.padd(loadp(recC,x0,jp,0,NP,N),loadp(recC,x0,jp,18,NP,N),-1.0f0), 2, gamma,ch,smallr,pfl)
            Fyl = llf_dir_v(G.padd(loadp(recC,x0,jm,0,NP,N),loadp(recC,x0,jm,18,NP,N),1.0f0), G.padd(mh0,syc,-1.0f0), 2, gamma,ch,smallr,pfl)
            xy = ntuple(v -> (Fxl[v]-Fxh[v]) + (Fyl[v]-Fyh[v]), Val(9))
            mz = G.padd(mh0,szc,-1.0f0); pz = G.padd(mh0,szc,1.0f0)
            base = NP*(j-1)
            for v in 1:9
                vstore(xy[v], pointer(cXY, x0+base+NP*N*(v-1)))
                vstore(mz[v], pointer(cMz, x0+base+NP*N*(v-1)))
                vstore(pz[v], pointer(cPz, x0+base+NP*N*(v-1)))
            end
        end
    end
end

function fzupd1!(unew, uold, kupd::Int, doupd::Bool, pPz, cMz, pXY, pFz, Fz,
                 p::G.Params, dtdx::Float32, ch::Float32, glm_fac::Float32)
    N = p.N; NP = N + 2XM; gamma = p.gamma; smallr = p.smallr; pfl = G.pfloor(p); nx = N ÷ W
    kz = mod(kupd,N)+1
    @inbounds for j in 1:N
        for xc in 0:nx-1
            x0 = XM + 1 + xc*W; base = NP*(j-1)
            pz = ntuple(v -> vload(VF, pointer(pPz, x0+base+NP*N*(v-1))), Val(9))
            mz = ntuple(v -> vload(VF, pointer(cMz, x0+base+NP*N*(v-1))), Val(9))
            fz = llf_dir_v(pz, mz, 3, gamma, ch, smallr, pfl)
            for v in 1:9; vstore(fz[v], pointer(Fz, x0+base+NP*N*(v-1))); end
            if doupd
                u0 = loadu(uold, x0, j, kz, NP, N)
                xy  = ntuple(v -> vload(VF, pointer(pXY, x0+base+NP*N*(v-1))), Val(9))
                flo = ntuple(v -> vload(VF, pointer(pFz, x0+base+NP*N*(v-1))), Val(9))
                r1 = ntuple(v -> u0[v] + dtdx*(xy[v] + (flo[v]-fz[v])), Val(9))
                emag = 0.5f0*(r1[6]*r1[6]+r1[7]*r1[7]+r1[8]*r1[8]); z=zero(VF); fl5=pfl/(gamma-1.0f0)+emag
                bad = r1[1] < smallr
                r = (vifelse(bad,VF(smallr),r1[1]), vifelse(bad,z,r1[2]), vifelse(bad,z,r1[3]),
                     vifelse(bad,z,r1[4]), vifelse(bad,fl5,r1[5]), r1[6], r1[7], r1[8], r1[9]*glm_fac)
                for v in 1:9; vstore(r[v], pointer(unew, x0+NP*((j-1)+N*((kz-1)+N*(v-1))))); end
            end
        end
    end
end

function alloc_chunkbufs(nchunks, NP, N)
    bufs = Vector{ChunkBufs}(undef, nchunks)
    z9() = zeros(Float32,NP,N,9)
    @threads for c in 1:nchunks          # first-touch each chunk's buffers on its thread's node
        bufs[c] = ChunkBufs(zeros(Float32,NP,N,36), z9(),z9(),z9(), z9(),z9(),z9(), z9())
    end
    return bufs
end

function cpu_stream_step!(unew, uold, bufs, p::G.Params, dt::Float32, nchunks::Int)
    N = p.N; NP = N + 2XM; dx = G.dxof(p); dtdx = dt/dx
    ch = p.courant*dx/dt/Float32(G.NDIM)*p.glm_ch_scale
    glm_fac = p.glm_cp_coef>0 ? exp(-(ch*ch/(p.glm_cp_coef*p.boxlen*ch))*dt) : 1.0f0
    @threads for k in 1:N                # periodic x-ghost of uold (1 col each side)
        @inbounds for v in 1:9, j in 1:N; uold[1,j,k,v]=uold[N+1,j,k,v]; uold[N+2,j,k,v]=uold[2,j,k,v]; end
    end
    @threads for c in 1:nchunks
        b = bufs[c]; (c0,c1) = crange(c,nchunks,N)
        recC=b.recC; cMz=b.cMz; cPz=b.cPz; cXY=b.cXY; pPz=b.pPz; pXY=b.pXY; pFz=b.pFz; Fz=b.Fz
        for kk in (c0-1):(c1+1)
            recon1!(recC, uold, kk, p, dtdx)
            faces1!(cMz, cPz, cXY, recC, p, ch)
            fzupd1!(unew, uold, kk-1, kk>=c0+1, pPz, cMz, pXY, pFz, Fz, p, dtdx, ch, glm_fac)
            cPz,pPz = pPz,cPz; cXY,pXY = pXY,cXY; Fz,pFz = pFz,Fz   # local ping-pong (no box)
        end
    end
    return nothing
end

function run_ot_cstream(; N=128, nsteps=60, nchunks=0)
    @assert N % W == 0
    # default: as many chunks as threads (capped so each chunk owns >=2 planes). On small grids
    # more chunks wins despite higher halo-recompute, because the recompute is cache-cheap and
    # parallelism dominates (64^3: 8 chunks->30, 32 chunks->57 Mcell/s).
    nchunks == 0 && (nchunks = min(Threads.nthreads(), max(1, N ÷ 2)))
    NP = N + 2XM; p = G.Params(N=N, gamma=5f0/3f0, cs0=1f0); be = G.KA.CPU()
    u0 = zeros(Float32, N,N,N,9); G.ic_ot_kernel!(be)(u0,N,p.gamma;ndrange=(N,N,N)); G.KA.synchronize(be)
    # NUMA first-touch u1/u2 BY CHUNK (the compute partition) so each chunk's planes are node-local.
    u1 = Array{Float32}(undef,NP,N,N,9); u2 = Array{Float32}(undef,NP,N,N,9)
    @threads for c in 1:nchunks
        c0,c1 = crange(c,nchunks,N)
        @inbounds for k in (c0+1):(c1+1), v in 1:9, j in 1:N, x in 1:NP
            u1[x,j,k,v]=0f0; u2[x,j,k,v]=0f0
        end
    end
    @threads for c in 1:nchunks
        c0,c1 = crange(c,nchunks,N)
        @inbounds for k in (c0+1):(c1+1), v in 1:9, j in 1:N, x in 1:N; u1[XM+x,j,k,v]=u0[x,j,k,v]; end
    end
    ct = zeros(Float32,N,N,N)
    bufs = alloc_chunkbufs(nchunks, NP, N)
    dt = 1f-3
    cpu_stream_step!(u2, u1, bufs, p, dt, nchunks); u1,u2 = u2,u1   # warmup
    t0=0.0; for s in 1:nsteps; tt=time(); cpu_stream_step!(u2,u1,bufs,p,dt,nchunks); t0+=time()-tt; u1,u2=u2,u1; end
    uint = Array(@view u1[XM+1:XM+N,:,:,:]); d = G.diagnostics(uint, ct, p, be)
    mc = N^3*nsteps/t0/1e6; gbs = mc*1e6*18*4/1e9
    @printf("CSTREAM-OT N=%d thr=%d chunks=%d: vrms=%.4f divB=%.3e finite=%s  %.1f Mcell/s (~%.0f GB/s, %.0f%% of 410)\n",
            N, Threads.nthreads(), nchunks, d.vrms, d.divbmax, d.finite, mc, gbs, 100gbs/410)
    return d, mc
end
