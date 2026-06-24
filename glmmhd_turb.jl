# glmmhd_turb.jl — single-file, GPU-resident GLM-MHD driven-turbulence solver.
#
# Reproduces the physics of the reference CUDA-Fortran solver (mini-ramses-metal,
# branch cuda-dedner-mhd, gpu/gpu_hydro.cuf GLMMHD path): 9-var GLM-MHD, PLM (MonCen)
# + MUSCL-Hancock predictor + HLLD Riemann + Dedner cleaning, near-isothermal gamma=1.01,
# driven turbulence (Mach~10, beta~2) with OU forcing on a 64^3 Fourier grid (cuFFT).
# Dense uniform periodic grid, fp32, everything on the GPU.
#
# Backends: KernelAbstractions (portable) for the simple kernels; the fused integrator
# is KA by default with a native CUDA.jl @cuda twin (USE_NATIVE_INTEGRATOR=true) for the
# performance fallback. Target: >=750 Mcell/s and <=72 B/cell at 512^3 on an RTX A6000.

module GLMMHDTurb

using CUDA
using KernelAbstractions
const KA = KernelAbstractions
using AbstractFFTs
using FFTW                      # CPU inverse FFT for the OU forcing (AbstractFFTs dispatches
                                # to CUFFT for CuArray, FFTW for Array — same call site)
import ThreadPinning            # CPU NUMA thread pinning (run_turb/run_ot backend=:cpu)
using Printf
using Random

# ============================================================================
# Parameters (all Float32 on the hot path)
# ============================================================================
Base.@kwdef struct Params
    N::Int            = 256          # cells per dim (uniform NxNxN)
    boxlen::Float32   = 1.0f0
    gamma::Float32    = 1.01f0       # near-isothermal
    courant::Float32  = 0.7f0
    smallr::Float32   = 1.0f-6
    smallc::Float32   = 1.0f-4
    # GLM / Dedner cleaning
    glm_ch_scale::Float32 = 0.25f0
    glm_cp_coef::Float32  = 0.18f0
    # LLF robustness switch (<=0 disables)
    switch_llf_dmin::Float32 = 0.0f0
    switch_llf_pmin::Float32 = 0.0f0
    # initial state (beta~2 Mach-10 turbulence box: rho0=1, cs=1 -> p0=rho0*cs^2/gamma, Bx0=1)
    rho0::Float32 = 1.0f0
    cs0::Float32  = 1.0f0
    b0::Float32   = 1.0f0
    # turbulent forcing (OU)
    turb_rms::Float32  = 130.0f0     # forcing amplitude (calibrated -> Mach~10 with cs=1)
    turb_T::Float32    = 0.3f0       # OU autocorrelation time
    turb_Ndt::Int      = 100         # OU substeps per turb_T
    comp_frac::Float32 = 0.5f0       # compressive fraction (Helmholtz)
    turb_min_rho::Float32 = 1.0f-5
    turb_seed::UInt32  = 0x00000001
end

@inline dxof(p::Params) = p.boxlen / p.N
@inline pfloor(p::Params) = p.smallr * p.smallc^2 / p.gamma

const NV = 9                          # conserved vars
const TURB_GS = 64                    # OU Fourier grid size
const NDIM = 3

# ============================================================================
# Device physics — @inline, fp32, operate on NTuple{9,Float32} states.
# Conserved order : (rho, rho*vx, rho*vy, rho*vz, E, Bx, By, Bz, psi)
# Primitive order : (rho, vx, vy, vz, p, Bx, By, Bz, psi)
# Faithful ports of gpu/gpu_hydro.cuf.
# ============================================================================

@fastmath @inline function cons2prim(c::NTuple{9}, gamma::Float32, smallr::Float32, pfl::Float32)
    rho = max(c[1], smallr)
    ir  = 1.0f0 / rho
    vx = c[2]*ir; vy = c[3]*ir; vz = c[4]*ir
    bx = c[6]; by = c[7]; bz = c[8]
    ekin = 0.5f0*(c[2]*c[2] + c[3]*c[3] + c[4]*c[4])*ir
    emag = 0.5f0*(bx*bx + by*by + bz*bz)
    p = max((gamma - 1.0f0)*(c[5] - ekin - emag), pfl)
    return (rho, vx, vy, vz, p, bx, by, bz, c[9])
end

@inline function prim2cons(q::NTuple{9}, gamma::Float32)
    rho = q[1]; vx = q[2]; vy = q[3]; vz = q[4]; p = q[5]
    bx = q[6]; by = q[7]; bz = q[8]
    ekin = 0.5f0*rho*(vx*vx + vy*vy + vz*vz)
    emag = 0.5f0*(bx*bx + by*by + bz*bz)
    E = p/(gamma - 1.0f0) + ekin + emag
    return (rho, rho*vx, rho*vy, rho*vz, E, bx, by, bz, q[9])
end

# Fast magnetosonic speed for normal field component bn (prim state).
@fastmath @inline function fast_speed(q::NTuple{9}, gamma::Float32, bn)
    rho = q[1]
    c2 = gamma * q[5] / rho
    b2 = (q[6]*q[6] + q[7]*q[7] + q[8]*q[8]) / rho
    d2 = 0.5f0*(b2 + c2)
    return sqrt(d2 + sqrt(max(d2*d2 - c2*bn*bn/rho, 0.0f0)))
end

# Ideal-MHD physical flux in the x-direction (prim state); Bx/psi entries set by glm_pair.
@inline function phys_flux_x(q::NTuple{9}, gamma::Float32)
    rho = q[1]; vx = q[2]; vy = q[3]; vz = q[4]; p = q[5]
    bx = q[6]; by = q[7]; bz = q[8]
    b2 = bx*bx + by*by + bz*bz
    ptot = p + 0.5f0*b2
    ekin = 0.5f0*rho*(vx*vx + vy*vy + vz*vz)
    E = p/(gamma - 1.0f0) + ekin + 0.5f0*b2
    vdotb = vx*bx + vy*by + vz*bz
    return (rho*vx,
            rho*vx*vx + ptot - bx*bx,
            rho*vx*vy - bx*by,
            rho*vx*vz - bx*bz,
            (E + ptot)*vx - bx*vdotb,
            zero(rho),              # F[Bx] -> set by glm_pair
            vx*by - vy*bx,
            vx*bz - vz*bx,
            zero(rho))              # F[psi] -> set by glm_pair
end

# Rotate a state so direction `dir` becomes the x-normal (1=x identity, 2=y, 3=z).
# Cyclic permutation of (vx,vy,vz) and (Bx,By,Bz).
@inline function rot_to(q::NTuple{9}, dir::Int)
    if dir == 1
        return q
    elseif dir == 2          # y -> x : (x,y,z) -> (y,z,x)
        return (q[1], q[3], q[4], q[2], q[5], q[7], q[8], q[6], q[9])
    else                     # z -> x : (x,y,z) -> (z,x,y)
        return (q[1], q[4], q[2], q[3], q[5], q[8], q[6], q[7], q[9])
    end
end

# Rotate a flux (conserved-component vector) from the x-normal frame back to `dir`.
@inline function rot_flux_from(f::NTuple{9}, dir::Int)
    if dir == 1
        return f
    elseif dir == 2          # inverse of (x,y,z)->(y,z,x) is (y,z,x)->? : place back
        return (f[1], f[4], f[2], f[3], f[5], f[8], f[6], f[7], f[9])
    else
        return (f[1], f[3], f[4], f[2], f[5], f[7], f[8], f[6], f[9])
    end
end

# MonCen slope limiter (slope_type=2, factor 2), per the reference slope_moncen.
# @fastmath is ESSENTIAL: Julia's plain min/max carry NaN-propagation (extra compares);
# without it, the 27 min/max/abs per cell in the trace cost ~65% of the whole step.
# @fastmath lowers them to the hardware fmin/fmax intrinsics. Branchless (ifelse->select)
# also avoids warp divergence; equivalent to sgn*min(2min|dl|,|dr|,|dc|).
@fastmath @inline function moncen(dl, dr)
    dc  = 0.5f0*(dl + dr)
    sgn = ifelse(dc >= 0.0f0, 1.0f0, -1.0f0)
    val = sgn * min(2.0f0*min(abs(dl), abs(dr)), abs(dc))
    return ifelse(dl*dr <= 0.0f0, 0.0f0, val)
end

# Half-slope of a primitive state vector from (L, M, R) neighbours.
@fastmath @inline function prim_slope(L::NTuple{9}, M::NTuple{9}, R::NTuple{9})
    return ntuple(i -> 0.5f0*moncen(M[i]-L[i], R[i]-M[i]), 9)
end

@inline padd(q::NTuple{9}, s::NTuple{9}, a::Float32) =
    ntuple(i -> q[i] + a*s[i], 9)

# Directional ideal-MHD flux of a primitive state in direction `dir` (rotate, flux, rotate back).
@inline dir_flux(q::NTuple{9}, dir::Int, gamma::Float32) =
    rot_flux_from(phys_flux_x(rot_to(q, dir), gamma), dir)

# MUSCL-Hancock half-step predictor: u0 += -0.5*dtdx*(F(m0+s_d) - F(m0-s_d)) over 3 dirs.
@fastmath @inline function hancock(m0::NTuple{9}, sx, sy, sz, dtdx::Float32, gamma::Float32)
    u0 = prim2cons(m0, gamma)
    h = 0.5f0*dtdx
    fxp = dir_flux(padd(m0, sx, 1.0f0), 1, gamma); fxm = dir_flux(padd(m0, sx, -1.0f0), 1, gamma)
    fyp = dir_flux(padd(m0, sy, 1.0f0), 2, gamma); fym = dir_flux(padd(m0, sy, -1.0f0), 2, gamma)
    fzp = dir_flux(padd(m0, sz, 1.0f0), 3, gamma); fzm = dir_flux(padd(m0, sz, -1.0f0), 3, gamma)
    return ntuple(i -> u0[i] - h*((fxp[i]-fxm[i]) + (fyp[i]-fym[i]) + (fzp[i]-fzm[i])), 9)
end

# Dedner GLM pair: clean the normal field; returns (bn*, psi*, F[bn]=psi*, F[psi]=ch^2*bn*).
@inline function glm_pair(bnL, bnR, psiL, psiR, ch::Float32)
    bns  = 0.5f0*(bnL + bnR) - 0.5f0*(psiR - psiL)/ch
    psis = 0.5f0*(psiL + psiR) - 0.5f0*ch*(bnR - bnL)
    return bns, psis
end

# LLF (Rusanov) flux in the x-normal frame for primitive L,R (Bx already = cleaned bn*).
@fastmath @inline function llf_x(L::NTuple{9}, R::NTuple{9}, gamma::Float32, ch::Float32,
                       fbn, fpsi)
    fL = phys_flux_x(L, gamma); fR = phys_flux_x(R, gamma)
    uL = prim2cons(L, gamma);   uR = prim2cons(R, gamma)
    smax = max(abs(L[2]) + fast_speed(L, gamma, L[6]),
               abs(R[2]) + fast_speed(R, gamma, R[6]), ch)
    f = ntuple(i -> 0.5f0*(fL[i] + fR[i]) - 0.5f0*smax*(uR[i] - uL[i]), 9)
    return (f[1], f[2], f[3], f[4], f[5], fbn, f[7], f[8], fpsi)
end

# HLLD (Miyoshi & Kusano 2005) flux in the x-normal frame. L,R primitive with Bx=bn* (cleaned).
@fastmath @inline function hlld_x(L::NTuple{9}, R::NTuple{9}, gamma::Float32, ch::Float32,
                        bn::Float32, fbn::Float32, fpsi::Float32)
    dL,uL,vL,wL,pL = L[1],L[2],L[3],L[4],L[5]; byL,bzL = L[7],L[8]
    dR,uR,vR,wR,pR = R[1],R[2],R[3],R[4],R[5]; byR,bzR = R[7],R[8]
    b2L = bn*bn + byL*byL + bzL*bzL
    b2R = bn*bn + byR*byR + bzR*bzR
    ptL = pL + 0.5f0*b2L
    ptR = pR + 0.5f0*b2R
    EL = pL/(gamma-1.0f0) + 0.5f0*dL*(uL*uL+vL*vL+wL*wL) + 0.5f0*b2L
    ER = pR/(gamma-1.0f0) + 0.5f0*dR*(uR*uR+vR*vR+wR*wR) + 0.5f0*b2R
    cfL = fast_speed(L, gamma, bn)
    cfR = fast_speed(R, gamma, bn)
    SL = min(min(uL, uR) - max(cfL, cfR), 0.0f0)
    SR = max(max(uL, uR) + max(cfL, cfR), 0.0f0)
    # conserved L,R (with cleaned bn) and their x-fluxes
    UL = (dL, dL*uL, dL*vL, dL*wL, EL, bn, byL, bzL, 0.0f0)
    UR = (dR, dR*uR, dR*vR, dR*wR, ER, bn, byR, bzR, 0.0f0)
    FL = phys_flux_x((dL,uL,vL,wL,pL,bn,byL,bzL,0.0f0), gamma)
    FR = phys_flux_x((dR,uR,vR,wR,pR,bn,byR,bzR,0.0f0), gamma)
    # contact speed SM and star total pressure
    den = (SR - uR)*dR - (SL - uL)*dL
    SM  = ((SR - uR)*dR*uR - (SL - uL)*dL*uL - ptR + ptL) / den
    pts = ((SR - uR)*dR*ptL - (SL - uL)*dL*ptR +
           dL*dR*(SR - uR)*(SL - uL)*(uR - uL)) / den
    # star (single) states L*, R*
    dLs = max(dL*(SL - uL)/(SL - SM), 1.0f-12)
    dRs = max(dR*(SR - uR)/(SR - SM), 1.0f-12)
    sqdLs = sqrt(dLs); sqdRs = sqrt(dRs)
    SLs = SM - abs(bn)/sqdLs        # Alfven speeds
    SRs = SM + abs(bn)/sqdRs
    absbn = abs(bn)
    # transverse fields/velocities in L*, R* (Miyoshi-Kusano eq. 42-47)
    function star(d, u, v, w, by, bz, pt, S, dstar)
        denK = d*(S-u)*(S-SM) - bn*bn
        small = abs(denK) < 1.0f-12
        if small
            vy = v; vz = w; byS = by; bzS = bz
        else
            fac = bn*(SM - u)/denK
            vy = v - by*fac
            vz = w - bz*fac
            byS = by*(d*(S-u)*(S-u) - bn*bn)/denK
            bzS = bz*(d*(S-u)*(S-u) - bn*bn)/denK
        end
        return vy, vz, byS, bzS
    end
    vyLs, vzLs, byLs, bzLs = star(dL,uL,vL,wL,byL,bzL,ptL,SL,dLs)
    vyRs, vzRs, byRs, bzRs = star(dR,uR,vR,wR,byR,bzR,ptR,SR,dRs)
    vdotbL  = uL*bn + vL*byL + wL*bzL
    vdotbLs = SM*bn + vyLs*byLs + vzLs*bzLs
    ELs = ((SL - uL)*EL - ptL*uL + pts*SM + bn*(vdotbL - vdotbLs))/(SL - SM)
    vdotbR  = uR*bn + vR*byR + wR*bzR
    vdotbRs = SM*bn + vyRs*byRs + vzRs*bzRs
    ERs = ((SR - uR)*ER - ptR*uR + pts*SM + bn*(vdotbR - vdotbRs))/(SR - SM)
    ULs = (dLs, dLs*SM, dLs*vyLs, dLs*vzLs, ELs, bn, byLs, bzLs, 0.0f0)
    URs = (dRs, dRs*SM, dRs*vyRs, dRs*vzRs, ERs, bn, byRs, bzRs, 0.0f0)
    # double-star state (between Alfven waves)
    sgn = bn >= 0.0f0 ? 1.0f0 : -1.0f0
    invsum = 1.0f0/(sqdLs + sqdRs)
    vyss = (sqdLs*vyLs + sqdRs*vyRs + (byRs - byLs)*sgn)*invsum
    vzss = (sqdLs*vzLs + sqdRs*vzRs + (bzRs - bzLs)*sgn)*invsum
    byss = (sqdLs*byRs + sqdRs*byLs + sqdLs*sqdRs*(vyRs - vyLs)*sgn)*invsum
    bzss = (sqdLs*bzRs + sqdRs*bzLs + sqdLs*sqdRs*(vzRs - vzLs)*sgn)*invsum
    vdotbss = SM*bn + vyss*byss + vzss*bzss
    ELss = ELs - sqdLs*(vdotbLs - vdotbss)*sgn
    ERss = ERs + sqdRs*(vdotbRs - vdotbss)*sgn
    ULss = (dLs, dLs*SM, dLs*vyss, dLs*vzss, ELss, bn, byss, bzss, 0.0f0)
    URss = (dRs, dRs*SM, dRs*vyss, dRs*vzss, ERss, bn, byss, bzss, 0.0f0)
    # sample the state at x/t = 0 (SL<=0<=SR by construction)
    if SLs >= 0.0f0
        f = ntuple(i -> FL[i] + SL*(ULs[i] - UL[i]), 9)
    elseif SM >= 0.0f0
        f = ntuple(i -> FL[i] + SLs*ULss[i] - (SLs - SL)*ULs[i] - SL*UL[i], 9)
    elseif SRs >= 0.0f0
        f = ntuple(i -> FR[i] + SRs*URss[i] - (SRs - SR)*URs[i] - SR*UR[i], 9)
    else
        f = ntuple(i -> FR[i] + SR*(URs[i] - UR[i]), 9)
    end
    return (f[1], f[2], f[3], f[4], f[5], fbn, f[7], f[8], fpsi)
end

# Riemann dispatch in direction `dir`: rotate L,R to x-normal, clean GLM, HLLD (LLF fallback), rotate back.
@fastmath @inline function riemann(Lq::NTuple{9}, Rq::NTuple{9}, dir::Int,
                         gamma::Float32, ch::Float32, smallr::Float32, pfl::Float32,
                         llf_dmin::Float32, llf_pmin::Float32, use_hlld::Bool)
    L0 = rot_to(Lq, dir); R0 = rot_to(Rq, dir)
    # Floor the reconstructed interface states (PLM can overshoot to negative rho/p in voids).
    L = (max(L0[1],smallr), L0[2],L0[3],L0[4], max(L0[5],pfl), L0[6],L0[7],L0[8],L0[9])
    R = (max(R0[1],smallr), R0[2],R0[3],R0[4], max(R0[5],pfl), R0[6],R0[7],R0[8],R0[9])
    bns, psis = glm_pair(L[6], R[6], L[9], R[9], ch)
    fbn  = psis
    fpsi = ch*ch*bns
    Lc = (L[1],L[2],L[3],L[4],L[5],bns,L[7],L[8],L[9])
    Rc = (R[1],R[2],R[3],R[4],R[5],bns,R[7],R[8],R[9])
    uself = (llf_dmin > 0.0f0 && min(L[1],R[1]) < llf_dmin) ||
            (llf_pmin > 0.0f0 && min(L[5],R[5]) < llf_pmin) || !use_hlld
    f = uself ? llf_x(Lc, Rc, gamma, ch, fbn, fpsi) :
                hlld_x(Lc, Rc, gamma, ch, bns, fbn, fpsi)
    return rot_flux_from(f, dir)
end

# ============================================================================
# LEAN 2nd-order path: minmod slopes + LLF with an UPPER-BOUND wave speed (one sqrt, no
# discriminant) — no HLLD in the call tree. Register-light → higher occupancy. Still formally
# 2nd order (PLM space + flux-Hancock time). Selected at compile time via Val{LEAN} so the lean
# kernel specialization literally elides HLLD's ~30 temps. (Cf. the RAMSES slope_type=1 +
# riemann=llf config — more diffusive, much cheaper.)
@fastmath @inline minmod(a::Float32, b::Float32) =
    ifelse(a*b <= 0.0f0, 0.0f0, ifelse(abs(a) < abs(b), a, b))
@fastmath @inline prim_slope_mm(L::NTuple{9}, M::NTuple{9}, R::NTuple{9}) =
    ntuple(i -> 0.5f0*minmod(M[i]-L[i], R[i]-M[i]), 9)

# Upper bound on the fast magnetosonic speed: cf <= sqrt(a^2 + b^2/rho). One sqrt; LLF only
# needs an over-estimate of the max signal speed (the slack is extra numerical diffusion).
@fastmath @inline function fast_speed_bound(q::NTuple{9}, gamma::Float32)
    ir = 1.0f0/q[1]
    return sqrt((gamma*q[5] + q[6]*q[6] + q[7]*q[7] + q[8]*q[8])*ir)
end

@fastmath @inline function llf_bound(L::NTuple{9}, R::NTuple{9}, gamma::Float32, ch::Float32, fbn, fpsi)
    fL = phys_flux_x(L, gamma); fR = phys_flux_x(R, gamma)
    uL = prim2cons(L, gamma);   uR = prim2cons(R, gamma)
    smax = max(abs(L[2]) + fast_speed_bound(L, gamma),
               abs(R[2]) + fast_speed_bound(R, gamma), ch)
    f = ntuple(i -> 0.5f0*(fL[i] + fR[i]) - 0.5f0*smax*(uR[i] - uL[i]), 9)
    return (f[1], f[2], f[3], f[4], f[5], fbn, f[7], f[8], fpsi)
end

@fastmath @inline function riemann_lean(Lq::NTuple{9}, Rq::NTuple{9}, dir::Int,
                                        gamma::Float32, ch::Float32, smallr::Float32, pfl::Float32)
    L0 = rot_to(Lq, dir); R0 = rot_to(Rq, dir)
    L = (max(L0[1],smallr), L0[2],L0[3],L0[4], max(L0[5],pfl), L0[6],L0[7],L0[8],L0[9])
    R = (max(R0[1],smallr), R0[2],R0[3],R0[4], max(R0[5],pfl), R0[6],R0[7],R0[8],R0[9])
    bns, psis = glm_pair(L[6], R[6], L[9], R[9], ch)
    Lc = (L[1],L[2],L[3],L[4],L[5],bns,L[7],L[8],L[9])
    Rc = (R[1],R[2],R[3],R[4],R[5],bns,R[7],R[8],R[9])
    f = llf_bound(Lc, Rc, gamma, ch, psis, ch*ch*bns)
    return rot_flux_from(f, dir)
end

# GLM-MHD HLL (2-wave): signal speeds from the exact fast magnetosonic speed (Bx=bn* cleaned),
# branchless (SL clamped <=0, SR >=0). Less diffusive than LLF, far cheaper than HLLD.
@fastmath @inline function hll_x(L::NTuple{9}, R::NTuple{9}, gamma::Float32, bn::Float32, fbn, fpsi)
    cfL = fast_speed(L, gamma, bn); cfR = fast_speed(R, gamma, bn)
    SL = min(min(L[2]-cfL, R[2]-cfR), 0.0f0); SR = max(max(L[2]+cfL, R[2]+cfR), 0.0f0)
    fL = phys_flux_x(L, gamma); fR = phys_flux_x(R, gamma)
    uL = prim2cons(L, gamma);   uR = prim2cons(R, gamma)
    ihd = 1.0f0/(SR - SL)
    f = ntuple(i -> (SR*fL[i] - SL*fR[i] + SL*SR*(uR[i]-uL[i]))*ihd, 9)
    return (f[1],f[2],f[3],f[4],f[5], fbn, f[7], f[8], fpsi)
end
@fastmath @inline function riemann_hll(Lq::NTuple{9}, Rq::NTuple{9}, dir::Int,
                                       gamma::Float32, ch::Float32, smallr::Float32, pfl::Float32)
    L0 = rot_to(Lq, dir); R0 = rot_to(Rq, dir)
    L = (max(L0[1],smallr), L0[2],L0[3],L0[4], max(L0[5],pfl), L0[6],L0[7],L0[8],L0[9])
    R = (max(R0[1],smallr), R0[2],R0[3],R0[4], max(R0[5],pfl), R0[6],R0[7],R0[8],R0[9])
    bns, psis = glm_pair(L[6], R[6], L[9], R[9], ch)
    Lc = (L[1],L[2],L[3],L[4],L[5],bns,L[7],L[8],L[9])
    Rc = (R[1],R[2],R[3],R[4],R[5],bns,R[7],R[8],R[9])
    f = hll_x(Lc, Rc, gamma, bns, psis, ch*ch*bns)
    return rot_flux_from(f, dir)
end

# 3-way compile-time Riemann selector for the parametric PLM kernel.
@inline riemann_sel(::Val{:hll},  Lq,Rq,dir,gamma,ch,smallr,pfl,ld,lp) = riemann_hll(Lq,Rq,dir,gamma,ch,smallr,pfl)
@inline riemann_sel(::Val{:llf},  Lq,Rq,dir,gamma,ch,smallr,pfl,ld,lp) = riemann_lean(Lq,Rq,dir,gamma,ch,smallr,pfl)
@inline riemann_sel(::Val{:hlld}, Lq,Rq,dir,gamma,ch,smallr,pfl,ld,lp) = riemann(Lq,Rq,dir,gamma,ch,smallr,pfl,ld,lp,true)

# Compile-time dispatch: Val{true}=lean (minmod+LLF-bound), Val{false}=full (MonCen+HLLD).
@inline slope_d(::Val{true},  L, M, R) = prim_slope_mm(L, M, R)
@inline slope_d(::Val{false}, L, M, R) = prim_slope(L, M, R)
@inline riemann_d(::Val{true},  Lq, Rq, dir, gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld) =
    riemann_lean(Lq, Rq, dir, gamma, ch, smallr, pfl)
@inline riemann_d(::Val{false}, Lq, Rq, dir, gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld) =
    riemann(Lq, Rq, dir, gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld)

# ============================================================================
# Grid helpers (dense (N,N,N,9), periodic) + multi-kernel integrator (validation path).
# This path is correct and memory-clean (no recompute) but uses scratch arrays rec/flx;
# the fused shared-memory kernel (perf/512^3 path) is added separately.
# ============================================================================
@inline wrp(i::Int, N::Int) = i < 1 ? i + N : (i > N ? i - N : i)
@inline loadc(u, i, j, k) = ntuple(v -> @inbounds(u[i, j, k, v]), 9)

# Phase 1: primitive, MonCen slopes (3 dirs), MUSCL-Hancock predictor -> rec[*, 1:9]=mh, 10:18=sx, 19:27=sy, 28:36=sz
@kernel function recon_kernel!(rec, @Const(u), gamma::Float32, smallr::Float32, pfl::Float32, dtdx::Float32, N::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        m0 = cons2prim(loadc(u, i, j, k), gamma, smallr, pfl)
        mxl = cons2prim(loadc(u, wrp(i-1,N), j, k), gamma, smallr, pfl)
        mxr = cons2prim(loadc(u, wrp(i+1,N), j, k), gamma, smallr, pfl)
        myl = cons2prim(loadc(u, i, wrp(j-1,N), k), gamma, smallr, pfl)
        myr = cons2prim(loadc(u, i, wrp(j+1,N), k), gamma, smallr, pfl)
        mzl = cons2prim(loadc(u, i, j, wrp(k-1,N)), gamma, smallr, pfl)
        mzr = cons2prim(loadc(u, i, j, wrp(k+1,N)), gamma, smallr, pfl)
        sx = prim_slope(mxl, m0, mxr)
        sy = prim_slope(myl, m0, myr)
        sz = prim_slope(mzl, m0, mzr)
        uh = hancock(m0, sx, sy, sz, dtdx, gamma)
        mh = cons2prim(uh, gamma, smallr, pfl)
        if mh[1] <= smallr || mh[5] <= pfl
            mh = m0
        end
        for v in 1:9
            rec[i, j, k, v]      = mh[v]
            rec[i, j, k, 9+v]    = sx[v]
            rec[i, j, k, 18+v]   = sy[v]
            rec[i, j, k, 27+v]   = sz[v]
        end
    end
end

@inline rec_mh(rec, i, j, k) = ntuple(v -> @inbounds(rec[i, j, k, v]), 9)
@inline rec_s(rec, i, j, k, off) = ntuple(v -> @inbounds(rec[i, j, k, off+v]), 9)

# Phase 2: face fluxes. flx[i,j,k, dir-block] = F at cell i's +dir face (i+1/2 etc).
@kernel function flux_kernel!(flx, @Const(rec), gamma::Float32, ch::Float32, smallr::Float32, pfl::Float32,
                              llf_dmin::Float32, llf_pmin::Float32, use_hlld::Bool, N::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        mh0 = rec_mh(rec, i, j, k)
        # x face (i+1/2): L = +x edge of cell i, R = -x edge of cell i+1
        ip = wrp(i+1, N); jp = wrp(j+1, N); kp = wrp(k+1, N)
        Lx = padd(mh0, rec_s(rec,i,j,k,9),  1.0f0)
        Rx = padd(rec_mh(rec,ip,j,k), rec_s(rec,ip,j,k,9), -1.0f0)
        Fx = riemann(Lx, Rx, 1, gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld)
        Ly = padd(mh0, rec_s(rec,i,j,k,18), 1.0f0)
        Ry = padd(rec_mh(rec,i,jp,k), rec_s(rec,i,jp,k,18), -1.0f0)
        Fy = riemann(Ly, Ry, 2, gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld)
        Lz = padd(mh0, rec_s(rec,i,j,k,27), 1.0f0)
        Rz = padd(rec_mh(rec,i,j,kp), rec_s(rec,i,j,kp,27), -1.0f0)
        Fz = riemann(Lz, Rz, 3, gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld)
        for v in 1:9
            flx[i, j, k, v]    = Fx[v]
            flx[i, j, k, 9+v]  = Fy[v]
            flx[i, j, k, 18+v] = Fz[v]
        end
    end
end

@inline flx_blk(flx, i, j, k, off) = ntuple(v -> @inbounds(flx[i, j, k, off+v]), 9)

# Phase 3: conservative update unew = uold + dtdx*sum_dir(F_lo - F_hi); psi *= glm_fac; fused turb.
@kernel function update_kernel!(unew, @Const(uold), @Const(flx), @Const(afield),
                                dtdx::Float32, glm_fac::Float32, N::Int, boxlen::Float32,
                                ramp::Float32, dteff::Float32, smallr::Float32, smallc::Float32,
                                gamma::Float32, pfl::Float32, turb_min_rho::Float32, do_turb::Bool)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        u0 = loadc(uold, i, j, k)
        im = wrp(i-1, N); jm = wrp(j-1, N); km = wrp(k-1, N)
        Fhi_x = flx_blk(flx, i, j, k, 0);  Flo_x = flx_blk(flx, im, j, k, 0)
        Fhi_y = flx_blk(flx, i, j, k, 9);  Flo_y = flx_blk(flx, i, jm, k, 9)
        Fhi_z = flx_blk(flx, i, j, k, 18); Flo_z = flx_blk(flx, i, j, km, 18)
        r1 = ntuple(v -> u0[v] + dtdx*((Flo_x[v]-Fhi_x[v]) + (Flo_y[v]-Fhi_y[v]) + (Flo_z[v]-Fhi_z[v])), 9)
        # positivity floor: a void cell (rho<smallr) -> static floored state with B preserved.
        # Prevents negative density / runaway v=mom/rho that collapses the CFL dt at high Mach.
        if r1[1] < smallr
            emag = 0.5f0*(r1[6]*r1[6] + r1[7]*r1[7] + r1[8]*r1[8])
            r1 = (smallr, 0.0f0, 0.0f0, 0.0f0, pfl/(gamma-1.0f0) + emag, r1[6], r1[7], r1[8], r1[9])
        end
        # operator-split GLM psi damping
        r = (r1[1],r1[2],r1[3],r1[4],r1[5],r1[6],r1[7],r1[8], r1[9]*glm_fac)
        # fused turbulent driving (trilinear interp + remove-KE / kick / restore-E)
        if do_turb
            rho = r[1]
            if rho >= turb_min_rho
                dx = boxlen / N
                ax, ay, az = turb_interp(afield, i, j, k, dx, boxlen, ramp)
                rhom = max(rho, smallr)
                e = r[5]
                e = max(e - 0.5f0*r[2]*r[2]/rhom, rho*smallc*smallc)
                e = max(e - 0.5f0*r[3]*r[3]/rhom, rho*smallc*smallc)
                e = max(e - 0.5f0*r[4]*r[4]/rhom, rho*smallc*smallc)
                m2 = r[2] + rhom*ax*dteff
                m3 = r[3] + rhom*ay*dteff
                m4 = r[4] + rhom*az*dteff
                e = max(e + 0.5f0*m2*m2/rhom, rho*smallc*smallc)
                e = max(e + 0.5f0*m3*m3/rhom, rho*smallc*smallc)
                e = max(e + 0.5f0*m4*m4/rhom, rho*smallc*smallc)
                r = (r[1], m2, m3, m4, e, r[6], r[7], r[8], r[9])
            end
        end
        for v in 1:9
            unew[i, j, k, v] = r[v]
        end
    end
end

# Trilinear interpolation of the 3-component accel field afield (3,TURB_GS,TURB_GS,TURB_GS), periodic.
@inline function turb_interp(afield, i::Int, j::Int, k::Int, dx::Float32, boxlen::Float32, ramp::Float32)
    g = Float32(TURB_GS)
    # cell-centre physical position -> turb-grid coordinate
    cx = (i - 0.5f0)*dx / boxlen * g
    cy = (j - 0.5f0)*dx / boxlen * g
    cz = (k - 0.5f0)*dx / boxlen * g
    @inbounds begin
        i0 = floor(Int, cx); j0 = floor(Int, cy); k0 = floor(Int, cz)
        wx = cx - i0; wy = cy - j0; wz = cz - k0
        i0w = mod(i0, TURB_GS)+1; i1w = mod(i0+1, TURB_GS)+1
        j0w = mod(j0, TURB_GS)+1; j1w = mod(j0+1, TURB_GS)+1
        k0w = mod(k0, TURB_GS)+1; k1w = mod(k0+1, TURB_GS)+1
        function trilin(c)
            a000=afield[c,i0w,j0w,k0w]; a100=afield[c,i1w,j0w,k0w]
            a010=afield[c,i0w,j1w,k0w]; a110=afield[c,i1w,j1w,k0w]
            a001=afield[c,i0w,j0w,k1w]; a101=afield[c,i1w,j0w,k1w]
            a011=afield[c,i0w,j1w,k1w]; a111=afield[c,i1w,j1w,k1w]
            return (((a000*(1-wx)+a100*wx)*(1-wy) + (a010*(1-wx)+a110*wx)*wy)*(1-wz) +
                    ((a001*(1-wx)+a101*wx)*(1-wy) + (a011*(1-wx)+a111*wx)*wy)*wz)
        end
        return ramp*trilin(1), ramp*trilin(2), ramp*trilin(3)
    end
end

# ============================================================================
# Fused integrator (no scratch arrays -> meets the <=72 B/cell memory gate).
# v1: recompute reconstruction of the needed neighbours per cell (no shared mem).
# Correct + memory-clean; shared-memory tiling for >=750 Mcell/s is layered on after.
# ============================================================================
# Reconstruction of one cell: primitive + MonCen slopes (3 dirs) + MUSCL-Hancock predictor.
@inline function recon_cell(uold, I::Int, J::Int, K::Int, N::Int,
                            gamma::Float32, smallr::Float32, pfl::Float32, dtdx::Float32)
    m0  = cons2prim(loadc(uold, I, J, K), gamma, smallr, pfl)
    mxl = cons2prim(loadc(uold, wrp(I-1,N), J, K), gamma, smallr, pfl)
    mxr = cons2prim(loadc(uold, wrp(I+1,N), J, K), gamma, smallr, pfl)
    myl = cons2prim(loadc(uold, I, wrp(J-1,N), K), gamma, smallr, pfl)
    myr = cons2prim(loadc(uold, I, wrp(J+1,N), K), gamma, smallr, pfl)
    mzl = cons2prim(loadc(uold, I, J, wrp(K-1,N)), gamma, smallr, pfl)
    mzr = cons2prim(loadc(uold, I, J, wrp(K+1,N)), gamma, smallr, pfl)
    sx = prim_slope(mxl, m0, mxr); sy = prim_slope(myl, m0, myr); sz = prim_slope(mzl, m0, mzr)
    uh = hancock(m0, sx, sy, sz, dtdx, gamma)
    mh = cons2prim(uh, gamma, smallr, pfl)
    if mh[1] <= smallr || mh[5] <= pfl
        mh = m0
    end
    return mh, sx, sy, sz
end

@kernel function integrator_fused!(unew, @Const(uold), @Const(afield),
                                   gamma::Float32, smallr::Float32, pfl::Float32, smallc::Float32,
                                   dt::Float32, dx::Float32, ch::Float32, glm_fac::Float32,
                                   llf_dmin::Float32, llf_pmin::Float32, use_hlld::Bool, N::Int,
                                   boxlen::Float32, ramp::Float32, turb_min_rho::Float32, do_turb::Bool)
    i, j, k = @index(Global, NTuple)
    @inbounds @fastmath begin
        dtdx = dt/dx
        ip = wrp(i+1,N); im = wrp(i-1,N); jp = wrp(j+1,N); jm = wrp(j-1,N); kp = wrp(k+1,N); km = wrp(k-1,N)
        mhc, sxc, syc, szc = recon_cell(uold, i, j, k, N, gamma, smallr, pfl, dtdx)
        # x faces
        mhxp, sxxp, _, _ = recon_cell(uold, ip, j, k, N, gamma, smallr, pfl, dtdx)
        mhxm, sxxm, _, _ = recon_cell(uold, im, j, k, N, gamma, smallr, pfl, dtdx)
        Fhx = riemann(padd(mhc,sxc,1.0f0),  padd(mhxp,sxxp,-1.0f0), 1, gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld)
        Flx = riemann(padd(mhxm,sxxm,1.0f0), padd(mhc,sxc,-1.0f0),  1, gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld)
        # y faces
        mhyp, _, syyp, _ = recon_cell(uold, i, jp, k, N, gamma, smallr, pfl, dtdx)
        mhym, _, syym, _ = recon_cell(uold, i, jm, k, N, gamma, smallr, pfl, dtdx)
        Fhy = riemann(padd(mhc,syc,1.0f0),  padd(mhyp,syyp,-1.0f0), 2, gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld)
        Fly = riemann(padd(mhym,syym,1.0f0), padd(mhc,syc,-1.0f0),  2, gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld)
        # z faces
        mhzp, _, _, szzp = recon_cell(uold, i, j, kp, N, gamma, smallr, pfl, dtdx)
        mhzm, _, _, szzm = recon_cell(uold, i, j, km, N, gamma, smallr, pfl, dtdx)
        Fhz = riemann(padd(mhc,szc,1.0f0),  padd(mhzp,szzp,-1.0f0), 3, gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld)
        Flz = riemann(padd(mhzm,szzm,1.0f0), padd(mhc,szc,-1.0f0),  3, gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld)
        u0 = loadc(uold, i, j, k)
        r1 = ntuple(v -> u0[v] + dtdx*((Flx[v]-Fhx[v]) + (Fly[v]-Fhy[v]) + (Flz[v]-Fhz[v])), 9)
        if r1[1] < smallr
            emag = 0.5f0*(r1[6]*r1[6] + r1[7]*r1[7] + r1[8]*r1[8])
            r1 = (smallr, 0.0f0, 0.0f0, 0.0f0, pfl/(gamma-1.0f0) + emag, r1[6], r1[7], r1[8], r1[9])
        end
        r = (r1[1],r1[2],r1[3],r1[4],r1[5],r1[6],r1[7],r1[8], r1[9]*glm_fac)
        if do_turb && r[1] >= turb_min_rho
            ax, ay, az = turb_interp(afield, i, j, k, boxlen/N, boxlen, ramp)
            rhom = max(r[1], smallr); sc2 = smallc*smallc; rho = r[1]
            e = r[5]
            e = max(e - 0.5f0*r[2]*r[2]/rhom, rho*sc2); e = max(e - 0.5f0*r[3]*r[3]/rhom, rho*sc2); e = max(e - 0.5f0*r[4]*r[4]/rhom, rho*sc2)
            m2 = r[2] + rhom*ax*dt; m3 = r[3] + rhom*ay*dt; m4 = r[4] + rhom*az*dt
            e = max(e + 0.5f0*m2*m2/rhom, rho*sc2); e = max(e + 0.5f0*m3*m3/rhom, rho*sc2); e = max(e + 0.5f0*m4*m4/rhom, rho*sc2)
            r = (r[1], m2, m3, m4, e, r[6], r[7], r[8], r[9])
        end
        for v in 1:9
            unew[i,j,k,v] = r[v]
        end
    end
end

# Fused step (no scratch): one kernel uold -> unew.
function step_fused!(uold, unew, afield, p::Params, dt::Float32, t::Float32, be;
                     use_hlld::Bool=true, do_turb::Bool=false)
    N = p.N; dx = dxof(p); pfl = pfloor(p)
    ch = p.courant*dx/dt/Float32(NDIM)*p.glm_ch_scale
    glm_fac = p.glm_cp_coef > 0 ? exp(-(ch*ch/(p.glm_cp_coef*p.boxlen*ch))*dt) : 1.0f0
    ramp = min(t/p.turb_T, 1.0f0)
    integrator_fused!(be)(unew, uold, afield, p.gamma, p.smallr, pfl, p.smallc, dt, dx, ch, glm_fac,
                          p.switch_llf_dmin, p.switch_llf_pmin, use_hlld, N, p.boxlen, ramp,
                          p.turb_min_rho, do_turb; ndrange=(N,N,N))
    KA.synchronize(be)
end

# ============================================================================
# Tiled shared-memory integrator (native CUDA.jl). Mirrors the reference GLM kernel:
# nsubgrid=2 -> 4^3 owned cells/block, 8^3 primitive tile (2-cell halo) in shared,
# 3 stages (c2p-load -> trace to per-face interface subgrids -> Riemann -> update).
# Each reconstruction computed ONCE, each face Riemann solved ONCE (vs the 7x recon /
# 2x Riemann of integrator_fused!). Shared budget (fp32): 9*512 + 6*9*80 = 8928 floats
# = 34.9 KB < 48 KB. All physics reuse the validated @inline device fns.
# ============================================================================
const TB = 4                 # owned cells per dim per block (nsubgrid=2)
const TT = TB + 4            # 8: primitive tile incl. 2-cell halo
const NCP = TT*TT*TT                  # 512  prim-tile cells
const NCF = (TB+1)*TB*TB              # 80   faces per direction
@inline lin_tile(pi, pj, pk) = pi + TT*(pj + TT*pk)            # 0..511
@inline lin_fx(fi, fj, fk)   = fi + (TB+1)*(fj + TB*fk)        # x faces (5,4,4) 0..79
@inline lin_fy(fi, fj, fk)   = fi + TB*(fj + (TB+1)*fk)        # y faces (4,5,4)
@inline lin_fz(fi, fj, fk)   = fi + TB*(fj + TB*fk)            # z faces (4,4,5)
# SoA shared layout: variable-major (S[(v-1)*NC + lin + 1]), NC = #cells inferred from
# the (statically sized) array length. A warp reading the same var across consecutive
# cells (lin) hits consecutive banks -> conflict-free/coalesced (vs AoS stride-9).
@inline shget(S, lin) = (NC = length(S) ÷ 9; ntuple(v -> @inbounds(S[(v-1)*NC + lin + 1]), 9))
@inline function shput!(S, lin, q)
    NC = length(S) ÷ 9
    @inbounds for v in 1:9
        S[(v-1)*NC + lin + 1] = q[v]
    end
end
# Revert an interface state to first-order (cell value m0) if it went unphysical.
@inline iflo(f, m0, smallr, pfl) = (f[1] < smallr || f[5] < pfl) ? m0 : f

function integrator_tiled!(lv::Val, unew, uold, afield,
                           gamma::Float32, smallr::Float32, pfl::Float32, smallc::Float32,
                           dt::Float32, dx::Float32, ch::Float32, glm_fac::Float32,
                           llf_dmin::Float32, llf_pmin::Float32, use_hlld::Bool, N::Int,
                           boxlen::Float32, ramp::Float32, turb_min_rho::Float32, do_turb::Bool)
    SP = CuStaticSharedArray(Float32, 9*TT*TT*TT)
    LX = CuStaticSharedArray(Float32, 9*(TB+1)*TB*TB); RX = CuStaticSharedArray(Float32, 9*(TB+1)*TB*TB)
    LY = CuStaticSharedArray(Float32, 9*TB*(TB+1)*TB); RY = CuStaticSharedArray(Float32, 9*TB*(TB+1)*TB)
    LZ = CuStaticSharedArray(Float32, 9*TB*TB*(TB+1)); RZ = CuStaticSharedArray(Float32, 9*TB*TB*(TB+1))
    @fastmath @inbounds begin
        dtdx = dt/dx
        tid = threadIdx().x; nth = blockDim().x
        ox = (blockIdx().x-1)*TB; oy = (blockIdx().y-1)*TB; oz = (blockIdx().z-1)*TB
        # ---- Stage 1: load conserved -> primitives into the 8^3 shared tile ----
        t = tid
        while t <= TT*TT*TT
            l = t-1; pi = l % TT; pj = (l ÷ TT) % TT; pk = l ÷ (TT*TT)
            gi = wrp(ox + pi - 1, N); gj = wrp(oy + pj - 1, N); gk = wrp(oz + pk - 1, N)
            q = cons2prim(loadc(uold, gi, gj, gk), gamma, smallr, pfl)
            shput!(SP, lin_tile(pi,pj,pk), q)
            t += nth
        end
        sync_threads()
        # ---- Stage 2: trace (MUSCL-Hancock) over inner 6^3 -> face interface states ----
        t = tid
        while t <= (TB+2)^3
            l = t-1; ci = l % (TB+2); cj = (l ÷ (TB+2)) % (TB+2); ck = l ÷ ((TB+2)*(TB+2))
            pi = ci+1; pj = cj+1; pk = ck+1                  # inner tile index 1..6
            m0  = shget(SP, lin_tile(pi,pj,pk))
            sx = slope_d(lv, shget(SP,lin_tile(pi-1,pj,pk)), m0, shget(SP,lin_tile(pi+1,pj,pk)))
            sy = slope_d(lv, shget(SP,lin_tile(pi,pj-1,pk)), m0, shget(SP,lin_tile(pi,pj+1,pk)))
            sz = slope_d(lv, shget(SP,lin_tile(pi,pj,pk-1)), m0, shget(SP,lin_tile(pi,pj,pk+1)))
            uh = hancock(m0, sx, sy, sz, dtdx, gamma)
            mh = cons2prim(uh, gamma, smallr, pfl)
            if mh[1] <= smallr || mh[5] <= pfl
                mh = m0
            end
            inxt = (pj >= 2 && pj <= TB+1) && (pk >= 2 && pk <= TB+1)
            inyt = (pi >= 2 && pi <= TB+1) && (pk >= 2 && pk <= TB+1)
            inzt = (pi >= 2 && pi <= TB+1) && (pj >= 2 && pj <= TB+1)
            # x faces
            if inxt
                if pi <= TB+1                                # fl (mh+sx) at face fi=pi-1
                    shput!(LX, lin_fx(pi-1, pj-2, pk-2), iflo(padd(mh,sx, 1.0f0), m0, smallr, pfl))
                end
                if pi >= 2                                   # fr (mh-sx) at face fi=pi-2
                    shput!(RX, lin_fx(pi-2, pj-2, pk-2), iflo(padd(mh,sx,-1.0f0), m0, smallr, pfl))
                end
            end
            # y faces
            if inyt
                if pj <= TB+1
                    shput!(LY, lin_fy(pi-2, pj-1, pk-2), iflo(padd(mh,sy, 1.0f0), m0, smallr, pfl))
                end
                if pj >= 2
                    shput!(RY, lin_fy(pi-2, pj-2, pk-2), iflo(padd(mh,sy,-1.0f0), m0, smallr, pfl))
                end
            end
            # z faces
            if inzt
                if pk <= TB+1
                    shput!(LZ, lin_fz(pi-2, pj-2, pk-1), iflo(padd(mh,sz, 1.0f0), m0, smallr, pfl))
                end
                if pk >= 2
                    shput!(RZ, lin_fz(pi-2, pj-2, pk-2), iflo(padd(mh,sz,-1.0f0), m0, smallr, pfl))
                end
            end
            t += nth
        end
        sync_threads()
        # ---- Stage 3: Riemann on every face (once), flux written back into L* ----
        nfx = (TB+1)*TB*TB
        t = tid
        while t <= 3*nfx
            if t <= nfx
                l = t-1; fi = l % (TB+1); fj = (l ÷ (TB+1)) % TB; fk = l ÷ ((TB+1)*TB)
                F = riemann_d(lv, shget(LX,lin_fx(fi,fj,fk)), shget(RX,lin_fx(fi,fj,fk)), 1,
                            gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld)
                shput!(LX, lin_fx(fi,fj,fk), F)
            elseif t <= 2*nfx
                l = t-1-nfx; fi = l % TB; fj = (l ÷ TB) % (TB+1); fk = l ÷ (TB*(TB+1))
                F = riemann_d(lv, shget(LY,lin_fy(fi,fj,fk)), shget(RY,lin_fy(fi,fj,fk)), 2,
                            gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld)
                shput!(LY, lin_fy(fi,fj,fk), F)
            else
                l = t-1-2*nfx; fi = l % TB; fj = (l ÷ TB) % TB; fk = l ÷ (TB*TB)
                F = riemann_d(lv, shget(LZ,lin_fz(fi,fj,fk)), shget(RZ,lin_fz(fi,fj,fk)), 3,
                            gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld)
                shput!(LZ, lin_fz(fi,fj,fk), F)
            end
            t += nth
        end
        sync_threads()
        # ---- Stage 4: conservative update of the 4^3 owned cells ----
        t = tid
        while t <= TB*TB*TB
            l = t-1; a = l % TB; b = (l ÷ TB) % TB; c = l ÷ (TB*TB)
            gi = wrp(ox + a + 1, N); gj = wrp(oy + b + 1, N); gk = wrp(oz + c + 1, N)
            Fxl = shget(LX, lin_fx(a,  b, c)); Fxh = shget(LX, lin_fx(a+1, b, c))
            Fyl = shget(LY, lin_fy(a, b,  c)); Fyh = shget(LY, lin_fy(a, b+1, c))
            Fzl = shget(LZ, lin_fz(a, b, c )); Fzh = shget(LZ, lin_fz(a, b, c+1))
            u0 = loadc(uold, gi, gj, gk)
            r1 = ntuple(v -> u0[v] + dtdx*((Fxl[v]-Fxh[v]) + (Fyl[v]-Fyh[v]) + (Fzl[v]-Fzh[v])), 9)
            if r1[1] < smallr
                emag = 0.5f0*(r1[6]*r1[6] + r1[7]*r1[7] + r1[8]*r1[8])
                r1 = (smallr, 0.0f0, 0.0f0, 0.0f0, pfl/(gamma-1.0f0) + emag, r1[6], r1[7], r1[8], r1[9])
            end
            r = (r1[1],r1[2],r1[3],r1[4],r1[5],r1[6],r1[7],r1[8], r1[9]*glm_fac)
            if do_turb && r[1] >= turb_min_rho
                ax, ay, az = turb_interp(afield, gi, gj, gk, boxlen/N, boxlen, ramp)
                rhom = max(r[1], smallr); sc2 = smallc*smallc; rho = r[1]; e = r[5]
                e = max(e - 0.5f0*r[2]*r[2]/rhom, rho*sc2); e = max(e - 0.5f0*r[3]*r[3]/rhom, rho*sc2); e = max(e - 0.5f0*r[4]*r[4]/rhom, rho*sc2)
                m2 = r[2] + rhom*ax*dt; m3 = r[3] + rhom*ay*dt; m4 = r[4] + rhom*az*dt
                e = max(e + 0.5f0*m2*m2/rhom, rho*sc2); e = max(e + 0.5f0*m3*m3/rhom, rho*sc2); e = max(e + 0.5f0*m4*m4/rhom, rho*sc2)
                r = (r[1], m2, m3, m4, e, r[6], r[7], r[8], r[9])
            end
            for v in 1:9
                unew[gi,gj,gk,v] = r[v]
            end
            t += nth
        end
    end
    return nothing
end

# Tiled step (no scratch): one kernel launch, (N/4)^3 blocks of 128 threads.
function step_tiled!(uold, unew, afield, p::Params, dt::Float32, t::Float32;
                     use_hlld::Bool=true, do_turb::Bool=false, nthreads::Int=192, maxregs::Int=0,
                     lean::Bool=false)
    N = p.N; dx = dxof(p); pfl = pfloor(p)
    @assert N % TB == 0 "N must be a multiple of $TB"
    ch = p.courant*dx/dt/Float32(NDIM)*p.glm_ch_scale
    glm_fac = p.glm_cp_coef > 0 ? exp(-(ch*ch/(p.glm_cp_coef*p.boxlen*ch))*dt) : 1.0f0
    ramp = min(t/p.turb_T, 1.0f0)
    nb = N ÷ TB
    # branch so each @cuda sees a CONCRETE Val (a Union{Val{true},Val{false}} confuses the launch
    # and segfaults host LLVM); the Val(true) specialization elides HLLD from the call tree.
    if lean
        @cuda threads=nthreads blocks=(nb,nb,nb) integrator_tiled!(
            Val(true), unew, uold, afield, p.gamma, p.smallr, pfl, p.smallc, dt, dx, ch, glm_fac,
            p.switch_llf_dmin, p.switch_llf_pmin, use_hlld, N, p.boxlen, ramp, p.turb_min_rho, do_turb)
    else
        @cuda threads=nthreads blocks=(nb,nb,nb) integrator_tiled!(
            Val(false), unew, uold, afield, p.gamma, p.smallr, pfl, p.smallc, dt, dx, ch, glm_fac,
            p.switch_llf_dmin, p.switch_llf_pmin, use_hlld, N, p.boxlen, ramp, p.turb_min_rho, do_turb)
    end
    return nothing
end

# ============================================================================
# Production PLM kernel — parametric over tile size TB, f16 shared (HALF), and Riemann
# (RIEM ∈ :hlld/:hll/:llf). MonCen-PLM + MUSCL-Hancock + GLM cleaning. f16 SHARED storage with
# an f32 UPDATE: the lossy f16 round-trip only touches the reconstruction; u_old + dt·divF is in
# f32, so conserved accumulation (and div·B) stay accurate (OT: Δvrms 6e-6 vs f32, div·B controlled).
# Default config (TB=6, f16, HLL) fits 48 KB static shared and runs ~2080 Mcell/s at 512³ (1.8×
# the f32/TB4/HLLD kernel). Shared element type halved -> nsubgrid=3 fits; its lower halo (4.6×
# vs 8×) is the win. f32/TB4/:hlld remains a flag for bit-reproducibility.
# ============================================================================
@inline _sgT(S, lin) = ntuple(v -> Float32(@inbounds(S[9*lin + v])), 9)   # ST -> f32 on read
@inline function _spT(S, lin, q)                                          # f32 -> ST on write
    @inbounds for v in 1:9; S[9*lin + v] = q[v]; end
end
function integrator_plm!(::Val{TB}, ::Val{HALF}, ::Val{RIEM}, unew, uold, afield,
                         gamma::Float32, smallr::Float32, pfl::Float32, smallc::Float32,
                         dt::Float32, dx::Float32, ch::Float32, glm_fac::Float32,
                         llf_dmin::Float32, llf_pmin::Float32, N::Int,
                         boxlen::Float32, ramp::Float32, turb_min_rho::Float32, do_turb::Bool) where {TB,HALF,RIEM}
    TT = TB + 4; ST = HALF ? Float16 : Float32
    SP = CuStaticSharedArray(ST, 9*TT*TT*TT)
    LX = CuStaticSharedArray(ST, 9*(TB+1)*TB*TB); RX = CuStaticSharedArray(ST, 9*(TB+1)*TB*TB)
    LY = CuStaticSharedArray(ST, 9*TB*(TB+1)*TB); RY = CuStaticSharedArray(ST, 9*TB*(TB+1)*TB)
    LZ = CuStaticSharedArray(ST, 9*TB*TB*(TB+1)); RZ = CuStaticSharedArray(ST, 9*TB*TB*(TB+1))
    lt(pi,pj,pk)=pi+TT*(pj+TT*pk); lfx(fi,fj,fk)=fi+(TB+1)*(fj+TB*fk); lfy(fi,fj,fk)=fi+TB*(fj+(TB+1)*fk); lfz(fi,fj,fk)=fi+TB*(fj+TB*fk)
    rv = Val(RIEM)
    @fastmath @inbounds begin
        dtdx = dt/dx; tid = threadIdx().x; nth = blockDim().x
        ox=(blockIdx().x-1)*TB; oy=(blockIdx().y-1)*TB; oz=(blockIdx().z-1)*TB
        t = tid
        while t <= TT*TT*TT
            l=t-1; pi=l%TT; pj=(l÷TT)%TT; pk=l÷(TT*TT)
            _spT(SP, lt(pi,pj,pk), cons2prim(loadc(uold, wrp(ox+pi-1,N), wrp(oy+pj-1,N), wrp(oz+pk-1,N)), gamma,smallr,pfl)); t+=nth
        end
        sync_threads(); t=tid
        while t <= (TB+2)^3
            l=t-1; ci=l%(TB+2); cj=(l÷(TB+2))%(TB+2); ck=l÷((TB+2)*(TB+2)); pi=ci+1;pj=cj+1;pk=ck+1
            m0 = _sgT(SP, lt(pi,pj,pk))
            sx = prim_slope(_sgT(SP,lt(pi-1,pj,pk)), m0, _sgT(SP,lt(pi+1,pj,pk)))
            sy = prim_slope(_sgT(SP,lt(pi,pj-1,pk)), m0, _sgT(SP,lt(pi,pj+1,pk)))
            sz = prim_slope(_sgT(SP,lt(pi,pj,pk-1)), m0, _sgT(SP,lt(pi,pj,pk+1)))
            mh = cons2prim(hancock(m0,sx,sy,sz,dtdx,gamma), gamma,smallr,pfl)
            (mh[1]<=smallr || mh[5]<=pfl) && (mh = m0)
            inxt=(pj>=2&&pj<=TB+1)&&(pk>=2&&pk<=TB+1); inyt=(pi>=2&&pi<=TB+1)&&(pk>=2&&pk<=TB+1); inzt=(pi>=2&&pi<=TB+1)&&(pj>=2&&pj<=TB+1)
            if inxt
                pi<=TB+1 && _spT(LX, lfx(pi-1,pj-2,pk-2), iflo(padd(mh,sx,1.0f0),m0,smallr,pfl))
                pi>=2    && _spT(RX, lfx(pi-2,pj-2,pk-2), iflo(padd(mh,sx,-1.0f0),m0,smallr,pfl))
            end
            if inyt
                pj<=TB+1 && _spT(LY, lfy(pi-2,pj-1,pk-2), iflo(padd(mh,sy,1.0f0),m0,smallr,pfl))
                pj>=2    && _spT(RY, lfy(pi-2,pj-2,pk-2), iflo(padd(mh,sy,-1.0f0),m0,smallr,pfl))
            end
            if inzt
                pk<=TB+1 && _spT(LZ, lfz(pi-2,pj-2,pk-1), iflo(padd(mh,sz,1.0f0),m0,smallr,pfl))
                pk>=2    && _spT(RZ, lfz(pi-2,pj-2,pk-2), iflo(padd(mh,sz,-1.0f0),m0,smallr,pfl))
            end
            t+=nth
        end
        sync_threads(); nfx=(TB+1)*TB*TB; t=tid
        while t <= 3*nfx
            if t<=nfx
                l=t-1;fi=l%(TB+1);fj=(l÷(TB+1))%TB;fk=l÷((TB+1)*TB)
                _spT(LX, lfx(fi,fj,fk), riemann_sel(rv, _sgT(LX,lfx(fi,fj,fk)), _sgT(RX,lfx(fi,fj,fk)), 1, gamma,ch,smallr,pfl,llf_dmin,llf_pmin))
            elseif t<=2*nfx
                l=t-1-nfx;fi=l%TB;fj=(l÷TB)%(TB+1);fk=l÷(TB*(TB+1))
                _spT(LY, lfy(fi,fj,fk), riemann_sel(rv, _sgT(LY,lfy(fi,fj,fk)), _sgT(RY,lfy(fi,fj,fk)), 2, gamma,ch,smallr,pfl,llf_dmin,llf_pmin))
            else
                l=t-1-2*nfx;fi=l%TB;fj=(l÷TB)%TB;fk=l÷(TB*TB)
                _spT(LZ, lfz(fi,fj,fk), riemann_sel(rv, _sgT(LZ,lfz(fi,fj,fk)), _sgT(RZ,lfz(fi,fj,fk)), 3, gamma,ch,smallr,pfl,llf_dmin,llf_pmin))
            end
            t+=nth
        end
        sync_threads(); t=tid
        while t <= TB*TB*TB
            l=t-1;a=l%TB;b=(l÷TB)%TB;c=l÷(TB*TB); gi=wrp(ox+a+1,N);gj=wrp(oy+b+1,N);gk=wrp(oz+c+1,N)
            Fxl=_sgT(LX,lfx(a,b,c));Fxh=_sgT(LX,lfx(a+1,b,c));Fyl=_sgT(LY,lfy(a,b,c));Fyh=_sgT(LY,lfy(a,b+1,c));Fzl=_sgT(LZ,lfz(a,b,c));Fzh=_sgT(LZ,lfz(a,b,c+1))
            u0 = loadc(uold,gi,gj,gk)   # f32 update: increment promoted to f32, added to f32 state
            r1 = ntuple(v -> u0[v] + dtdx*((Fxl[v]-Fxh[v])+(Fyl[v]-Fyh[v])+(Fzl[v]-Fzh[v])), 9)
            if r1[1] < smallr
                emag=0.5f0*(r1[6]*r1[6]+r1[7]*r1[7]+r1[8]*r1[8])
                r1=(smallr,0.0f0,0.0f0,0.0f0,pfl/(gamma-1.0f0)+emag,r1[6],r1[7],r1[8],r1[9])
            end
            r=(r1[1],r1[2],r1[3],r1[4],r1[5],r1[6],r1[7],r1[8], r1[9]*glm_fac)
            if do_turb && r[1] >= turb_min_rho
                ax,ay,az = turb_interp(afield, gi,gj,gk, boxlen/N, boxlen, ramp)
                rhom=max(r[1],smallr); sc2=smallc*smallc; rho=r[1]; e=r[5]
                e=max(e-0.5f0*r[2]*r[2]/rhom,rho*sc2);e=max(e-0.5f0*r[3]*r[3]/rhom,rho*sc2);e=max(e-0.5f0*r[4]*r[4]/rhom,rho*sc2)
                m2=r[2]+rhom*ax*dt;m3=r[3]+rhom*ay*dt;m4=r[4]+rhom*az*dt
                e=max(e+0.5f0*m2*m2/rhom,rho*sc2);e=max(e+0.5f0*m3*m3/rhom,rho*sc2);e=max(e+0.5f0*m4*m4/rhom,rho*sc2)
                r=(r[1],m2,m3,m4,e,r[6],r[7],r[8],r[9])
            end
            for v in 1:9; unew[gi,gj,gk,v]=r[v]; end
            t+=nth
        end
    end
    return nothing
end

# Launch helper: the (TB,HALF,RIEM) Vals MUST be concrete literals at the call site (a runtime
# Val(tb)/Val(riemann) is type-unstable and crashes the host LLVM), so step_plm! branches below.
@inline function _plm_launch(tv::Val, hv::Val, rv::Val, nthreads, nb, unew, uold, afield, p::Params,
                             pfl, dt, dx, ch, glm_fac, ramp, do_turb)
    @cuda threads=nthreads blocks=(nb,nb,nb) integrator_plm!(
        tv, hv, rv, unew, uold, afield, p.gamma, p.smallr, pfl, p.smallc, dt, dx, ch, glm_fac,
        p.switch_llf_dmin, p.switch_llf_pmin, p.N, p.boxlen, ramp, p.turb_min_rho, do_turb)
end

# Production PLM step. Default = Hancock + HLL + GLM + PLM, f16 shared, TB=6 (nsubgrid=3).
# Supported configs: (tb=6,half=true,riemann=:hll|:llf), (tb=4,half=false,riemann=:hlld bit-repro),
# (tb=4,half=true,riemann=:hll). Other combos: error (add a branch if needed).
function step_plm!(uold, unew, afield, p::Params, dt::Float32, t::Float32;
                   do_turb::Bool=false, nthreads::Int=192, tb::Int=0, half::Bool=true,
                   riemann::Symbol=:hll, recon::Symbol=:plm)
    N = p.N; dx = dxof(p); pfl = pfloor(p)
    tb == 0 && (tb = N % 6 == 0 ? 6 : 4)   # auto: nsubgrid=3 when it divides N, else nsubgrid=2
    @assert N % tb == 0 "N=$N not divisible by tb=$tb (need N%6==0 for the fast TB=6 path, or N%4==0)"
    ch = p.courant*dx/dt/Float32(NDIM)*p.glm_ch_scale
    glm_fac = p.glm_cp_coef > 0 ? exp(-(ch*ch/(p.glm_cp_coef*p.boxlen*ch))*dt) : 1.0f0
    ramp = min(t/p.turb_T, 1.0f0); nb = N ÷ tb
    if recon === :ppm   # lean parabolic-PPM (single-zone edges + Hancock + shock fallback); f16+HLL only
        (half && riemann===:hll) || error("step_plm! recon=:ppm supports only half=true,riemann=:hll (got half=$half, riemann=$riemann)")
        if tb==6
            _plm_ppm_launch(Val(6),Val(true),Val(:hll), nthreads,nb,unew,uold,afield,p,pfl,dt,dx,ch,glm_fac,ramp,do_turb)
        elseif tb==4
            _plm_ppm_launch(Val(4),Val(true),Val(:hll), nthreads,nb,unew,uold,afield,p,pfl,dt,dx,ch,glm_fac,ramp,do_turb)
        else
            error("step_plm! recon=:ppm: unsupported tb=$tb (use 4 or 6)")
        end
    elseif tb==6 && half && riemann===:hll
        _plm_launch(Val(6),Val(true),Val(:hll), nthreads,nb,unew,uold,afield,p,pfl,dt,dx,ch,glm_fac,ramp,do_turb)
    elseif tb==6 && half && riemann===:llf
        _plm_launch(Val(6),Val(true),Val(:llf), nthreads,nb,unew,uold,afield,p,pfl,dt,dx,ch,glm_fac,ramp,do_turb)
    elseif tb==4 && !half && riemann===:hlld
        _plm_launch(Val(4),Val(false),Val(:hlld), nthreads,nb,unew,uold,afield,p,pfl,dt,dx,ch,glm_fac,ramp,do_turb)
    elseif tb==4 && half && riemann===:hll
        _plm_launch(Val(4),Val(true),Val(:hll), nthreads,nb,unew,uold,afield,p,pfl,dt,dx,ch,glm_fac,ramp,do_turb)
    else
        error("step_plm!: unsupported (tb=$tb, half=$half, riemann=$riemann); add a branch")
    end
    return nothing
end

# ============================================================================
# PPM reconstruction (single-zone, ±1 stencil → SAME (TB+4) tile as PLM). Transliterated from
# the reference `mini-ramses-metal/gpu/gpu_hydro.cuf` (`local_ppm_*`, `trace_3d_mhd_par`) and
# Vespa.jl `lib/PPMKernels`: 3-point parabolic edges + Colella-Woodward (CW84) monotonize.
# Penalty vs PLM (N=480, f16, TB6, HLL): hydro 1.49x, GLM-MHD 2.86x — an occupancy cliff set by
# the register budget (9-var MHD PLM is already 144 regs/2 blocks/SM; PPM's 201 regs → 1 block/SM).
# ============================================================================
@fastmath @inline function ppm_mono(ql, q0, qr)   # CW84 monotonize -> (lo, hi)
    dq=qr-ql; diff=(q0-0.5f0*(ql+qr))*dq
    ifelse((qr-q0)*(q0-ql)<=0f0, (q0,q0),
      ifelse(diff>dq*dq/6f0, (3f0*q0-2f0*qr, qr),
        ifelse(diff<(-dq*dq/6f0), (ql, 3f0*q0-2f0*ql), (ql,qr))))
end
@fastmath @inline ppm_edges(qm,q0,qp) = (slope=0.25f0*(qp-qm); curve=(qm-2f0*q0+qp)/12f0; ppm_mono(q0-slope+curve, q0, q0+slope+curve))
@fastmath @inline function ppm9(l,m,r); e=ntuple(v->ppm_edges(l[v],m[v],r[v]),9); (ntuple(v->e[v][1],9), ntuple(v->e[v][2],9)); end
@fastmath @inline mhd_face9(mh,ed,m0) = ntuple(v->mh[v]+ed[v]-m0[v], 9)   # PPM spatial edge + Hancock time prediction
@fastmath @inline spj(a,b,c) = (hi=max(a,max(b,c)); lo=max(min(a,min(b,c)),1f-20); hi/lo>2.0f0)   # strong_pressure_jump

# MHD lean parabolic-PPM kernel: identical to integrator_plm! except the reconstruction stage —
# MonCen slope + Hancock predictor (unchanged), PPM only sets the spatial face mh+edge-m0, with a
# per-cell strong_pressure_jump -> PLM fallback at shocks.
function integrator_plm_ppm!(::Val{TB}, ::Val{HALF}, ::Val{RIEM}, unew, uold, afield,
                         gamma::Float32, smallr::Float32, pfl::Float32, smallc::Float32,
                         dt::Float32, dx::Float32, ch::Float32, glm_fac::Float32,
                         llf_dmin::Float32, llf_pmin::Float32, N::Int,
                         boxlen::Float32, ramp::Float32, turb_min_rho::Float32, do_turb::Bool) where {TB,HALF,RIEM}
    TT = TB + 4; ST = HALF ? Float16 : Float32
    SP = CuStaticSharedArray(ST, 9*TT*TT*TT)
    LX = CuStaticSharedArray(ST, 9*(TB+1)*TB*TB); RX = CuStaticSharedArray(ST, 9*(TB+1)*TB*TB)
    LY = CuStaticSharedArray(ST, 9*TB*(TB+1)*TB); RY = CuStaticSharedArray(ST, 9*TB*(TB+1)*TB)
    LZ = CuStaticSharedArray(ST, 9*TB*TB*(TB+1)); RZ = CuStaticSharedArray(ST, 9*TB*TB*(TB+1))
    lt(pi,pj,pk)=pi+TT*(pj+TT*pk); lfx(fi,fj,fk)=fi+(TB+1)*(fj+TB*fk); lfy(fi,fj,fk)=fi+TB*(fj+(TB+1)*fk); lfz(fi,fj,fk)=fi+TB*(fj+TB*fk)
    rv = Val(RIEM)
    @fastmath @inbounds begin
        dtdx = dt/dx; tid = threadIdx().x; nth = blockDim().x
        ox=(blockIdx().x-1)*TB; oy=(blockIdx().y-1)*TB; oz=(blockIdx().z-1)*TB
        t = tid
        while t <= TT*TT*TT
            l=t-1; pi=l%TT; pj=(l÷TT)%TT; pk=l÷(TT*TT)
            _spT(SP, lt(pi,pj,pk), cons2prim(loadc(uold, wrp(ox+pi-1,N), wrp(oy+pj-1,N), wrp(oz+pk-1,N)), gamma,smallr,pfl)); t+=nth
        end
        sync_threads(); t=tid
        while t <= (TB+2)^3
            l=t-1; ci=l%(TB+2); cj=(l÷(TB+2))%(TB+2); ck=l÷((TB+2)*(TB+2)); pi=ci+1;pj=cj+1;pk=ck+1
            m0 = _sgT(SP, lt(pi,pj,pk))
            mxl=_sgT(SP,lt(pi-1,pj,pk));mxr=_sgT(SP,lt(pi+1,pj,pk));myl=_sgT(SP,lt(pi,pj-1,pk));myr=_sgT(SP,lt(pi,pj+1,pk));mzl=_sgT(SP,lt(pi,pj,pk-1));mzr=_sgT(SP,lt(pi,pj,pk+1))
            sx=prim_slope(mxl,m0,mxr); sy=prim_slope(myl,m0,myr); sz=prim_slope(mzl,m0,mzr)
            mh = cons2prim(hancock(m0,sx,sy,sz,dtdx,gamma), gamma,smallr,pfl)
            (mh[1]<=smallr || mh[5]<=pfl) && (mh = m0)
            up = !(spj(mxl[5],m0[5],mxr[5])||spj(myl[5],m0[5],myr[5])||spj(mzl[5],m0[5],mzr[5]))
            pmx,ppx=ppm9(mxl,m0,mxr); pmy,ppy=ppm9(myl,m0,myr); pmz,ppz=ppm9(mzl,m0,mzr)
            inxt=(pj>=2&&pj<=TB+1)&&(pk>=2&&pk<=TB+1); inyt=(pi>=2&&pi<=TB+1)&&(pk>=2&&pk<=TB+1); inzt=(pi>=2&&pi<=TB+1)&&(pj>=2&&pj<=TB+1)
            if inxt
                pi<=TB+1 && _spT(LX, lfx(pi-1,pj-2,pk-2), iflo(up ? mhd_face9(mh,ppx,m0) : padd(mh,sx,1.0f0),m0,smallr,pfl))
                pi>=2    && _spT(RX, lfx(pi-2,pj-2,pk-2), iflo(up ? mhd_face9(mh,pmx,m0) : padd(mh,sx,-1.0f0),m0,smallr,pfl))
            end
            if inyt
                pj<=TB+1 && _spT(LY, lfy(pi-2,pj-1,pk-2), iflo(up ? mhd_face9(mh,ppy,m0) : padd(mh,sy,1.0f0),m0,smallr,pfl))
                pj>=2    && _spT(RY, lfy(pi-2,pj-2,pk-2), iflo(up ? mhd_face9(mh,pmy,m0) : padd(mh,sy,-1.0f0),m0,smallr,pfl))
            end
            if inzt
                pk<=TB+1 && _spT(LZ, lfz(pi-2,pj-2,pk-1), iflo(up ? mhd_face9(mh,ppz,m0) : padd(mh,sz,1.0f0),m0,smallr,pfl))
                pk>=2    && _spT(RZ, lfz(pi-2,pj-2,pk-2), iflo(up ? mhd_face9(mh,pmz,m0) : padd(mh,sz,-1.0f0),m0,smallr,pfl))
            end
            t+=nth
        end
        sync_threads(); nfx=(TB+1)*TB*TB; t=tid
        while t <= 3*nfx
            if t<=nfx
                l=t-1;fi=l%(TB+1);fj=(l÷(TB+1))%TB;fk=l÷((TB+1)*TB)
                _spT(LX, lfx(fi,fj,fk), riemann_sel(rv, _sgT(LX,lfx(fi,fj,fk)), _sgT(RX,lfx(fi,fj,fk)), 1, gamma,ch,smallr,pfl,llf_dmin,llf_pmin))
            elseif t<=2*nfx
                l=t-1-nfx;fi=l%TB;fj=(l÷TB)%(TB+1);fk=l÷(TB*(TB+1))
                _spT(LY, lfy(fi,fj,fk), riemann_sel(rv, _sgT(LY,lfy(fi,fj,fk)), _sgT(RY,lfy(fi,fj,fk)), 2, gamma,ch,smallr,pfl,llf_dmin,llf_pmin))
            else
                l=t-1-2*nfx;fi=l%TB;fj=(l÷TB)%TB;fk=l÷(TB*TB)
                _spT(LZ, lfz(fi,fj,fk), riemann_sel(rv, _sgT(LZ,lfz(fi,fj,fk)), _sgT(RZ,lfz(fi,fj,fk)), 3, gamma,ch,smallr,pfl,llf_dmin,llf_pmin))
            end
            t+=nth
        end
        sync_threads(); t=tid
        while t <= TB*TB*TB
            l=t-1;a=l%TB;b=(l÷TB)%TB;c=l÷(TB*TB); gi=wrp(ox+a+1,N);gj=wrp(oy+b+1,N);gk=wrp(oz+c+1,N)
            Fxl=_sgT(LX,lfx(a,b,c));Fxh=_sgT(LX,lfx(a+1,b,c));Fyl=_sgT(LY,lfy(a,b,c));Fyh=_sgT(LY,lfy(a,b+1,c));Fzl=_sgT(LZ,lfz(a,b,c));Fzh=_sgT(LZ,lfz(a,b,c+1))
            u0 = loadc(uold,gi,gj,gk)
            r1 = ntuple(v -> u0[v] + dtdx*((Fxl[v]-Fxh[v])+(Fyl[v]-Fyh[v])+(Fzl[v]-Fzh[v])), 9)
            if r1[1] < smallr
                emag=0.5f0*(r1[6]*r1[6]+r1[7]*r1[7]+r1[8]*r1[8])
                r1=(smallr,0.0f0,0.0f0,0.0f0,pfl/(gamma-1.0f0)+emag,r1[6],r1[7],r1[8],r1[9])
            end
            r=(r1[1],r1[2],r1[3],r1[4],r1[5],r1[6],r1[7],r1[8], r1[9]*glm_fac)
            if do_turb && r[1] >= turb_min_rho
                ax,ay,az = turb_interp(afield, gi,gj,gk, boxlen/N, boxlen, ramp)
                rhom=max(r[1],smallr); sc2=smallc*smallc; rho=r[1]; e=r[5]
                e=max(e-0.5f0*r[2]*r[2]/rhom,rho*sc2);e=max(e-0.5f0*r[3]*r[3]/rhom,rho*sc2);e=max(e-0.5f0*r[4]*r[4]/rhom,rho*sc2)
                m2=r[2]+rhom*ax*dt;m3=r[3]+rhom*ay*dt;m4=r[4]+rhom*az*dt
                e=max(e+0.5f0*m2*m2/rhom,rho*sc2);e=max(e+0.5f0*m3*m3/rhom,rho*sc2);e=max(e+0.5f0*m4*m4/rhom,rho*sc2)
                r=(r[1],m2,m3,m4,e,r[6],r[7],r[8],r[9])
            end
            for v in 1:9; unew[gi,gj,gk,v]=r[v]; end
            t+=nth
        end
    end
    return nothing
end

@inline function _plm_ppm_launch(tv::Val, hv::Val, rv::Val, nthreads, nb, unew, uold, afield, p::Params,
                                 pfl, dt, dx, ch, glm_fac, ramp, do_turb)
    @cuda threads=nthreads blocks=(nb,nb,nb) integrator_plm_ppm!(
        tv, hv, rv, unew, uold, afield, p.gamma, p.smallr, pfl, p.smallc, dt, dx, ch, glm_fac,
        p.switch_llf_dmin, p.switch_llf_pmin, p.N, p.boxlen, ramp, p.turb_min_rho, do_turb)
end

# ============================================================================
# Pure-hydro (5-var Euler) production solver: same architecture and defaults as the GLM-MHD PLM
# kernel — MonCen-PLM + MUSCL-Hancock (primitive source-term predictor) + HLL, f16 shared + f32
# update, TB=6 (nsubgrid=3). No B-field / GLM. ~4300 Mcell/s (5-var) on the A6000.
# ============================================================================
@inline loadc5(u,i,j,k) = ntuple(v -> @inbounds(u[i,j,k,v]), 5)
@fastmath @inline function cons2prim5(c, gamma::Float32, smallr::Float32, pfl::Float32)
    rho=max(c[1],smallr); ir=1.0f0/rho
    p=max((gamma-1.0f0)*(c[5]-0.5f0*(c[2]*c[2]+c[3]*c[3]+c[4]*c[4])*ir), pfl)
    (rho, c[2]*ir, c[3]*ir, c[4]*ir, p)
end
@fastmath @inline function prim2cons5(q, gamma::Float32)
    rho=q[1];vx=q[2];vy=q[3];vz=q[4];p=q[5]
    (rho, rho*vx, rho*vy, rho*vz, p/(gamma-1.0f0)+0.5f0*rho*(vx*vx+vy*vy+vz*vz))
end
@fastmath @inline function phys_flux_x5(q, gamma::Float32)
    rho=q[1];vx=q[2];vy=q[3];vz=q[4];p=q[5]; E=p/(gamma-1.0f0)+0.5f0*rho*(vx*vx+vy*vy+vz*vz)
    (rho*vx, rho*vx*vx+p, rho*vx*vy, rho*vx*vz, (E+p)*vx)
end
@fastmath @inline sound5(q, gamma::Float32) = sqrt(gamma*q[5]/q[1])
@inline rot_to5(q,dir) = dir==1 ? q : (dir==2 ? (q[1],q[3],q[4],q[2],q[5]) : (q[1],q[4],q[2],q[3],q[5]))
@inline rot_flux5(f,dir) = dir==1 ? f : (dir==2 ? (f[1],f[4],f[2],f[3],f[5]) : (f[1],f[3],f[4],f[2],f[5]))
@fastmath @inline prim_slope5(L,M,R) = ntuple(i -> 0.5f0*minmod(M[i]-L[i], R[i]-M[i]), 5)
@inline padd5(q,s,a) = ntuple(i -> q[i]+a*s[i], 5)
# primitive source-term predictor (no flux evals); returns predicted PRIMITIVE state
@fastmath @inline function hancock5(m0,sx,sy,sz,dtdx::Float32,gamma::Float32)
    rho=m0[1];vx=m0[2];vy=m0[3];vz=m0[4];p=m0[5]; dv=sx[2]+sy[3]+sz[4]
    (rho+dtdx*(-vx*sx[1]-vy*sy[1]-vz*sz[1]-dv*rho),
     vx+dtdx*(-vx*sx[2]-vy*sy[2]-vz*sz[2]-sx[5]/rho),
     vy+dtdx*(-vx*sx[3]-vy*sy[3]-vz*sz[3]-sy[5]/rho),
     vz+dtdx*(-vx*sx[4]-vy*sy[4]-vz*sz[4]-sz[5]/rho),
     p+dtdx*(-vx*sx[5]-vy*sy[5]-vz*sz[5]-dv*gamma*p))
end
@fastmath @inline function hll5x(L,R,gamma::Float32)
    fL=phys_flux_x5(L,gamma);fR=phys_flux_x5(R,gamma);uL=prim2cons5(L,gamma);uR=prim2cons5(R,gamma)
    cl=sound5(L,gamma);cr=sound5(R,gamma); SL=min(min(L[2]-cl,R[2]-cr),0.0f0); SR=max(max(L[2]+cl,R[2]+cr),0.0f0); ihd=1.0f0/(SR-SL)
    ntuple(i->(SR*fL[i]-SL*fR[i]+SL*SR*(uR[i]-uL[i]))*ihd,5)
end
@fastmath @inline function llf5x(L,R,gamma::Float32)
    fL=phys_flux_x5(L,gamma);fR=phys_flux_x5(R,gamma);uL=prim2cons5(L,gamma);uR=prim2cons5(R,gamma)
    s=max(abs(L[2])+sound5(L,gamma), abs(R[2])+sound5(R,gamma))
    ntuple(i->0.5f0*(fL[i]+fR[i])-0.5f0*s*(uR[i]-uL[i]),5)
end
@fastmath @inline function riem5(Lq,Rq,dir,gamma::Float32,smallr::Float32,pfl::Float32,::Val{RIEM}) where RIEM
    L0=rot_to5(Lq,dir);R0=rot_to5(Rq,dir)
    L=(max(L0[1],smallr),L0[2],L0[3],L0[4],max(L0[5],pfl)); R=(max(R0[1],smallr),R0[2],R0[3],R0[4],max(R0[5],pfl))
    rot_flux5(RIEM===:hll ? hll5x(L,R,gamma) : llf5x(L,R,gamma), dir)
end
@inline iflo5(f,m0,smallr,pfl) = (f[1]<smallr||f[5]<pfl) ? m0 : f
@inline _sg5(S,lin) = ntuple(v -> Float32(@inbounds(S[5*lin+v])), 5)
@inline function _sp5(S,lin,q); @inbounds for v in 1:5; S[5*lin+v]=q[v]; end; end

function integrator_hydro!(::Val{TB},::Val{HALF},::Val{RIEM}, unew, uold, afield,
                           gamma::Float32, smallr::Float32, pfl::Float32, smallc::Float32,
                           dt::Float32, dx::Float32, N::Int, boxlen::Float32, ramp::Float32,
                           turb_min_rho::Float32, do_turb::Bool) where {TB,HALF,RIEM}
    TT=TB+4; ST=HALF ? Float16 : Float32
    SP=CuStaticSharedArray(ST,5*TT*TT*TT)
    LX=CuStaticSharedArray(ST,5*(TB+1)*TB*TB);RX=CuStaticSharedArray(ST,5*(TB+1)*TB*TB)
    LY=CuStaticSharedArray(ST,5*TB*(TB+1)*TB);RY=CuStaticSharedArray(ST,5*TB*(TB+1)*TB)
    LZ=CuStaticSharedArray(ST,5*TB*TB*(TB+1));RZ=CuStaticSharedArray(ST,5*TB*TB*(TB+1))
    lt(pi,pj,pk)=pi+TT*(pj+TT*pk);lfx(fi,fj,fk)=fi+(TB+1)*(fj+TB*fk);lfy(fi,fj,fk)=fi+TB*(fj+(TB+1)*fk);lfz(fi,fj,fk)=fi+TB*(fj+TB*fk)
    rv=Val(RIEM)
    @fastmath @inbounds begin
        dtdx=dt/dx;tid=threadIdx().x;nth=blockDim().x
        ox=(blockIdx().x-1)*TB;oy=(blockIdx().y-1)*TB;oz=(blockIdx().z-1)*TB
        t=tid
        while t<=TT*TT*TT
            l=t-1;pi=l%TT;pj=(l÷TT)%TT;pk=l÷(TT*TT)
            _sp5(SP,lt(pi,pj,pk),cons2prim5(loadc5(uold,wrp(ox+pi-1,N),wrp(oy+pj-1,N),wrp(oz+pk-1,N)),gamma,smallr,pfl));t+=nth
        end
        sync_threads();t=tid
        while t<=(TB+2)^3
            l=t-1;ci=l%(TB+2);cj=(l÷(TB+2))%(TB+2);ck=l÷((TB+2)*(TB+2));pi=ci+1;pj=cj+1;pk=ck+1
            m0=_sg5(SP,lt(pi,pj,pk))
            sx=prim_slope5(_sg5(SP,lt(pi-1,pj,pk)),m0,_sg5(SP,lt(pi+1,pj,pk)));sy=prim_slope5(_sg5(SP,lt(pi,pj-1,pk)),m0,_sg5(SP,lt(pi,pj+1,pk)));sz=prim_slope5(_sg5(SP,lt(pi,pj,pk-1)),m0,_sg5(SP,lt(pi,pj,pk+1)))
            mh=hancock5(m0,sx,sy,sz,dtdx,gamma); (mh[1]<=smallr||mh[5]<=pfl)&&(mh=m0)
            inx=(pj>=2&&pj<=TB+1)&&(pk>=2&&pk<=TB+1);iny=(pi>=2&&pi<=TB+1)&&(pk>=2&&pk<=TB+1);inz=(pi>=2&&pi<=TB+1)&&(pj>=2&&pj<=TB+1)
            if inx; pi<=TB+1&&_sp5(LX,lfx(pi-1,pj-2,pk-2),iflo5(padd5(mh,sx,1.0f0),m0,smallr,pfl)); pi>=2&&_sp5(RX,lfx(pi-2,pj-2,pk-2),iflo5(padd5(mh,sx,-1.0f0),m0,smallr,pfl)); end
            if iny; pj<=TB+1&&_sp5(LY,lfy(pi-2,pj-1,pk-2),iflo5(padd5(mh,sy,1.0f0),m0,smallr,pfl)); pj>=2&&_sp5(RY,lfy(pi-2,pj-2,pk-2),iflo5(padd5(mh,sy,-1.0f0),m0,smallr,pfl)); end
            if inz; pk<=TB+1&&_sp5(LZ,lfz(pi-2,pj-2,pk-1),iflo5(padd5(mh,sz,1.0f0),m0,smallr,pfl)); pk>=2&&_sp5(RZ,lfz(pi-2,pj-2,pk-2),iflo5(padd5(mh,sz,-1.0f0),m0,smallr,pfl)); end
            t+=nth
        end
        sync_threads();nfx=(TB+1)*TB*TB;t=tid
        while t<=3*nfx
            if t<=nfx;l=t-1;fi=l%(TB+1);fj=(l÷(TB+1))%TB;fk=l÷((TB+1)*TB);_sp5(LX,lfx(fi,fj,fk),riem5(_sg5(LX,lfx(fi,fj,fk)),_sg5(RX,lfx(fi,fj,fk)),1,gamma,smallr,pfl,rv))
            elseif t<=2*nfx;l=t-1-nfx;fi=l%TB;fj=(l÷TB)%(TB+1);fk=l÷(TB*(TB+1));_sp5(LY,lfy(fi,fj,fk),riem5(_sg5(LY,lfy(fi,fj,fk)),_sg5(RY,lfy(fi,fj,fk)),2,gamma,smallr,pfl,rv))
            else;l=t-1-2*nfx;fi=l%TB;fj=(l÷TB)%TB;fk=l÷(TB*TB);_sp5(LZ,lfz(fi,fj,fk),riem5(_sg5(LZ,lfz(fi,fj,fk)),_sg5(RZ,lfz(fi,fj,fk)),3,gamma,smallr,pfl,rv)) end
            t+=nth
        end
        sync_threads();t=tid
        while t<=TB*TB*TB
            l=t-1;a=l%TB;b=(l÷TB)%TB;c=l÷(TB*TB);gi=wrp(ox+a+1,N);gj=wrp(oy+b+1,N);gk=wrp(oz+c+1,N)
            Fxl=_sg5(LX,lfx(a,b,c));Fxh=_sg5(LX,lfx(a+1,b,c));Fyl=_sg5(LY,lfy(a,b,c));Fyh=_sg5(LY,lfy(a,b+1,c));Fzl=_sg5(LZ,lfz(a,b,c));Fzh=_sg5(LZ,lfz(a,b,c+1))
            u0=loadc5(uold,gi,gj,gk)
            r=ntuple(v->u0[v]+dtdx*((Fxl[v]-Fxh[v])+(Fyl[v]-Fyh[v])+(Fzl[v]-Fzh[v])),5)  # f32 update
            r[1]<smallr && (r=(smallr,0.0f0,0.0f0,0.0f0,pfl/(gamma-1.0f0)))
            if do_turb && r[1]>=turb_min_rho
                ax,ay,az=turb_interp(afield,gi,gj,gk,boxlen/N,boxlen,ramp)
                rhom=max(r[1],smallr);sc2=smallc*smallc;rho=r[1];e=r[5]
                e=max(e-0.5f0*r[2]*r[2]/rhom,rho*sc2);e=max(e-0.5f0*r[3]*r[3]/rhom,rho*sc2);e=max(e-0.5f0*r[4]*r[4]/rhom,rho*sc2)
                m2=r[2]+rhom*ax*dt;m3=r[3]+rhom*ay*dt;m4=r[4]+rhom*az*dt
                e=max(e+0.5f0*m2*m2/rhom,rho*sc2);e=max(e+0.5f0*m3*m3/rhom,rho*sc2);e=max(e+0.5f0*m4*m4/rhom,rho*sc2)
                r=(r[1],m2,m3,m4,e)
            end
            for v in 1:5; unew[gi,gj,gk,v]=r[v]; end
            t+=nth
        end
    end
    return nothing
end

@inline function _hydro_launch(tv::Val,hv::Val,rv::Val,nthreads,nb,unew,uold,afield,p::Params,pfl,dt,dx,ramp,do_turb)
    @cuda threads=nthreads blocks=(nb,nb,nb) integrator_hydro!(
        tv,hv,rv,unew,uold,afield,p.gamma,p.smallr,pfl,p.smallc,dt,dx,p.N,p.boxlen,ramp,p.turb_min_rho,do_turb)
end
# Pure-hydro PLM step. Default = MonCen-PLM + Hancock + HLL, f16, TB auto (6 if N%6==0 else 4).
function step_hydro!(uold, unew, afield, p::Params, dt::Float32, t::Float32;
                     do_turb::Bool=false, nthreads::Int=192, tb::Int=0, half::Bool=true,
                     riemann::Symbol=:hll, recon::Symbol=:plm)
    N=p.N; dx=dxof(p); pfl=pfloor(p); ramp=min(t/p.turb_T,1.0f0)
    tb==0 && (tb = N%6==0 ? 6 : 4); @assert N%tb==0 "N=$N not divisible by tb=$tb"
    nb=N÷tb
    if recon === :ppm   # single-zone 3-wave characteristic PPM; f16+HLL only
        (half && riemann===:hll) || error("step_hydro! recon=:ppm supports only half=true,riemann=:hll")
        if tb==6
            _hydro_ppm_launch(Val(6),Val(true),Val(:hll), nthreads,nb,unew,uold,afield,p,pfl,dt,dx,ramp,do_turb)
        elseif tb==4
            _hydro_ppm_launch(Val(4),Val(true),Val(:hll), nthreads,nb,unew,uold,afield,p,pfl,dt,dx,ramp,do_turb)
        else
            error("step_hydro! recon=:ppm: unsupported tb=$tb (use 4 or 6)")
        end
    elseif tb==6 && half && riemann===:hll
        _hydro_launch(Val(6),Val(true),Val(:hll), nthreads,nb,unew,uold,afield,p,pfl,dt,dx,ramp,do_turb)
    elseif tb==6 && half && riemann===:llf
        _hydro_launch(Val(6),Val(true),Val(:llf), nthreads,nb,unew,uold,afield,p,pfl,dt,dx,ramp,do_turb)
    elseif tb==4 && half && riemann===:hll
        _hydro_launch(Val(4),Val(true),Val(:hll), nthreads,nb,unew,uold,afield,p,pfl,dt,dx,ramp,do_turb)
    elseif tb==4 && !half && riemann===:hll
        _hydro_launch(Val(4),Val(false),Val(:hll), nthreads,nb,unew,uold,afield,p,pfl,dt,dx,ramp,do_turb)
    else
        error("step_hydro!: unsupported (tb=$tb, half=$half, riemann=$riemann); add a branch")
    end
    return nothing
end

# ============================================================================
# 2.5D fused line-march hydro kernel (unigrid). Each thread owns one (x,y) column
# and marches z through a rolling 5-plane f16 ring; per cell it does fused
# c2p(at load)->slopes/Hancock->HLL x6 faces->update with neighbor traces
# recomputed (no shared face tiles). Two barriers per plane-advance (vs the cube's
# 4 staged barriers) and EVERY thread stays active every step — the fix for what
# sank the earlier integrator_stream! (per-plane phases underfilled warps). Same
# physics as integrator_hydro! (MonCen-PLM + Hancock + HLL, f16 ring). Tests
# whether the CUDA spike's +40-75% over the cube materializes here.
# ============================================================================
@fastmath @inline function mhsl5(m0,xm,xp,ym,yp,zm,zp,dtdx,gamma,smallr,pfl)
    sx=prim_slope5(xm,m0,xp); sy=prim_slope5(ym,m0,yp); sz=prim_slope5(zm,m0,zp)
    mh=hancock5(m0,sx,sy,sz,dtdx,gamma); (mh[1]<=smallr||mh[5]<=pfl) && (mh=m0)
    (mh,sx,sy,sz)
end

function integrator_hydro_march!(::Val{OX},::Val{OY},::Val{RIEM}, unew,uold,afield,
        gamma::Float32,smallr::Float32,pfl::Float32,smallc::Float32,
        dt::Float32,dx::Float32,N::Int,boxlen::Float32,ramp::Float32,
        turb_min_rho::Float32,do_turb::Bool) where {OX,OY,RIEM}
    GX=OX+4; GY=OY+4; GG=GX*GY; PL=5
    SP=CuStaticSharedArray(Float16, 5*PL*GG)
    rv=Val(RIEM)
    rget(s,lx,ly)=(b=5*(s*GG+lx+GX*ly); ntuple(v->Float32(@inbounds SP[b+v]),5))
    rput(s,lx,ly,q)=(b=5*(s*GG+lx+GX*ly); @inbounds for v in 1:5; SP[b+v]=Float16(q[v]); end)
    @fastmath @inbounds begin
        dtdx=dt/dx; tid=threadIdx().x; nth=blockDim().x
        tx=(tid-1)%OX; ty=(tid-1)÷OX
        ox=(blockIdx().x-1)*OX; oy=(blockIdx().y-1)*OY
        li=tx+2; lj=ty+2
        loadpl(s,gz)=begin
            c=tid
            while c<=GG
                lx=(c-1)%GX; ly=(c-1)÷GX
                rput(s,lx,ly, cons2prim5(loadc5(uold,wrp(ox+lx-1,N),wrp(oy+ly-1,N),gz),gamma,smallr,pfl))
                c+=nth
            end
        end
        for pp in -2:2; loadpl(mod(pp,PL), wrp(pp+1,N)); end
        sync_threads()
        p=0
        while p < N
            sm2=mod(p-2,PL);sm1=mod(p-1,PL);s0=mod(p,PL);sp1=mod(p+1,PL);sp2=mod(p+2,PL)
            m0=rget(s0,li,lj)
            (mh0,sx0,sy0,sz0)=mhsl5(m0,rget(s0,li-1,lj),rget(s0,li+1,lj),rget(s0,li,lj-1),rget(s0,li,lj+1),rget(sm1,li,lj),rget(sp1,li,lj),dtdx,gamma,smallr,pfl)
            mxm=rget(s0,li-1,lj); (mhxm,sxm,_,_)=mhsl5(mxm,rget(s0,li-2,lj),m0,rget(s0,li-1,lj-1),rget(s0,li-1,lj+1),rget(sm1,li-1,lj),rget(sp1,li-1,lj),dtdx,gamma,smallr,pfl)
            mxp=rget(s0,li+1,lj); (mhxp,sxp,_,_)=mhsl5(mxp,m0,rget(s0,li+2,lj),rget(s0,li+1,lj-1),rget(s0,li+1,lj+1),rget(sm1,li+1,lj),rget(sp1,li+1,lj),dtdx,gamma,smallr,pfl)
            mym=rget(s0,li,lj-1); (mhym,_,sym,_)=mhsl5(mym,rget(s0,li-1,lj-1),rget(s0,li+1,lj-1),rget(s0,li,lj-2),m0,rget(sm1,li,lj-1),rget(sp1,li,lj-1),dtdx,gamma,smallr,pfl)
            myp=rget(s0,li,lj+1); (mhyp,_,syp,_)=mhsl5(myp,rget(s0,li-1,lj+1),rget(s0,li+1,lj+1),m0,rget(s0,li,lj+2),rget(sm1,li,lj+1),rget(sp1,li,lj+1),dtdx,gamma,smallr,pfl)
            mzm=rget(sm1,li,lj); (mhzm,_,_,szm)=mhsl5(mzm,rget(sm1,li-1,lj),rget(sm1,li+1,lj),rget(sm1,li,lj-1),rget(sm1,li,lj+1),rget(sm2,li,lj),m0,dtdx,gamma,smallr,pfl)
            mzp=rget(sp1,li,lj); (mhzp,_,_,szp)=mhsl5(mzp,rget(sp1,li-1,lj),rget(sp1,li+1,lj),rget(sp1,li,lj-1),rget(sp1,li,lj+1),m0,rget(sp2,li,lj),dtdx,gamma,smallr,pfl)
            Fxl=riem5(iflo5(padd5(mhxm,sxm, 1f0),mxm,smallr,pfl), iflo5(padd5(mh0,sx0,-1f0),m0,smallr,pfl),1,gamma,smallr,pfl,rv)
            Fxh=riem5(iflo5(padd5(mh0,sx0, 1f0),m0,smallr,pfl),   iflo5(padd5(mhxp,sxp,-1f0),mxp,smallr,pfl),1,gamma,smallr,pfl,rv)
            Fyl=riem5(iflo5(padd5(mhym,sym, 1f0),mym,smallr,pfl), iflo5(padd5(mh0,sy0,-1f0),m0,smallr,pfl),2,gamma,smallr,pfl,rv)
            Fyh=riem5(iflo5(padd5(mh0,sy0, 1f0),m0,smallr,pfl),   iflo5(padd5(mhyp,syp,-1f0),myp,smallr,pfl),2,gamma,smallr,pfl,rv)
            Fzl=riem5(iflo5(padd5(mhzm,szm, 1f0),mzm,smallr,pfl), iflo5(padd5(mh0,sz0,-1f0),m0,smallr,pfl),3,gamma,smallr,pfl,rv)
            Fzh=riem5(iflo5(padd5(mh0,sz0, 1f0),m0,smallr,pfl),   iflo5(padd5(mhzp,szp,-1f0),mzp,smallr,pfl),3,gamma,smallr,pfl,rv)
            gi=wrp(ox+tx+1,N);gj=wrp(oy+ty+1,N);gk=p+1
            u0=loadc5(uold,gi,gj,gk)
            r=ntuple(v->u0[v]+dtdx*((Fxl[v]-Fxh[v])+(Fyl[v]-Fyh[v])+(Fzl[v]-Fzh[v])),5)
            r[1]<smallr && (r=(smallr,0f0,0f0,0f0,pfl/(gamma-1f0)))
            if do_turb && r[1]>=turb_min_rho
                ax,ay,az=turb_interp(afield,gi,gj,gk,boxlen/N,boxlen,ramp)
                rhom=max(r[1],smallr);sc2=smallc*smallc;rho=r[1];e=r[5]
                e=max(e-0.5f0*r[2]*r[2]/rhom,rho*sc2);e=max(e-0.5f0*r[3]*r[3]/rhom,rho*sc2);e=max(e-0.5f0*r[4]*r[4]/rhom,rho*sc2)
                m2=r[2]+rhom*ax*dt;m3=r[3]+rhom*ay*dt;m4=r[4]+rhom*az*dt
                e=max(e+0.5f0*m2*m2/rhom,rho*sc2);e=max(e+0.5f0*m3*m3/rhom,rho*sc2);e=max(e+0.5f0*m4*m4/rhom,rho*sc2)
                r=(r[1],m2,m3,m4,e)
            end
            for v in 1:5; unew[gi,gj,gk,v]=r[v]; end
            sync_threads(); loadpl(sm2, wrp(p+4,N)); sync_threads()
            p+=1
        end
    end
    return nothing
end

@inline function _hydro_march_launch(::Val{OX},::Val{OY},rv::Val,unew,uold,afield,p::Params,pfl,dt,dx,ramp,do_turb) where {OX,OY}
    @cuda threads=(OX*OY) blocks=(p.N÷OX,p.N÷OY,1) integrator_hydro_march!(
        Val(OX),Val(OY),rv,unew,uold,afield,p.gamma,p.smallr,pfl,p.smallc,dt,dx,p.N,p.boxlen,ramp,p.turb_min_rho,do_turb)
end

# Fused 2.5D march hydro step. ox,oy = owned tile per block (concrete-literal Vals).
function step_hydro_march!(uold,unew,afield,p::Params,dt::Float32,t::Float32;
                           do_turb::Bool=false, ox::Int=32, oy::Int=8, riemann::Symbol=:hll)
    dx=dxof(p); pfl=pfloor(p); ramp=min(t/p.turb_T,1f0)
    @assert p.N%ox==0 && p.N%oy==0 "N=$(p.N) not divisible by ox=$ox,oy=$oy"
    rv = riemann===:hll ? Val(:hll) : Val(:llf)
    if     ox==32&&oy==8;  _hydro_march_launch(Val(32),Val(8), rv,unew,uold,afield,p,pfl,dt,dx,ramp,do_turb)
    elseif ox==16&&oy==16; _hydro_march_launch(Val(16),Val(16),rv,unew,uold,afield,p,pfl,dt,dx,ramp,do_turb)
    elseif ox==16&&oy==8;  _hydro_march_launch(Val(16),Val(8), rv,unew,uold,afield,p,pfl,dt,dx,ramp,do_turb)
    elseif ox==24&&oy==8;  _hydro_march_launch(Val(24),Val(8), rv,unew,uold,afield,p,pfl,dt,dx,ramp,do_turb)
    elseif ox==48&&oy==8;  _hydro_march_launch(Val(48),Val(8), rv,unew,uold,afield,p,pfl,dt,dx,ramp,do_turb)
    elseif ox==32&&oy==4;  _hydro_march_launch(Val(32),Val(4), rv,unew,uold,afield,p,pfl,dt,dx,ramp,do_turb)
    else error("step_hydro_march!: add a literal-Val branch for ox=$ox,oy=$oy")
    end
    return nothing
end

# Bench + validate the march vs the cube (step_hydro!) on identical random IC.
function bench_hydro(; N::Int=480, nsteps::Int=30, warm::Int=5, kind::Symbol=:cube,
                     ox::Int=32, oy::Int=8, do_turb::Bool=false, tb::Int=0)
    p = Params(N=N)
    u1 = CUDA.zeros(Float32,N,N,N,NV); u2 = CUDA.zeros(Float32,N,N,N,NV)
    afield = CUDA.zeros(Float32,3,TURB_GS,TURB_GS,TURB_GS)
    p0=p.rho0*p.cs0^2/p.gamma
    ic_uniform_kernel!(CUDABackend())(u1,p.rho0,p0,p.b0,p.gamma; ndrange=(N,N,N)); CUDA.synchronize()
    dt=1f-4
    stepfn = kind===:march ?
        (a,b)->step_hydro_march!(a,b,afield,p,dt,0.1f0; do_turb=do_turb, ox=ox, oy=oy) :
        (a,b)->step_hydro!(a,b,afield,p,dt,0.1f0; do_turb=do_turb, tb=tb)
    for s in 1:warm; stepfn(u1,u2); u1,u2=u2,u1; end; CUDA.synchronize()
    el = CUDA.@elapsed begin
        for s in 1:nsteps; stepfn(u1,u2); u1,u2=u2,u1; end; CUDA.synchronize()
    end
    mcell=N^3*nsteps/el/1e6; fin=all(isfinite,Array(@view u1[1:2,1:2,1:2,:]))
    @printf("HYDRO %-6s N=%d %s: %.0f Mcell/s (%.2f ms/step) finite=%s\n",
            kind, N, kind===:march ? "ox=$ox oy=$oy" : "tb=$(tb==0 ? (N%6==0 ? 6 : 4) : tb)",
            mcell, el/nsteps*1e3, fin)
    return mcell
end

function validate_hydro_march(; N::Int=96, nsteps::Int=15, ox::Int=32, oy::Int=8)
    p = Params(N=N); Random.seed!(7)
    h=zeros(Float32,N,N,N,NV)
    h[:,:,:,1].=1f0 .+ 0.3f0.*rand(Float32,N,N,N)
    h[:,:,:,2].=0.2f0.*(rand(Float32,N,N,N).-0.5f0)
    h[:,:,:,3].=0.2f0.*(rand(Float32,N,N,N).-0.5f0)
    h[:,:,:,4].=0.2f0.*(rand(Float32,N,N,N).-0.5f0)
    h[:,:,:,5].=2.5f0 .+ 0.5f0.*rand(Float32,N,N,N)
    uc=CuArray(h); uc2=CUDA.zeros(Float32,N,N,N,NV)
    um=copy(uc);   um2=CUDA.zeros(Float32,N,N,N,NV)
    af=CUDA.zeros(Float32,3,TURB_GS,TURB_GS,TURB_GS); dt=5f-4
    for s in 1:nsteps
        step_hydro!(uc,uc2,af,p,dt,0f0; do_turb=false); uc,uc2=uc2,uc
        step_hydro_march!(um,um2,af,p,dt,0f0; do_turb=false, ox=ox, oy=oy); um,um2=um2,um
    end
    CUDA.synchronize()
    a=Array(uc); b=Array(um)
    dmax=maximum(abs.(a[:,:,:,1].-b[:,:,:,1]))/maximum(abs.(a[:,:,:,1]))
    emax=maximum(abs.(a[:,:,:,5].-b[:,:,:,5]))/maximum(abs.(a[:,:,:,5]))
    @printf("VALIDATE march vs cube N=%d steps=%d: rel|Δρ|=%.2e rel|ΔE|=%.2e (f16-roundoff expected ~1e-2)\n",
            N, nsteps, dmax, emax)
    return (dmax=dmax, emax=emax)
end

# --- march2: non-redundant variant. Precompute Hancock-predicted prim mh ONCE per cell
# into a second rolling ring (MH), so the flux stage reads mh + recomputes only the
# 1 direction-slope per edge (no 7x mh recompute). prim-ring 2-ghost x5 planes,
# mh-ring 1-ghost x4 planes. 3 barriers/plane. Tests whether removing the Hancock
# redundancy lets the march beat the cube. ---
function integrator_hydro_march2!(::Val{OX},::Val{OY},::Val{RIEM}, unew,uold,afield,
        gamma::Float32,smallr::Float32,pfl::Float32,smallc::Float32,
        dt::Float32,dx::Float32,N::Int,boxlen::Float32,ramp::Float32,
        turb_min_rho::Float32,do_turb::Bool) where {OX,OY,RIEM}
    GX=OX+4;GY=OY+4;GG=GX*GY;PL=5;ML=4
    SP=CuStaticSharedArray(Float16,5*PL*GG)
    MH=CuStaticSharedArray(Float16,5*ML*GG)
    rv=Val(RIEM)
    pget(s,lx,ly)=(b=5*(s*GG+lx+GX*ly);ntuple(v->Float32(@inbounds SP[b+v]),5))
    pput(s,lx,ly,q)=(b=5*(s*GG+lx+GX*ly);@inbounds for v in 1:5;SP[b+v]=Float16(q[v]);end)
    mget(s,lx,ly)=(b=5*(s*GG+lx+GX*ly);ntuple(v->Float32(@inbounds MH[b+v]),5))
    mput(s,lx,ly,q)=(b=5*(s*GG+lx+GX*ly);@inbounds for v in 1:5;MH[b+v]=Float16(q[v]);end)
    @fastmath @inbounds begin
        dtdx=dt/dx;tid=threadIdx().x;nth=blockDim().x
        tx=(tid-1)%OX;ty=(tid-1)÷OX
        ox=(blockIdx().x-1)*OX;oy=(blockIdx().y-1)*OY
        li=tx+2;lj=ty+2
        loadpl(s,gz)=begin
            c=tid
            while c<=GG
                lx=(c-1)%GX;ly=(c-1)÷GX
                pput(s,lx,ly,cons2prim5(loadc5(uold,wrp(ox+lx-1,N),wrp(oy+ly-1,N),gz),gamma,smallr,pfl)); c+=nth
            end
        end
        mhpl(sM,sd,sP,su)=begin
            c=tid
            while c<=GG
                lx=(c-1)%GX;ly=(c-1)÷GX
                if lx>=1&&lx<=GX-2&&ly>=1&&ly<=GY-2
                    m0=pget(sP,lx,ly)
                    sx=prim_slope5(pget(sP,lx-1,ly),m0,pget(sP,lx+1,ly))
                    sy=prim_slope5(pget(sP,lx,ly-1),m0,pget(sP,lx,ly+1))
                    sz=prim_slope5(pget(sd,lx,ly),m0,pget(su,lx,ly))
                    mh=hancock5(m0,sx,sy,sz,dtdx,gamma);(mh[1]<=smallr||mh[5]<=pfl)&&(mh=m0)
                    mput(sM,lx,ly,mh)
                end
                c+=nth
            end
        end
        for pp in -2:2; loadpl(mod(pp,PL),wrp(pp+1,N)); end
        sync_threads()
        mhpl(mod(-1,ML),mod(-2,PL),mod(-1,PL),mod(0,PL))
        mhpl(mod(0,ML), mod(-1,PL),mod(0,PL), mod(1,PL))
        mhpl(mod(1,ML), mod(0,PL), mod(1,PL), mod(2,PL))
        sync_threads()
        p=0
        while p<N
            s0=mod(p,PL);sm1=mod(p-1,PL);sp1=mod(p+1,PL)
            mc=mod(p,ML);md=mod(p-1,ML);mu=mod(p+1,ML)
            m0=pget(s0,li,lj); mh0=mget(mc,li,lj)
            gi=wrp(ox+tx+1,N);gj=wrp(oy+ty+1,N);gk=p+1
            r0=loadc5(uold,gi,gj,gk)
            # accumulate flux divergence one direction at a time, FRESH name per stage
            # (reassigning a lambda-captured var boxes -> GC alloc; bind a new name instead).
            # Each block's slopes/fluxes die before the next -> small peak live state.
            rx=let mm=pget(s0,li-1,lj),mp=pget(s0,li+1,lj)
                sm=prim_slope5(pget(s0,li-2,lj),mm,m0); s0d=prim_slope5(mm,m0,mp); sp=prim_slope5(m0,mp,pget(s0,li+2,lj))
                Fl=riem5(iflo5(padd5(mget(mc,li-1,lj),sm,1f0),mm,smallr,pfl), iflo5(padd5(mh0,s0d,-1f0),m0,smallr,pfl),1,gamma,smallr,pfl,rv)
                Fh=riem5(iflo5(padd5(mh0,s0d,1f0),m0,smallr,pfl), iflo5(padd5(mget(mc,li+1,lj),sp,-1f0),mp,smallr,pfl),1,gamma,smallr,pfl,rv)
                ntuple(v->r0[v]+dtdx*(Fl[v]-Fh[v]),5)
            end
            ry=let mm=pget(s0,li,lj-1),mp=pget(s0,li,lj+1)
                sm=prim_slope5(pget(s0,li,lj-2),mm,m0); s0d=prim_slope5(mm,m0,mp); sp=prim_slope5(m0,mp,pget(s0,li,lj+2))
                Fl=riem5(iflo5(padd5(mget(mc,li,lj-1),sm,1f0),mm,smallr,pfl), iflo5(padd5(mh0,s0d,-1f0),m0,smallr,pfl),2,gamma,smallr,pfl,rv)
                Fh=riem5(iflo5(padd5(mh0,s0d,1f0),m0,smallr,pfl), iflo5(padd5(mget(mc,li,lj+1),sp,-1f0),mp,smallr,pfl),2,gamma,smallr,pfl,rv)
                ntuple(v->rx[v]+dtdx*(Fl[v]-Fh[v]),5)
            end
            r=let mm=pget(sm1,li,lj),mp=pget(sp1,li,lj)
                sm=prim_slope5(pget(mod(p-2,PL),li,lj),mm,m0); s0d=prim_slope5(mm,m0,mp); sp=prim_slope5(m0,mp,pget(mod(p+2,PL),li,lj))
                Fl=riem5(iflo5(padd5(mget(md,li,lj),sm,1f0),mm,smallr,pfl), iflo5(padd5(mh0,s0d,-1f0),m0,smallr,pfl),3,gamma,smallr,pfl,rv)
                Fh=riem5(iflo5(padd5(mh0,s0d,1f0),m0,smallr,pfl), iflo5(padd5(mget(mu,li,lj),sp,-1f0),mp,smallr,pfl),3,gamma,smallr,pfl,rv)
                ntuple(v->ry[v]+dtdx*(Fl[v]-Fh[v]),5)
            end
            r[1]<smallr&&(r=(smallr,0f0,0f0,0f0,pfl/(gamma-1f0)))
            if do_turb && r[1]>=turb_min_rho
                ax,ay,az=turb_interp(afield,gi,gj,gk,boxlen/N,boxlen,ramp)
                rhom=max(r[1],smallr);sc2=smallc*smallc;rho=r[1];e=r[5]
                e=max(e-0.5f0*r[2]*r[2]/rhom,rho*sc2);e=max(e-0.5f0*r[3]*r[3]/rhom,rho*sc2);e=max(e-0.5f0*r[4]*r[4]/rhom,rho*sc2)
                m2=r[2]+rhom*ax*dt;m3=r[3]+rhom*ay*dt;m4=r[4]+rhom*az*dt
                e=max(e+0.5f0*m2*m2/rhom,rho*sc2);e=max(e+0.5f0*m3*m3/rhom,rho*sc2);e=max(e+0.5f0*m4*m4/rhom,rho*sc2)
                r=(r[1],m2,m3,m4,e)
            end
            for v in 1:5;unew[gi,gj,gk,v]=r[v];end
            sync_threads(); loadpl(mod(p-2,PL),wrp(p+4,N)); sync_threads()
            mhpl(mod(p+2,ML),mod(p+1,PL),mod(p+2,PL),mod(p+3,PL)); sync_threads()
            p+=1
        end
    end
    return nothing
end

@inline function _hydro_march2_launch(::Val{OX},::Val{OY},rv::Val,unew,uold,afield,p::Params,pfl,dt,dx,ramp,do_turb) where {OX,OY}
    @cuda threads=(OX*OY) blocks=(p.N÷OX,p.N÷OY,1) integrator_hydro_march2!(
        Val(OX),Val(OY),rv,unew,uold,afield,p.gamma,p.smallr,pfl,p.smallc,dt,dx,p.N,p.boxlen,ramp,p.turb_min_rho,do_turb)
end
function step_hydro_march2!(uold,unew,afield,p::Params,dt::Float32,t::Float32; do_turb::Bool=false, ox::Int=32, oy::Int=8, riemann::Symbol=:hll)
    dx=dxof(p); pfl=pfloor(p); ramp=min(t/p.turb_T,1f0); rv = riemann===:hll ? Val(:hll) : Val(:llf)
    if     ox==32&&oy==8;  _hydro_march2_launch(Val(32),Val(8), rv,unew,uold,afield,p,pfl,dt,dx,ramp,do_turb)
    elseif ox==16&&oy==16; _hydro_march2_launch(Val(16),Val(16),rv,unew,uold,afield,p,pfl,dt,dx,ramp,do_turb)
    elseif ox==32&&oy==4;  _hydro_march2_launch(Val(32),Val(4), rv,unew,uold,afield,p,pfl,dt,dx,ramp,do_turb)
    elseif ox==16&&oy==8;  _hydro_march2_launch(Val(16),Val(8), rv,unew,uold,afield,p,pfl,dt,dx,ramp,do_turb)
    else error("step_hydro_march2!: add a literal-Val branch for ox=$ox,oy=$oy")
    end
    return nothing
end

# ---- Hydro single-zone PPM: parabolic edges (ppm_edges, shared with MHD) + 3-wave characteristic
# trace (local_ppm_trace: parabola integrated over each acoustic/entropy wave's domain of dependence
# via ppm_avg). ±1 stencil → same (TB+4) tile as PLM. ~2856 Mcell/s vs PLM 4250 (1.49x). ----
@fastmath @inline ppm_avg(ql,qa,qr,lo,hi) = (b=-4f0*ql+6f0*qa-2f0*qr; c=3f0*(ql-2f0*qa+qr); ql+0.5f0*b*(lo+hi)+c*(lo*lo+lo*hi+hi*hi)/3f0)
@fastmath @inline function ppm_side(::Val{R}, m, cs, dtdx, dl,dr,ul,ur,vl,vr,wl,wr,pl,pr) where {R}
    rg=R ? dr : dl; ug=R ? ur : ul; vg=R ? vr : vl; wg=R ? wr : wl; pg=R ? pr : pl
    od=0f0;ou=0f0;ov=0f0;ow=0f0;op=0f0; ics=1f0/cs; irho=1f0/m[1]
    @inbounds for wave in 1:3
        lam = wave==1 ? m[2]-cs : (wave==2 ? m[2] : m[2]+cs)
        go = R ? (lam>0f0) : (lam<0f0)
        if go
            sigma=min(abs(lam)*dtdx,1f0); a = R ? 1f0-sigma : 0f0; b = R ? 1f0 : sigma
            du=ppm_avg(ul,m[2],ur,a,b)-ug; dpr=ppm_avg(pl,m[5],pr,a,b)-pg
            if wave==1
                amp=-m[1]*du*0.5f0*ics+dpr*0.5f0*ics*ics; ou+=(-cs*irho)*amp; od+=amp; op+=cs*cs*amp
            elseif wave==3
                amp= m[1]*du*0.5f0*ics+dpr*0.5f0*ics*ics; ou+=( cs*irho)*amp; od+=amp; op+=cs*cs*amp
            else
                od+=ppm_avg(dl,m[1],dr,a,b)-rg-dpr*ics*ics; ov+=ppm_avg(vl,m[3],vr,a,b)-vg; ow+=ppm_avg(wl,m[4],wr,a,b)-wg
            end
        end
    end
    q=(rg+od,ug+ou,vg+ov,wg+ow,pg+op); (q[1]<=0f0||q[5]<=0f0) ? m : q
end
@fastmath @inline function ppm_trace5(l,m,r,gamma,dtdx)   # -> (qplus,qminus); normal velocity in slot 2
    sp=1f-30; sr=1f-10
    dl,dr=ppm_edges(l[1],m[1],r[1]); ul,ur=ppm_edges(l[2],m[2],r[2]); vl,vr=ppm_edges(l[3],m[3],r[3]); wl,wr=ppm_edges(l[4],m[4],r[4]); pl,pr=ppm_edges(l[5],m[5],r[5])
    cs=sqrt(gamma*max(m[5],sp)/max(m[1],sr))
    dl,dr=ppm_mono(dl,m[1],dr); ul,ur=ppm_mono(ul,m[2],ur); vl,vr=ppm_mono(vl,m[3],vr); wl,wr=ppm_mono(wl,m[4],wr); pl,pr=ppm_mono(pl,m[5],pr)
    (dl<=0f0||dr<=0f0)&&(dl=m[1];dr=m[1]); (pl<=0f0||pr<=0f0)&&(pl=m[5];pr=m[5])
    dl=max(dl,sr);dr=max(dr,sr);pl=max(pl,sp);pr=max(pr,sp)
    (ppm_side(Val(true),m,cs,dtdx,dl,dr,ul,ur,vl,vr,wl,wr,pl,pr), ppm_side(Val(false),m,cs,dtdx,dl,dr,ul,ur,vl,vr,wl,wr,pl,pr))
end
@fastmath @inline function ppm_dir(l,m,r,gamma,dtdx,::Val{D}) where {D}   # rotate dir D into slot 2, trace, rotate back
    qp,qm=ppm_trace5(rot_to5(l,D),rot_to5(m,D),rot_to5(r,D),gamma,dtdx); (rot_flux5(qp,D), rot_flux5(qm,D))
end

# Hydro PPM kernel: identical to integrator_hydro! except stage 2 uses the per-direction
# characteristic PPM trace instead of MonCen slope + Hancock.
function integrator_hydro_ppm!(::Val{TB},::Val{HALF},::Val{RIEM}, unew, uold, afield,
                           gamma::Float32, smallr::Float32, pfl::Float32, smallc::Float32,
                           dt::Float32, dx::Float32, N::Int, boxlen::Float32, ramp::Float32,
                           turb_min_rho::Float32, do_turb::Bool) where {TB,HALF,RIEM}
    TT=TB+4; ST=HALF ? Float16 : Float32
    SP=CuStaticSharedArray(ST,5*TT*TT*TT)
    LX=CuStaticSharedArray(ST,5*(TB+1)*TB*TB);RX=CuStaticSharedArray(ST,5*(TB+1)*TB*TB)
    LY=CuStaticSharedArray(ST,5*TB*(TB+1)*TB);RY=CuStaticSharedArray(ST,5*TB*(TB+1)*TB)
    LZ=CuStaticSharedArray(ST,5*TB*TB*(TB+1));RZ=CuStaticSharedArray(ST,5*TB*TB*(TB+1))
    lt(pi,pj,pk)=pi+TT*(pj+TT*pk);lfx(fi,fj,fk)=fi+(TB+1)*(fj+TB*fk);lfy(fi,fj,fk)=fi+TB*(fj+(TB+1)*fk);lfz(fi,fj,fk)=fi+TB*(fj+TB*fk)
    rv=Val(RIEM)
    @fastmath @inbounds begin
        dtdx=dt/dx;tid=threadIdx().x;nth=blockDim().x;ox=(blockIdx().x-1)*TB;oy=(blockIdx().y-1)*TB;oz=(blockIdx().z-1)*TB
        t=tid
        while t<=TT*TT*TT
            l=t-1;pi=l%TT;pj=(l÷TT)%TT;pk=l÷(TT*TT)
            _sp5(SP,lt(pi,pj,pk),cons2prim5(loadc5(uold,wrp(ox+pi-1,N),wrp(oy+pj-1,N),wrp(oz+pk-1,N)),gamma,smallr,pfl));t+=nth
        end
        sync_threads();t=tid
        while t<=(TB+2)^3
            l=t-1;ci=l%(TB+2);cj=(l÷(TB+2))%(TB+2);ck=l÷((TB+2)*(TB+2));pi=ci+1;pj=cj+1;pk=ck+1
            m0=_sg5(SP,lt(pi,pj,pk))
            qpx,qmx=ppm_trace5(_sg5(SP,lt(pi-1,pj,pk)),m0,_sg5(SP,lt(pi+1,pj,pk)),gamma,dtdx)
            qpy,qmy=ppm_dir(_sg5(SP,lt(pi,pj-1,pk)),m0,_sg5(SP,lt(pi,pj+1,pk)),gamma,dtdx,Val(2))
            qpz,qmz=ppm_dir(_sg5(SP,lt(pi,pj,pk-1)),m0,_sg5(SP,lt(pi,pj,pk+1)),gamma,dtdx,Val(3))
            inx=(pj>=2&&pj<=TB+1)&&(pk>=2&&pk<=TB+1);iny=(pi>=2&&pi<=TB+1)&&(pk>=2&&pk<=TB+1);inz=(pi>=2&&pi<=TB+1)&&(pj>=2&&pj<=TB+1)
            if inx; pi<=TB+1&&_sp5(LX,lfx(pi-1,pj-2,pk-2),iflo5(qpx,m0,smallr,pfl)); pi>=2&&_sp5(RX,lfx(pi-2,pj-2,pk-2),iflo5(qmx,m0,smallr,pfl)); end
            if iny; pj<=TB+1&&_sp5(LY,lfy(pi-2,pj-1,pk-2),iflo5(qpy,m0,smallr,pfl)); pj>=2&&_sp5(RY,lfy(pi-2,pj-2,pk-2),iflo5(qmy,m0,smallr,pfl)); end
            if inz; pk<=TB+1&&_sp5(LZ,lfz(pi-2,pj-2,pk-1),iflo5(qpz,m0,smallr,pfl)); pk>=2&&_sp5(RZ,lfz(pi-2,pj-2,pk-2),iflo5(qmz,m0,smallr,pfl)); end
            t+=nth
        end
        sync_threads();nfx=(TB+1)*TB*TB;t=tid
        while t<=3*nfx
            if t<=nfx;l=t-1;fi=l%(TB+1);fj=(l÷(TB+1))%TB;fk=l÷((TB+1)*TB);_sp5(LX,lfx(fi,fj,fk),riem5(_sg5(LX,lfx(fi,fj,fk)),_sg5(RX,lfx(fi,fj,fk)),1,gamma,smallr,pfl,rv))
            elseif t<=2*nfx;l=t-1-nfx;fi=l%TB;fj=(l÷TB)%(TB+1);fk=l÷(TB*(TB+1));_sp5(LY,lfy(fi,fj,fk),riem5(_sg5(LY,lfy(fi,fj,fk)),_sg5(RY,lfy(fi,fj,fk)),2,gamma,smallr,pfl,rv))
            else;l=t-1-2*nfx;fi=l%TB;fj=(l÷TB)%TB;fk=l÷(TB*TB);_sp5(LZ,lfz(fi,fj,fk),riem5(_sg5(LZ,lfz(fi,fj,fk)),_sg5(RZ,lfz(fi,fj,fk)),3,gamma,smallr,pfl,rv)) end
            t+=nth
        end
        sync_threads();t=tid
        while t<=TB*TB*TB
            l=t-1;a=l%TB;b=(l÷TB)%TB;c=l÷(TB*TB);gi=wrp(ox+a+1,N);gj=wrp(oy+b+1,N);gk=wrp(oz+c+1,N)
            Fxl=_sg5(LX,lfx(a,b,c));Fxh=_sg5(LX,lfx(a+1,b,c));Fyl=_sg5(LY,lfy(a,b,c));Fyh=_sg5(LY,lfy(a,b+1,c));Fzl=_sg5(LZ,lfz(a,b,c));Fzh=_sg5(LZ,lfz(a,b,c+1))
            u0=loadc5(uold,gi,gj,gk)
            r=ntuple(v->u0[v]+dtdx*((Fxl[v]-Fxh[v])+(Fyl[v]-Fyh[v])+(Fzl[v]-Fzh[v])),5)
            r[1]<smallr && (r=(smallr,0.0f0,0.0f0,0.0f0,pfl/(gamma-1.0f0)))
            if do_turb && r[1]>=turb_min_rho
                ax,ay,az=turb_interp(afield,gi,gj,gk,boxlen/N,boxlen,ramp)
                rhom=max(r[1],smallr);sc2=smallc*smallc;rho=r[1];e=r[5]
                e=max(e-0.5f0*r[2]*r[2]/rhom,rho*sc2);e=max(e-0.5f0*r[3]*r[3]/rhom,rho*sc2);e=max(e-0.5f0*r[4]*r[4]/rhom,rho*sc2)
                m2=r[2]+rhom*ax*dt;m3=r[3]+rhom*ay*dt;m4=r[4]+rhom*az*dt
                e=max(e+0.5f0*m2*m2/rhom,rho*sc2);e=max(e+0.5f0*m3*m3/rhom,rho*sc2);e=max(e+0.5f0*m4*m4/rhom,rho*sc2)
                r=(r[1],m2,m3,m4,e)
            end
            for v in 1:5; unew[gi,gj,gk,v]=r[v]; end
            t+=nth
        end
    end
    return nothing
end

@inline function _hydro_ppm_launch(tv::Val,hv::Val,rv::Val,nthreads,nb,unew,uold,afield,p::Params,pfl,dt,dx,ramp,do_turb)
    @cuda threads=nthreads blocks=(nb,nb,nb) integrator_hydro_ppm!(
        tv,hv,rv,unew,uold,afield,p.gamma,p.smallr,pfl,p.smallc,dt,dx,p.N,p.boxlen,ramp,p.turb_min_rho,do_turb)
end

# ============================================================================
# Streaming integrator (native CUDA.jl): tile the two coalesced dims (x,y), STREAM z.
# A block owns SBX*SBY (x,y) cells over a chunk of SCZ z-planes and marches k, keeping a
# sliding SWZ=5-plane primitive window (x,y tile + 2 halo) in shared. Each plane is
# reconstructed once; z-fluxes are carried between k-steps (1-plane-lagged update).
# Shared ~17.4 KB (vs the cube's 35.7); (x,y) halo is 4x not 8x.
#
# VERDICT (measured): this is the canonical fastest layout for memory/halo-bound stencils,
# but it LOSES to the cube here (512^3: 614 vs 1141 Mcell/s). After the moncen @fastmath
# fix the integrator is compute-bound, not halo-bound, so the halo/shared savings don't
# help; meanwhile streaming's fine-grained per-plane phases (16 owned / 36 inner cells)
# under-fill the warps and multiply barriers (~5 per plane) vs the cube's larger 3D tile
# (64 owned, 512-cell tile, 4 barriers total). Kept as a validated alternative; the cube
# (integrator_tiled!) is the production path. Bit-identical to the cube on Orszag-Tang.
# ============================================================================
const SBX = 4; const SBY = 4              # owned (x,y) cells per block
const STX = SBX + 4; const STY = SBY + 4  # 8x8 (x,y) tile incl. 2-cell halo
const SWZ = 5                             # z-window depth (k-1..k+1 recon needs +-2)
const SCZ = 16                            # z-planes streamed per block (chunk; bounds barriers)
@inline ppi(slot, tx, ty) = slot*(STX*STY) + tx + STX*ty        # prim window cell
@inline fxi(fi, fj) = fi + (SBX+1)*fj                           # x faces (5,4)
@inline fyi(fi, fj) = fi + SBX*fj                               # y faces (4,5)
@inline oi(a, b)    = a + SBX*b                                 # owned (x,y) cell
@inline sg(S, lin)  = ntuple(v -> @inbounds(S[9*lin + v]), 9)
@inline function sp!(S, lin, q)
    @inbounds for v in 1:9
        S[9*lin + v] = q[v]
    end
end
@inline mw(i, N) = mod(i, N)   # 0-based periodic wrap (any offset)

# Reconstruct one z-plane (slots sc center, sm=below, sp=above): MUSCL-Hancock trace ->
# x/y face interface states (LXs/RXs/LYs/RYs), z-face states (Mz=-z, Pz=+z) for owned
# cells, then x/y Riemann -> owned (x,y) flux divergence XY. Internal barriers.
@inline function reconstruct_plane!(PP, LXs, RXs, LYs, RYs, Mz, Pz, XY,
                                    sc::Int, sm::Int, sp::Int, tid::Int, nth::Int,
                                    gamma::Float32, smallr::Float32, pfl::Float32, dtdx::Float32,
                                    ch::Float32, llf_dmin::Float32, llf_pmin::Float32, use_hlld::Bool)
    @fastmath @inbounds begin
        # trace over the inner (SBX+2)^2 (x,y) cells (tx,ty in 1..6)
        t = tid
        while t <= (SBX+2)*(SBY+2)
            l = t-1; tx = l % (SBX+2) + 1; ty = l ÷ (SBX+2) + 1
            m0 = sg(PP, ppi(sc,tx,ty))
            sx = prim_slope(sg(PP,ppi(sc,tx-1,ty)), m0, sg(PP,ppi(sc,tx+1,ty)))
            sy = prim_slope(sg(PP,ppi(sc,tx,ty-1)), m0, sg(PP,ppi(sc,tx,ty+1)))
            sz = prim_slope(sg(PP,ppi(sm,tx,ty)),   m0, sg(PP,ppi(sp,tx,ty)))
            uh = hancock(m0, sx, sy, sz, dtdx, gamma)
            mh = cons2prim(uh, gamma, smallr, pfl)
            if mh[1] <= smallr || mh[5] <= pfl
                mh = m0
            end
            if ty >= 2 && ty <= SBY+1           # owned y row -> x faces
                if tx <= SBX+1
                    sp!(LXs, fxi(tx-1, ty-2), iflo(padd(mh,sx, 1.0f0), m0, smallr, pfl))
                end
                if tx >= 2
                    sp!(RXs, fxi(tx-2, ty-2), iflo(padd(mh,sx,-1.0f0), m0, smallr, pfl))
                end
            end
            if tx >= 2 && tx <= SBX+1           # owned x col -> y faces
                if ty <= SBY+1
                    sp!(LYs, fyi(tx-2, ty-1), iflo(padd(mh,sy, 1.0f0), m0, smallr, pfl))
                end
                if ty >= 2
                    sp!(RYs, fyi(tx-2, ty-2), iflo(padd(mh,sy,-1.0f0), m0, smallr, pfl))
                end
            end
            if (tx >= 2 && tx <= SBX+1) && (ty >= 2 && ty <= SBY+1)   # owned -> z face states
                sp!(Mz, oi(tx-2, ty-2), iflo(padd(mh,sz,-1.0f0), m0, smallr, pfl))
                sp!(Pz, oi(tx-2, ty-2), iflo(padd(mh,sz, 1.0f0), m0, smallr, pfl))
            end
            t += nth
        end
        sync_threads()
        # x,y Riemann (flux written back into L*); then owned (x,y) flux divergence
        nfx = (SBX+1)*SBY; nfy = SBX*(SBY+1)
        t = tid
        while t <= nfx + nfy
            if t <= nfx
                l = t-1; fi = l % (SBX+1); fj = l ÷ (SBX+1)
                F = riemann(sg(LXs,fxi(fi,fj)), sg(RXs,fxi(fi,fj)), 1, gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld)
                sp!(LXs, fxi(fi,fj), F)
            else
                l = t-1-nfx; fi = l % SBX; fj = l ÷ SBX
                F = riemann(sg(LYs,fyi(fi,fj)), sg(RYs,fyi(fi,fj)), 2, gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld)
                sp!(LYs, fyi(fi,fj), F)
            end
            t += nth
        end
        sync_threads()
        t = tid
        while t <= SBX*SBY
            l = t-1; a = l % SBX; b = l ÷ SBX
            Fxl = sg(LXs,fxi(a,b)); Fxh = sg(LXs,fxi(a+1,b))
            Fyl = sg(LYs,fyi(a,b)); Fyh = sg(LYs,fyi(a,b+1))
            sp!(XY, oi(a,b), ntuple(v -> (Fxl[v]-Fxh[v]) + (Fyl[v]-Fyh[v]), 9))
            t += nth
        end
        sync_threads()
    end
    return nothing
end

function integrator_stream!(unew, uold, afield,
                            gamma::Float32, smallr::Float32, pfl::Float32, smallc::Float32,
                            dt::Float32, dx::Float32, ch::Float32, glm_fac::Float32,
                            llf_dmin::Float32, llf_pmin::Float32, use_hlld::Bool, N::Int,
                            boxlen::Float32, ramp::Float32, turb_min_rho::Float32, do_turb::Bool)
    PP  = CuStaticSharedArray(Float32, 9*SWZ*STX*STY)
    LXs = CuStaticSharedArray(Float32, 9*(SBX+1)*SBY); RXs = CuStaticSharedArray(Float32, 9*(SBX+1)*SBY)
    LYs = CuStaticSharedArray(Float32, 9*SBX*(SBY+1)); RYs = CuStaticSharedArray(Float32, 9*SBX*(SBY+1))
    Mz  = CuStaticSharedArray(Float32, 9*SBX*SBY); Pz  = CuStaticSharedArray(Float32, 9*SBX*SBY)
    XY  = CuStaticSharedArray(Float32, 9*SBX*SBY)
    pPz = CuStaticSharedArray(Float32, 9*SBX*SBY); pXY = CuStaticSharedArray(Float32, 9*SBX*SBY)
    pFz = CuStaticSharedArray(Float32, 9*SBX*SBY); Fz  = CuStaticSharedArray(Float32, 9*SBX*SBY)
    @fastmath @inbounds begin
        dtdx = dt/dx
        tid = Int(threadIdx().x); nth = Int(blockDim().x)
        ox = Int(blockIdx().x-1)*SBX; oy = Int(blockIdx().y-1)*SBY
        k0 = Int(blockIdx().z-1)*SCZ            # first owned z-plane of this chunk
        # --- preload window planes (k0-3)..(k0) (slots mod SWZ) ---
        for j in (k0-3):k0
            slot = mod(j, SWZ)
            t = tid
            while t <= STX*STY
                l = t-1; tx = l % STX; ty = l ÷ STX
                gx = mw(ox+tx-2, N); gy = mw(oy+ty-2, N); gz = mw(j, N)
                sp!(PP, ppi(slot,tx,ty), cons2prim(loadc(uold, gx+1, gy+1, gz+1), gamma, smallr, pfl))
                t += nth
            end
        end
        sync_threads()
        # --- seed: reconstruct plane (k0-1) -> pPz (only the +z face state is kept) ---
        reconstruct_plane!(PP, LXs, RXs, LYs, RYs, Mz, Pz, XY,
                           mod(k0-1,SWZ), mod(k0-2,SWZ), mod(k0,SWZ), tid, nth,
                           gamma, smallr, pfl, dtdx, ch, llf_dmin, llf_pmin, use_hlld)
        t = tid
        while t <= SBX*SBY
            sp!(pPz, t-1, sg(Pz, t-1)); t += nth
        end
        sync_threads()
        # --- stream the chunk [k0, k0+SCZ): reconstruct plane i, carry z-flux, update plane i-1 ---
        i = k0
        while i <= k0 + SCZ
            # load leading plane (i+1) into its ring slot
            slot = mod(i+1, SWZ)
            t = tid
            while t <= STX*STY
                l = t-1; tx = l % STX; ty = l ÷ STX
                gx = mw(ox+tx-2, N); gy = mw(oy+ty-2, N); gz = mw(i+1, N)
                sp!(PP, ppi(slot,tx,ty), cons2prim(loadc(uold, gx+1, gy+1, gz+1), gamma, smallr, pfl))
                t += nth
            end
            sync_threads()
            reconstruct_plane!(PP, LXs, RXs, LYs, RYs, Mz, Pz, XY,
                               mod(i,SWZ), mod(i-1,SWZ), mod(i+1,SWZ), tid, nth,
                               gamma, smallr, pfl, dtdx, ch, llf_dmin, llf_pmin, use_hlld)
            # z-flux at face (i-1/2) = riemann(prev plane +z, this plane -z)
            t = tid
            while t <= SBX*SBY
                l = t-1
                sp!(Fz, l, riemann(sg(pPz,l), sg(Mz,l), 3, gamma, ch, smallr, pfl, llf_dmin, llf_pmin, use_hlld))
                t += nth
            end
            sync_threads()
            # update plane (i-1): xy-div(prev) + (Fz_lo(prev) - Fz_hi(now))
            if i >= k0 + 1
                gzm = mw(i-1, N)
                t = tid
                while t <= SBX*SBY
                    l = t-1; a = l % SBX; b = l ÷ SBX
                    gx = mw(ox+a, N); gy = mw(oy+b, N)
                    u0 = loadc(uold, gx+1, gy+1, gzm+1)
                    xy = sg(pXY, l); flo = sg(pFz, l); fhi = sg(Fz, l)
                    r1 = ntuple(v -> u0[v] + dtdx*(xy[v] + (flo[v]-fhi[v])), 9)
                    if r1[1] < smallr
                        emag = 0.5f0*(r1[6]*r1[6] + r1[7]*r1[7] + r1[8]*r1[8])
                        r1 = (smallr, 0.0f0, 0.0f0, 0.0f0, pfl/(gamma-1.0f0) + emag, r1[6], r1[7], r1[8], r1[9])
                    end
                    r = (r1[1],r1[2],r1[3],r1[4],r1[5],r1[6],r1[7],r1[8], r1[9]*glm_fac)
                    if do_turb && r[1] >= turb_min_rho
                        ax, ay, az = turb_interp(afield, gx+1, gy+1, gzm+1, boxlen/N, boxlen, ramp)
                        rhom = max(r[1], smallr); sc2 = smallc*smallc; rho = r[1]; e = r[5]
                        e = max(e - 0.5f0*r[2]*r[2]/rhom, rho*sc2); e = max(e - 0.5f0*r[3]*r[3]/rhom, rho*sc2); e = max(e - 0.5f0*r[4]*r[4]/rhom, rho*sc2)
                        m2 = r[2] + rhom*ax*dt; m3 = r[3] + rhom*ay*dt; m4 = r[4] + rhom*az*dt
                        e = max(e + 0.5f0*m2*m2/rhom, rho*sc2); e = max(e + 0.5f0*m3*m3/rhom, rho*sc2); e = max(e + 0.5f0*m4*m4/rhom, rho*sc2)
                        r = (r[1], m2, m3, m4, e, r[6], r[7], r[8], r[9])
                    end
                    for v in 1:9
                        unew[gx+1, gy+1, gzm+1, v] = r[v]
                    end
                    t += nth
                end
            end
            sync_threads()
            # carry: prev <- current (z +face state, xy divergence, z low-flux)
            t = tid
            while t <= SBX*SBY
                l = t-1
                sp!(pPz, l, sg(Pz, l)); sp!(pXY, l, sg(XY, l)); sp!(pFz, l, sg(Fz, l))
                t += nth
            end
            sync_threads()
            i += 1
        end
    end
    return nothing
end

function step_stream!(uold, unew, afield, p::Params, dt::Float32, t::Float32;
                      use_hlld::Bool=true, do_turb::Bool=false, nthreads::Int=64)
    N = p.N; dx = dxof(p); pfl = pfloor(p)
    @assert N % SBX == 0 && N % SBY == 0 && N % SCZ == 0 "N must be a multiple of $SBX and $SCZ"
    ch = p.courant*dx/dt/Float32(NDIM)*p.glm_ch_scale
    glm_fac = p.glm_cp_coef > 0 ? exp(-(ch*ch/(p.glm_cp_coef*p.boxlen*ch))*dt) : 1.0f0
    ramp = min(t/p.turb_T, 1.0f0)
    @cuda threads=nthreads blocks=(N÷SBX, N÷SBY, N÷SCZ) integrator_stream!(
        unew, uold, afield, p.gamma, p.smallr, pfl, p.smallc, dt, dx, ch, glm_fac,
        p.switch_llf_dmin, p.switch_llf_pmin, use_hlld, N, p.boxlen, ramp,
        p.turb_min_rho, do_turb)
    return nothing
end

# ---- dt reduction (ctot per cell -> max) -----------------------------------
@kernel function ctot_kernel!(ct, @Const(u), gamma::Float32, smallr::Float32, pfl::Float32)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        q = cons2prim(loadc(u, i, j, k), gamma, smallr, pfl)
        ct[i, j, k] = (abs(q[2]) + fast_speed(q, gamma, q[6])) +
                      (abs(q[3]) + fast_speed(q, gamma, q[7])) +
                      (abs(q[4]) + fast_speed(q, gamma, q[8]))
    end
end

function compute_dt(uold, ct, p::Params, be)
    ctot_kernel!(be)(ct, uold, p.gamma, p.smallr, pfloor(p); ndrange=(p.N,p.N,p.N))
    KA.synchronize(be)
    cmax = maximum(ct)
    return p.courant * dxof(p) / cmax
end

# ---- multi-kernel step (validation path) -----------------------------------
function step_multi!(uold, unew, rec, flx, ct, afield, p::Params, dt::Float32, t::Float32,
                     be; use_hlld::Bool=true, do_turb::Bool=false)
    N = p.N; dx = dxof(p); dtdx = dt/dx; pfl = pfloor(p)
    ch = p.courant*dx/dt/Float32(NDIM)*p.glm_ch_scale
    glm_fac = p.glm_cp_coef > 0 ? exp(-(ch*ch/(p.glm_cp_coef*p.boxlen*ch))*dt) : 1.0f0
    ramp = min(t/p.turb_T, 1.0f0)
    recon_kernel!(be)(rec, uold, p.gamma, p.smallr, pfl, dtdx, N; ndrange=(N,N,N))
    flux_kernel!(be)(flx, rec, p.gamma, ch, p.smallr, pfl, p.switch_llf_dmin, p.switch_llf_pmin, use_hlld, N; ndrange=(N,N,N))
    update_kernel!(be)(unew, uold, flx, afield, dtdx, glm_fac, N, p.boxlen, ramp, dt,
                       p.smallr, p.smallc, p.gamma, pfl, p.turb_min_rho, do_turb; ndrange=(N,N,N))
    KA.synchronize(be)
end

# ---- diagnostics -----------------------------------------------------------
# cell-centred |div B| (central differences) and mass-weighted v_rms (Mach), on GPU.
@kernel function divb_kernel!(db, @Const(u), N::Int, dx::Float32)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        bxp = u[wrp(i+1,N),j,k,6]; bxm = u[wrp(i-1,N),j,k,6]
        byp = u[i,wrp(j+1,N),k,7]; bym = u[i,wrp(j-1,N),k,7]
        bzp = u[i,j,wrp(k+1,N),8]; bzm = u[i,j,wrp(k-1,N),8]
        db[i,j,k] = abs((bxp-bxm) + (byp-bym) + (bzp-bzm)) / (2.0f0*dx)
    end
end

function diagnostics(u, ct, p::Params, be)
    N = p.N; pfl = pfloor(p)
    # reuse ct as scratch for divB
    divb_kernel!(be)(ct, u, N, dxof(p); ndrange=(N,N,N)); KA.synchronize(be)
    divbmax = maximum(ct)
    rho = @view u[:,:,:,1]
    mx = @view u[:,:,:,2]; my = @view u[:,:,:,3]; mz = @view u[:,:,:,4]
    mass = sum(rho)
    ke2 = sum(@. (mx*mx + my*my + mz*mz) / max(rho, p.smallr))   # sum rho*v^2
    vrms = sqrt(ke2/mass)
    return (mass=mass, vrms=vrms, mach=vrms/p.cs0, divbmax=divbmax,
            rhomin=minimum(rho), finite=all(isfinite, u))
end

# ---- Orszag-Tang IC (2D, validation) ---------------------------------------
@kernel function ic_ot_kernel!(u, N::Int, gamma::Float32)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        twopi = 2.0f0*Float32(pi)
        x = (i - 0.5f0)/N; y = (j - 0.5f0)/N
        rho = 25.0f0/(36.0f0*Float32(pi))
        pr  = 5.0f0/(12.0f0*Float32(pi))
        vx = -sin(twopi*y); vy = sin(twopi*x); vz = 0.0f0
        sb = 1.0f0/sqrt(4.0f0*Float32(pi))
        bx = -sb*sin(twopi*y); by = sb*sin(2.0f0*twopi*x); bz = 0.0f0
        ekin = 0.5f0*rho*(vx*vx+vy*vy+vz*vz); emag = 0.5f0*(bx*bx+by*by+bz*bz)
        u[i,j,k,1]=rho; u[i,j,k,2]=rho*vx; u[i,j,k,3]=rho*vy; u[i,j,k,4]=rho*vz
        u[i,j,k,5]=pr/(gamma-1.0f0)+ekin+emag
        u[i,j,k,6]=bx; u[i,j,k,7]=by; u[i,j,k,8]=bz; u[i,j,k,9]=0.0f0
    end
end

# Validation driver: Orszag-Tang to t_end, no forcing. Returns final diagnostics.
# Orszag-Tang on the portable multi-kernel path. backend=:gpu or :cpu (same KA kernels);
# the CPU path pins threads via `pin`. Useful as a cross-platform correctness check.
function run_ot(; N::Int=128, t_end::Float32=0.5f0, use_hlld::Bool=true,
                  backend::Symbol=:gpu, pin::Symbol=:numa)
    if backend === :gpu
        be = CUDABackend()
    elseif backend === :cpu
        pin_threads!(pin); be = KA.CPU()
    else
        error("backend must be :gpu or :cpu")
    end
    p = Params(N=N, gamma=5.0f0/3.0f0, cs0=1.0f0)
    u1 = KA.zeros(be, Float32, N,N,N,NV); u2 = KA.zeros(be, Float32, N,N,N,NV)
    rec = KA.zeros(be, Float32, N,N,N,36); flx = KA.zeros(be, Float32, N,N,N,27)
    ct  = KA.zeros(be, Float32, N,N,N)
    afield = KA.zeros(be, Float32, 3,TURB_GS,TURB_GS,TURB_GS)
    ic_ot_kernel!(be)(u1, N, p.gamma; ndrange=(N,N,N)); KA.synchronize(be)
    t = 0.0f0; step = 0
    while t < t_end
        dt = compute_dt(u1, ct, p, be)
        dt = min(dt, t_end - t)
        step_multi!(u1, u2, rec, flx, ct, afield, p, dt, t, be; use_hlld=use_hlld, do_turb=false)
        u1, u2 = u2, u1
        t += dt; step += 1
    end
    d = diagnostics(u1, ct, p, be)
    @printf("OT[%s] N=%d HLLD=%s: steps=%d t=%.3f vrms=%.4f divBmax=%.3e rhomin=%.4f finite=%s\n",
            backend, N, use_hlld, step, t, d.vrms, d.divbmax, d.rhomin, d.finite)
    return d
end

# ============================================================================
# GPU-resident OU turbulent forcing (Fourier on a 64^3 grid via CUFFT).
# Clean OU in k-space: Fhat <- Fhat*(1-dt/T) + sqrt(2 dt/T)*sqrt(P(k))*Helmholtz(gauss).
# Amplitude `amp` is calibrated to hit Mach~10. iFFT -> real accel field, applied in update.
# ============================================================================
@inline kfreq(i::Int, n::Int) = i-1 <= n÷2 ? Float32(i-1) : Float32(i-1-n)

# Precompute power spectrum sqrt(P)=|k|^-1 (P=k^-2), 1<=|k|<=n/2, and unit-k for Helmholtz.
@kernel function ou_init_kernel!(sqp, unitk, n::Int)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        kx = kfreq(i,n); ky = kfreq(j,n); kz = kfreq(k,n)
        kk = sqrt(kx*kx + ky*ky + kz*kz)
        if kk >= 1.0f0 && kk <= Float32(n÷2)
            sqp[i,j,k] = 1.0f0/kk            # sqrt(P) = |k|^-1
            unitk[1,i,j,k]=kx/kk; unitk[2,i,j,k]=ky/kk; unitk[3,i,j,k]=kz/kk
        else
            sqp[i,j,k]=0.0f0
            unitk[1,i,j,k]=0.0f0; unitk[2,i,j,k]=0.0f0; unitk[3,i,j,k]=0.0f0
        end
    end
end

# One OU step in k-space: read 6 Gaussians/mode (3 complex), Helmholtz-project, OU-update Fhat.
@kernel function ou_step_kernel!(Fhat, @Const(g), @Const(sqp), @Const(unitk),
                                 alpha::Float32, beta::Float32, comp::Float32, sol::Float32)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        s = sqp[i,j,k]
        # raw complex increment per component
        dwx = ComplexF32(g[1,i,j,k], g[2,i,j,k])*s
        dwy = ComplexF32(g[3,i,j,k], g[4,i,j,k])*s
        dwz = ComplexF32(g[5,i,j,k], g[6,i,j,k])*s
        ux = unitk[1,i,j,k]; uy = unitk[2,i,j,k]; uz = unitk[3,i,j,k]
        kdotF = ux*dwx + uy*dwy + uz*dwz          # k̂·dW (complex)
        cx = ux*kdotF; cy = uy*kdotF; cz = uz*kdotF   # compressive part
        px = comp*cx + sol*(dwx-cx)                # Helmholtz-mixed
        py = comp*cy + sol*(dwy-cy)
        pz = comp*cz + sol*(dwz-cz)
        Fhat[1,i,j,k] = alpha*Fhat[1,i,j,k] + beta*px
        Fhat[2,i,j,k] = alpha*Fhat[2,i,j,k] + beta*py
        Fhat[3,i,j,k] = alpha*Fhat[3,i,j,k] + beta*pz
    end
end

# Backend-agnostic: arrays are CuArray on a GPU backend, Array on CPU(); the iFFT plan is
# CUFFT or FFTW (chosen by array type via AbstractFFTs); fields untyped so one struct serves both.
mutable struct Forcing
    Fhat                               # (3,n,n,n) k-space OU field
    sqp                                # sqrt(P(k))
    unitk                              # (3,n,n,n)
    afield                             # (3,n,n,n) real accel
    gbuf                               # (6,n,n,n) Gaussian scratch
    tmp                                # iFFT scratch
    plan                               # reused inverse-FFT plan (CUFFT or FFTW)
    amp::Float32                       # calibration amplitude
    next_time::Float32
end

# Portable Gaussian fill: CUDA's device RNG for CuArray, Random for Array.
_randn!(g::CuArray) = CUDA.randn!(g)
_randn!(g::AbstractArray) = Random.randn!(g)

function Forcing(p::Params, be; amp::Float32=p.turb_rms)
    n = TURB_GS
    Fhat = KA.zeros(be, ComplexF32, 3,n,n,n)
    sqp = KA.zeros(be, Float32, n,n,n); unitk = KA.zeros(be, Float32, 3,n,n,n)
    afield = KA.zeros(be, Float32, 3,n,n,n); gbuf = KA.zeros(be, Float32, 6,n,n,n)
    tmp = KA.zeros(be, ComplexF32, n,n,n)
    plan = AbstractFFTs.plan_bfft!(tmp)        # CUFFT for CuArray, FFTW for Array
    ou_init_kernel!(be)(sqp, unitk, n; ndrange=(n,n,n)); KA.synchronize(be)
    f = Forcing(Fhat, sqp, unitk, afield, gbuf, tmp, plan, amp, 0.0f0)
    return f
end

# Advance the OU field by one turb_dt and refresh the real accel field.
function ou_advance!(f::Forcing, p::Params, be)
    n = TURB_GS
    dt = p.turb_T/Float32(p.turb_Ndt)
    alpha = 1.0f0 - dt/p.turb_T
    beta  = sqrt(2.0f0*dt/p.turb_T)
    _randn!(f.gbuf)
    ou_step_kernel!(be)(f.Fhat, f.gbuf, f.sqp, f.unitk, alpha, beta, p.comp_frac, 1.0f0-p.comp_frac; ndrange=(n,n,n))
    KA.synchronize(be)
    for c in 1:3
        @views copyto!(f.tmp, f.Fhat[c,:,:,:])
        f.plan * f.tmp                              # in-place inverse FFT
        @views f.afield[c,:,:,:] .= real.(f.tmp) .* (f.amp / Float32(n)^3)
    end
    KA.synchronize(be)
    f.next_time += dt
end

# Uniform turbulence IC: rho0, p0=rho0*cs0^2/gamma, v=0, Bx=b0, By=Bz=0, psi=0.
@kernel function ic_uniform_kernel!(u, rho0::Float32, p0::Float32, b0::Float32, gamma::Float32)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        emag = 0.5f0*b0*b0
        u[i,j,k,1]=rho0; u[i,j,k,2]=0.0f0; u[i,j,k,3]=0.0f0; u[i,j,k,4]=0.0f0
        u[i,j,k,5]=p0/(gamma-1.0f0)+emag
        u[i,j,k,6]=b0; u[i,j,k,7]=0.0f0; u[i,j,k,8]=0.0f0; u[i,j,k,9]=0.0f0
    end
end

# Pin Julia threads for the CPU backend on NUMA hardware. On multi-socket / many-NUMA-node
# boxes :numa (or :cores) gives the best locality and stops the OS migrating threads across
# nodes; measured ~1.6x on a dual-EPYC (16 NUMA nodes). Pass :none to skip. No-op single-thread.
# NB: don't also use `numactl --interleave` — it round-robins pages and defeats the pinning.
function pin_threads!(strategy::Symbol=:numa)
    (Threads.nthreads() == 1 || strategy === :none) && return false
    try
        ThreadPinning.pinthreads(strategy); return true
    catch e
        @warn "ThreadPinning.pinthreads($strategy) failed; continuing unpinned" exception=e
        return false
    end
end

# Driven-turbulence run. backend=:gpu uses the fast no-scratch tiled integrator (GPU only);
# backend=:cpu runs the portable KA multi-kernel path on CPU() (pins threads via `pin`).
# Same physics either way; the forcing FFT dispatches CUFFT/FFTW automatically.
function run_turb(; N::Int=128, t_end::Float32=0.6f0, amp::Float32=130.0f0, use_hlld::Bool=true,
                    ncontrol::Int=200, verbose::Bool=true, nstepmax::Int=20000, nthreads::Int=192,
                    backend::Symbol=:gpu, pin::Symbol=:numa, solver::Symbol=:plm)
    gpu = backend === :gpu
    if gpu
        be = CUDABackend()
    elseif backend === :cpu
        pin_threads!(pin)
        Threads.nthreads() == 1 && @warn "CPU backend with 1 thread — start Julia with `-t <N>` (≈3/4 of cores on NUMA boxes)"
        be = KA.CPU()
    else
        error("backend must be :gpu or :cpu")
    end
    p0_ = (1.0f0)*(1.0f0)^2/1.01f0   # rho0*cs0^2/gamma with defaults
    p = Params(N=N, turb_rms=amp, switch_llf_dmin=0.01f0, switch_llf_pmin=0.01f0*p0_)
    u1 = KA.zeros(be, Float32, N,N,N,NV); u2 = KA.zeros(be, Float32, N,N,N,NV)
    ct = KA.zeros(be, Float32, N,N,N)
    # the CPU multi-kernel path needs reconstruction/flux scratch; the GPU tiled path is scratch-free
    rec = gpu ? nothing : KA.zeros(be, Float32, N,N,N,36)
    flx = gpu ? nothing : KA.zeros(be, Float32, N,N,N,27)
    p0 = p.rho0*p.cs0^2/p.gamma
    ic_uniform_kernel!(be)(u1, p.rho0, p0, p.b0, p.gamma; ndrange=(N,N,N)); KA.synchronize(be)
    f = Forcing(p, be; amp=amp)
    ou_advance!(f, p, be)                  # seed the first field
    t = 0.0f0; step = 0
    while t < t_end && step < nstepmax
        while t >= f.next_time
            ou_advance!(f, p, be)
        end
        dt = compute_dt(u1, ct, p, be)
        dt = min(dt, f.next_time - t, t_end - t)
        if gpu && solver === :plm        # Hancock + HLL + GLM + PLM, f16 shared; TB auto (6 if N%6==0 else 4)
            step_plm!(u1, u2, f.afield, p, dt, t; do_turb=true, nthreads=nthreads, tb=0, half=true, riemann=:hll)
        elseif gpu
            step_tiled!(u1, u2, f.afield, p, dt, t; use_hlld=use_hlld, do_turb=true, nthreads=nthreads)
        else
            step_multi!(u1, u2, rec, flx, ct, f.afield, p, dt, t, be; use_hlld=use_hlld, do_turb=true)
        end
        u1, u2 = u2, u1
        t += dt; step += 1
        if verbose && step % ncontrol == 0
            d = diagnostics(u1, ct, p, be)
            @printf("  step=%d t=%.4f dt=%.2e Mach=%.3f divBmax=%.2e rhomin=%.3e finite=%s\n",
                    step, t, dt, d.mach, d.divbmax, d.rhomin, d.finite)
        end
    end
    d = diagnostics(u1, ct, p, be)
    @printf("TURB[%s] N=%d amp=%.0f: steps=%d t=%.3f Mach=%.3f divBmax=%.2e rhomin=%.3e finite=%s\n",
            backend, N, amp, step, t, d.mach, d.divbmax, d.rhomin, d.finite)
    return d
end

# ============================================================================
# Benchmark: fused integrator throughput + memory (the gates). Only uold/unew + a tiny
# ct scratch (folded later) + the 3 MB afield are allocated -> bytes/cell ~= 72.
# ============================================================================
function bench_fused(; N::Int=256, nsteps::Int=30, warm::Int=5, use_hlld::Bool=true, do_turb::Bool=true)
    be = CUDABackend()
    p = Params(N=N, switch_llf_dmin=0.01f0, switch_llf_pmin=0.01f0/1.01f0)
    GC.gc(); CUDA.reclaim()
    avail0 = CUDA.available_memory()
    u1 = CUDA.zeros(Float32, N,N,N,NV); u2 = CUDA.zeros(Float32, N,N,N,NV)
    afield = CUDA.zeros(Float32, 3,TURB_GS,TURB_GS,TURB_GS)
    avail1 = CUDA.available_memory()
    bytes = avail0 - avail1
    bpc = bytes / N^3
    p0 = p.rho0*p.cs0^2/p.gamma
    ic_uniform_kernel!(be)(u1, p.rho0, p0, p.b0, p.gamma; ndrange=(N,N,N)); KA.synchronize(be)
    dt = 1.0f-4
    for s in 1:warm
        step_fused!(u1, u2, afield, p, dt, 0.1f0, be; use_hlld=use_hlld, do_turb=do_turb); u1,u2 = u2,u1
    end
    CUDA.synchronize()
    el = CUDA.@elapsed begin
        for s in 1:nsteps
            step_fused!(u1, u2, afield, p, dt, 0.1f0, be; use_hlld=use_hlld, do_turb=do_turb); u1,u2 = u2,u1
        end
        CUDA.synchronize()
    end
    mcell = N^3 * nsteps / el / 1e6
    fin = all(isfinite, Array(@view u1[1:2,1:2,1:2,:]))
    @printf("BENCH N=%d HLLD=%s: %.1f Mcell/s (%.2f ms/step) | mem=%.0f B/cell (alloc %.2f GB) | finite=%s\n",
            N, use_hlld, mcell, el/nsteps*1e3, bpc, bytes/2^30, fin)
    return (mcell=mcell, bpc=bpc)
end

# Validate the tiled kernel on Orszag-Tang (should match fused/multi-kernel).
function validate_tiled_ot(; N::Int=64, t_end::Float32=0.2f0, lean::Bool=false)
    be = CUDABackend()
    p = Params(N=N, gamma=5.0f0/3.0f0, cs0=1.0f0)
    afield = CUDA.zeros(Float32, 3,TURB_GS,TURB_GS,TURB_GS)
    ct = CUDA.zeros(Float32, N,N,N)
    u1 = CUDA.zeros(Float32, N,N,N,NV); u2 = CUDA.zeros(Float32, N,N,N,NV)
    ic_ot_kernel!(be)(u1, N, p.gamma; ndrange=(N,N,N)); KA.synchronize(be)
    t=0.0f0; step=0
    while t < t_end
        dt = compute_dt(u1, ct, p, be); dt = min(dt, t_end-t)
        step_tiled!(u1, u2, afield, p, dt, t; use_hlld=true, do_turb=false, lean=lean); CUDA.synchronize(); u1,u2=u2,u1
        t += dt; step += 1
    end
    d = diagnostics(u1, ct, p, be)
    @printf("TILED-OT N=%d: steps=%d t=%.3f vrms=%.4f divBmax=%.3e rhomin=%.4f finite=%s\n",
            N, step, t, d.vrms, d.divbmax, d.rhomin, d.finite)
    return d
end

function bench_tiled(; N::Int=256, nsteps::Int=30, warm::Int=5, use_hlld::Bool=true, do_turb::Bool=true, nthreads::Int=192, maxregs::Int=0, lean::Bool=false)
    p = Params(N=N, switch_llf_dmin=0.01f0, switch_llf_pmin=0.01f0/1.01f0)
    be = CUDABackend()
    GC.gc(); CUDA.reclaim()
    avail0 = CUDA.available_memory()
    u1 = CUDA.zeros(Float32, N,N,N,NV); u2 = CUDA.zeros(Float32, N,N,N,NV)
    afield = CUDA.zeros(Float32, 3,TURB_GS,TURB_GS,TURB_GS)
    bytes = avail0 - CUDA.available_memory(); bpc = bytes / N^3
    p0 = p.rho0*p.cs0^2/p.gamma
    ic_uniform_kernel!(be)(u1, p.rho0, p0, p.b0, p.gamma; ndrange=(N,N,N)); KA.synchronize(be)
    dt = 1.0f-4
    for s in 1:warm
        step_tiled!(u1, u2, afield, p, dt, 0.1f0; use_hlld=use_hlld, do_turb=do_turb, nthreads=nthreads, maxregs=maxregs, lean=lean); u1,u2=u2,u1
    end
    CUDA.synchronize()
    el = CUDA.@elapsed begin
        for s in 1:nsteps
            step_tiled!(u1, u2, afield, p, dt, 0.1f0; use_hlld=use_hlld, do_turb=do_turb, nthreads=nthreads, maxregs=maxregs, lean=lean); u1,u2=u2,u1
        end
        CUDA.synchronize()
    end
    mcell = N^3 * nsteps / el / 1e6
    fin = all(isfinite, Array(@view u1[1:2,1:2,1:2,:]))
    @printf("TILED N=%d HLLD=%s thr=%d: %.1f Mcell/s (%.2f ms/step) | mem=%.0f B/cell | finite=%s\n",
            N, use_hlld, nthreads, mcell, el/nsteps*1e3, bpc, fin)
    return (mcell=mcell, bpc=bpc)
end

# Validate the fused kernel against the multi-kernel path on Orszag-Tang.
function validate_fused_ot(; N::Int=64, t_end::Float32=0.2f0)
    be = CUDABackend()
    p = Params(N=N, gamma=5.0f0/3.0f0, cs0=1.0f0)
    afield = CUDA.zeros(Float32, 3,TURB_GS,TURB_GS,TURB_GS)
    ct = CUDA.zeros(Float32, N,N,N)
    u1 = CUDA.zeros(Float32, N,N,N,NV); u2 = CUDA.zeros(Float32, N,N,N,NV)
    ic_ot_kernel!(be)(u1, N, p.gamma; ndrange=(N,N,N)); KA.synchronize(be)
    t=0.0f0; step=0
    while t < t_end
        dt = compute_dt(u1, ct, p, be); dt = min(dt, t_end-t)
        step_fused!(u1, u2, afield, p, dt, t, be; use_hlld=true, do_turb=false); u1,u2=u2,u1
        t += dt; step += 1
    end
    d = diagnostics(u1, ct, p, be)
    @printf("FUSED-OT N=%d: steps=%d t=%.3f vrms=%.4f divBmax=%.3e rhomin=%.4f finite=%s\n",
            N, step, t, d.vrms, d.divbmax, d.rhomin, d.finite)
    return d
end

end # module
