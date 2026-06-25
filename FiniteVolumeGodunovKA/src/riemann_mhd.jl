# HLLD Riemann solver for GLM-MHD (Miyoshi & Kusano 2005), keyed to GLMMHD.
#
# Resolves the fast, Alfvén, and contact waves (5 waves, 4 intermediate states) — far sharper
# than LLF for MHD. The GLM normal-field / ψ subsystem (waves ±ch) is solved first to give the
# interface Bx and ψ; the remaining 7-variable ideal-MHD system runs HLLD with that constant Bx.
# Branch-free (ifelse region selection + degeneracy guards) → GPU/SIMD ready.

@inline function _fastmhd(γ, ρ, p, Bx, By, Bz)
    a2 = γ * p / ρ; b2 = (Bx*Bx + By*By + Bz*Bz) / ρ; bx2 = Bx*Bx / ρ
    return sqrt(0.5f0 * (a2 + b2 + sqrt(max(0f0, (a2 + b2)*(a2 + b2) - 4f0*a2*bx2))))
end

# 5-region branch-free selection over a 7-component MHD state/flux tuple.
@inline _sel5(mL, FL, msL, FsL, mM, FssL, msR, FssR, mR, FsR, FR) =
    ifelse(mL, FL, ifelse(msL, FsL, ifelse(mM, FssL, ifelse(msR, FssR, ifelse(mR, FsR, FR)))))

@inline function riemann(::HLLD, s::GLMMHD, WL::NTuple{9,T}, WR::NTuple{9,T}) where {T}
    γ = s.γ; ch = s.ch
    ρL, uL, vL, wL, pL, BxL, ByL, BzL, ψL = WL
    ρR, uR, vR, wR, pR, BxR, ByR, BzR, ψR = WR

    # GLM subsystem: interface normal field & ψ (2-wave, speed ±ch).
    Bx  = 0.5f0 * (BxL + BxR) - (0.5f0 / ch) * (ψR - ψL)
    ψi  = 0.5f0 * (ψL + ψR) - (0.5f0 * ch) * (BxR - BxL)
    Bx2 = Bx * Bx

    BL2 = Bx2 + ByL*ByL + BzL*BzL; BR2 = Bx2 + ByR*ByR + BzR*BzR
    pTL = pL + 0.5f0 * BL2;        pTR = pR + 0.5f0 * BR2
    EL  = pL/(γ-1) + 0.5f0*ρL*(uL*uL+vL*vL+wL*wL) + 0.5f0*BL2
    ER  = pR/(γ-1) + 0.5f0*ρR*(uR*uR+vR*vR+wR*wR) + 0.5f0*BR2

    cfL = _fastmhd(γ, ρL, pL, Bx, ByL, BzL); cfR = _fastmhd(γ, ρR, pR, Bx, ByR, BzR)
    SL = min(uL, uR) - max(cfL, cfR); SR = max(uL, uR) + max(cfL, cfR)
    sL = SL - uL; sR = SR - uR                       # SK - uK (both signed away from the contact)

    # contact speed SM and the (constant) total pressure pT*.
    den = sR*ρR - sL*ρL
    SM  = (sR*ρR*uR - sL*ρL*uL - pTR + pTL) / den
    pT  = (sR*ρR*pTL - sL*ρL*pTR + ρL*ρR*sR*sL*(uR - uL)) / den

    # left/right physical MHD fluxes (constant normal Bx) — 7 components (ρ,ρu,ρv,ρw,E,By,Bz).
    vBL = uL*Bx + vL*ByL + wL*BzL; vBR = uR*Bx + vR*ByR + wR*BzR
    FL = (ρL*uL, ρL*uL*uL + pTL - Bx2, ρL*uL*vL - Bx*ByL, ρL*uL*wL - Bx*BzL,
          (EL + pTL)*uL - Bx*vBL, ByL*uL - Bx*vL, BzL*uL - Bx*wL)
    FR = (ρR*uR, ρR*uR*uR + pTR - Bx2, ρR*uR*vR - Bx*ByR, ρR*uR*wR - Bx*BzR,
          (ER + pTR)*uR - Bx*vBR, ByR*uR - Bx*vR, BzR*uR - Bx*wR)
    UL = (ρL, ρL*uL, ρL*vL, ρL*wL, EL, ByL, BzL)
    UR = (ρR, ρR*uR, ρR*vR, ρR*wR, ER, ByR, BzR)

    # star states (between fast and Alfvén waves).
    tiny = 1f-12
    ρLs = ρL * sL / (SL - SM); ρRs = ρR * sR / (SR - SM)
    # Degeneracy: the Miyoshi-Kusano transverse formula is valid only when dK = ρK(SK-uK)(SK-SM)
    # - Bx² > 0. When it is ≤ 0 (strong relative field / Alfvén ≈ entropy wave), fall back to the
    # un-rotated limit (v*=v, By*=By) — NOT 1/dK, which would be garbage.
    nL = ρL*sL*(SL - SM); nR = ρR*sR*(SR - SM)
    dL = nL - Bx2; dR = nR - Bx2
    okL = dL > 1f-8 * nL; okR = dR > 1f-8 * nR
    iL = ifelse(okL, 1f0/dL, 0f0); iR = ifelse(okR, 1f0/dR, 0f0)
    nbL = ρL*sL*sL - Bx2; nbR = ρR*sR*sR - Bx2
    vLs = vL - Bx*ByL*(SM - uL)*iL; wLs = wL - Bx*BzL*(SM - uL)*iL
    vRs = vR - Bx*ByR*(SM - uR)*iR; wRs = wR - Bx*BzR*(SM - uR)*iR
    ByLs = ifelse(okL, ByL*nbL*iL, ByL); BzLs = ifelse(okL, BzL*nbL*iL, BzL)
    ByRs = ifelse(okR, ByR*nbR*iR, ByR); BzRs = ifelse(okR, BzR*nbR*iR, BzR)
    vBLs = SM*Bx + vLs*ByLs + wLs*BzLs; vBRs = SM*Bx + vRs*ByRs + wRs*BzRs
    # E is an energy DENSITY → the convective term is (SK-uK)·EK, NOT ·ρK·EK.
    ELs = (sL*EL - pTL*uL + pT*SM + Bx*(vBL - vBLs)) / (SL - SM)
    ERs = (sR*ER - pTR*uR + pT*SM + Bx*(vBR - vBRs)) / (SR - SM)
    ULs = (ρLs, ρLs*SM, ρLs*vLs, ρLs*wLs, ELs, ByLs, BzLs)
    URs = (ρRs, ρRs*SM, ρRs*vRs, ρRs*wRs, ERs, ByRs, BzRs)

    # double-star states (between Alfvén waves). Guard the sqrt: ρ*K is physically positive in
    # the regions where these states are selected, but is computed unconditionally (branch-free).
    rLs = sqrt(max(ρLs, tiny)); rRs = sqrt(max(ρRs, tiny)); sB = sign(Bx); inv_r = 1f0 / (rLs + rRs)
    vss = (rLs*vLs + rRs*vRs + (ByRs - ByLs)*sB) * inv_r
    wss = (rLs*wLs + rRs*wRs + (BzRs - BzLs)*sB) * inv_r
    Byss = (rLs*ByRs + rRs*ByLs + rLs*rRs*(vRs - vLs)*sB) * inv_r
    Bzss = (rLs*BzRs + rRs*BzLs + rLs*rRs*(wRs - wLs)*sB) * inv_r
    vBss = SM*Bx + vss*Byss + wss*Bzss
    ELss = ELs - rLs*(vBLs - vBss)*sB
    ERss = ERs + rRs*(vBRs - vBss)*sB
    ULss = (ρLs, ρLs*SM, ρLs*vss, ρLs*wss, ELss, Byss, Bzss)
    URss = (ρRs, ρRs*SM, ρRs*vss, ρRs*wss, ERss, Byss, Bzss)

    SLs = SM - abs(Bx)/rLs; SRs = SM + abs(Bx)/rRs          # Alfvén speeds
    FLs  = map((f, ul, us) -> f + SL*(us - ul), FL, UL, ULs)
    FRs  = map((f, ur, us) -> f + SR*(us - ur), FR, UR, URs)
    FLss = map((f, us, uss) -> f + SLs*(uss - us), FLs, ULs, ULss)
    FRss = map((f, us, uss) -> f + SRs*(uss - us), FRs, URs, URss)

    z = zero(T)
    F7 = map(FL, FLs, FLss, FRss, FRs, FR) do fl, fls, flss, frss, frs, fr
        _sel5(SL ≥ z, fl, SLs ≥ z, fls, SM ≥ z, flss, SRs ≥ z, frss, SR ≥ z, frs, fr)
    end
    # assemble the 9-flux: GLM gives F[Bx]=ψi, F[ψ]=ch²·Bx.
    return (F7[1], F7[2], F7[3], F7[4], F7[5], ψi, F7[6], F7[7], ch*ch*Bx)
end
