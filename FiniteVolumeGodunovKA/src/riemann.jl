# Library-owned Riemann solvers. Each takes the system + left/right **primitive**
# face states and returns the conserved interface flux (an `NTuple{N,T}`).
# All branch-free (ifelse on scalar/Vec masks) → SIMD/GPU ready.
#
#   LLF  — Rusanov. Needs only `maxspeed_x`; works for ANY system.
#   HLL  — two-wave. Needs `eig_x` (u, c); works for any system that provides it.
#   HLLC — Euler contact-restoring. Built-in, assumes the Euler primitive layout
#          (ρ, u, v, w, P) / conserved (ρ, ρu, ρv, ρw, E).

struct LLF  end
struct HLL  end
struct HLLC end

@inline function riemann(::LLF, s::FVSystem, WL::NTuple{N,T}, WR::NTuple{N,T}) where {N,T}
    FL = physflux_x(s, WL); FR = physflux_x(s, WR)
    UL = prim2cons(s, WL);  UR = prim2cons(s, WR)
    a  = max(maxspeed_x(s, WL), maxspeed_x(s, WR))
    return map((fl, fr, ul, ur) -> 0.5f0 * (fl + fr) - 0.5f0 * a * (ur - ul), FL, FR, UL, UR)
end

@inline function riemann(::HLL, s::FVSystem, WL::NTuple{N,T}, WR::NTuple{N,T}) where {N,T}
    uL, cL = eig_x(s, WL);  uR, cR = eig_x(s, WR)
    SL = min(uL - cL, uR - cR);  SR = max(uL + cL, uR + cR)
    FL = physflux_x(s, WL); FR = physflux_x(s, WR)
    UL = prim2cons(s, WL);  UR = prim2cons(s, WR)
    invd = inv(SR - SL);  z = zero(SL)
    return map(FL, FR, UL, UR) do fl, fr, ul, ur
        fhll = (SR * fl - SL * fr + SL * SR * (ur - ul)) * invd
        ifelse(SL >= z, fl, ifelse(SR <= z, fr, fhll))
    end
end

@inline function riemann(::HLLC, s::FVSystem, WL::NTuple{N,T}, WR::NTuple{N,T}) where {N,T}
    ρL, uL, vL, wL, pL = WL[1], WL[2], WL[3], WL[4], WL[5]
    ρR, uR, vR, wR, pR = WR[1], WR[2], WR[3], WR[4], WR[5]
    _, cL = eig_x(s, WL);  _, cR = eig_x(s, WR)
    SL = min(uL - cL, uR - cR);  SR = max(uL + cL, uR + cR)

    mL = ρL * (SL - uL);  mR = ρR * (SR - uR)              # ρ(S-u)
    Sstar = (pR - pL + mL * uL - mR * uR) / (mL - mR)

    FL = physflux_x(s, WL); FR = physflux_x(s, WR)
    UL = prim2cons(s, WL);  UR = prim2cons(s, WR)
    EL = UL[5];  ER = UR[5]

    qL = mL / (SL - Sstar)                                  # ρ(S-u)/(S-S*)
    eL = EL / ρL + (Sstar - uL) * (Sstar + pL / mL)
    UsL = (qL, qL * Sstar, qL * vL, qL * wL, qL * eL)
    qR = mR / (SR - Sstar)
    eR = ER / ρR + (Sstar - uR) * (Sstar + pR / mR)
    UsR = (qR, qR * Sstar, qR * vR, qR * wR, qR * eR)

    z = zero(SL)
    return map(FL, FR, UL, UR, UsL, UsR) do fl, fr, ul, ur, usl, usr
        fsl = fl + SL * (usl - ul)
        fsr = fr + SR * (usr - ur)
        ifelse(SL >= z, fl, ifelse(Sstar >= z, fsl, ifelse(SR >= z, fsr, fr)))
    end
end
