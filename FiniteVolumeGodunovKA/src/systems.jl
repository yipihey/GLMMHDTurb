# The compressible Euler system, defined entirely through the @fvsystem contract.
# This is the worked example proving the contract expresses real physics; GLM-MHD is
# the same shape with B[3] + ψ added to the tuples and a ψ-damping source.
#
# Primitive  W = (ρ, u, v, w, P)
# Conserved  U = (ρ, ρu, ρv, ρw, E)     E = P/(γ-1) + ½ρ|v|²
# Every function is branch-free and generic over the element type T.

@fvsystem Euler begin
    nvars = 5
    vidx  = ((2, 3, 4),)                   # one rotating vector: momentum (x,y,z)
    @params γ = 5f0/3f0

    cons2prim(U, p) = begin
        ρ, mx, my, mz, E = U
        iρ = inv(ρ)
        u, v, w = mx * iρ, my * iρ, mz * iρ
        P = (p.γ - 1) * (E - 0.5f0 * ρ * (u*u + v*v + w*w))
        (ρ, u, v, w, P)
    end

    prim2cons(W, p) = begin
        ρ, u, v, w, P = W
        E = P / (p.γ - 1) + 0.5f0 * ρ * (u*u + v*v + w*w)
        (ρ, ρ*u, ρ*v, ρ*w, E)
    end

    physflux_x(W, p) = begin
        ρ, u, v, w, P = W
        E = P / (p.γ - 1) + 0.5f0 * ρ * (u*u + v*v + w*w)
        (ρ*u, ρ*u*u + P, ρ*u*v, ρ*u*w, u * (E + P))
    end

    maxspeed_x(W, p) = begin
        ρ, u, v, w, P = W
        abs(u) + sqrt(p.γ * P / ρ)
    end

    eig_x(W, p) = begin
        ρ, u, v, w, P = W
        (u, sqrt(p.γ * P / ρ))
    end
end

# GLM-MHD (Dedner et al. 2002) — the same contract, two rotating vectors. This is the test
# that "the user writes only physflux_x, the library rotates for y/z" survives a real MHD flux.
#
# Primitive  W = (ρ, u, v, w, P, Bx, By, Bz, ψ)
# Conserved  U = (ρ, ρu, ρv, ρw, E, Bx, By, Bz, ψ)   E = P/(γ-1) + ½ρ|v|² + ½|B|²
# GLM coupling: F[Bx] = ψ, F[ψ] = ch²·Bx (the divergence-cleaning wave at speed ch).
# v0 is the hyperbolic GLM with LLF; the parabolic ψ-damping source and HLLD are refinements.
@fvsystem GLMMHD begin
    nvars = 9
    vidx  = ((2, 3, 4), (6, 7, 8))         # TWO rotating vectors: momentum AND magnetic field
    @params γ  = 5f0/3f0
    @params ch = 1f0                        # cleaning speed (set to the max fast speed by the driver)

    cons2prim(U, p) = begin
        ρ, mx, my, mz, E, Bx, By, Bz, ψ = U
        iρ = inv(ρ); u, v, w = mx*iρ, my*iρ, mz*iρ
        B2 = Bx*Bx + By*By + Bz*Bz
        P = (p.γ - 1) * (E - 0.5f0*ρ*(u*u + v*v + w*w) - 0.5f0*B2)
        (ρ, u, v, w, P, Bx, By, Bz, ψ)
    end

    prim2cons(W, p) = begin
        ρ, u, v, w, P, Bx, By, Bz, ψ = W
        E = P/(p.γ - 1) + 0.5f0*ρ*(u*u + v*v + w*w) + 0.5f0*(Bx*Bx + By*By + Bz*Bz)
        (ρ, ρ*u, ρ*v, ρ*w, E, Bx, By, Bz, ψ)
    end

    physflux_x(W, p) = begin
        ρ, u, v, w, P, Bx, By, Bz, ψ = W
        B2 = Bx*Bx + By*By + Bz*Bz; vB = u*Bx + v*By + w*Bz
        ptot = P + 0.5f0*B2
        E = P/(p.γ - 1) + 0.5f0*ρ*(u*u + v*v + w*w) + 0.5f0*B2
        ch2 = p.ch * p.ch
        (ρ*u,
         ρ*u*u + ptot - Bx*Bx,
         ρ*u*v - Bx*By,
         ρ*u*w - Bx*Bz,
         (E + ptot)*u - Bx*vB,
         ψ,                                 # F[Bx] = ψ  (GLM)
         By*u - Bx*v,                        # F[By]
         Bz*u - Bx*w,                        # F[Bz]
         ch2*Bx)                             # F[ψ]  = ch²·Bx
    end

    maxspeed_x(W, p) = begin
        ρ, u, v, w, P, Bx, By, Bz, ψ = W
        a2 = p.γ*P/ρ; b2 = (Bx*Bx + By*By + Bz*Bz)/ρ; bx2 = Bx*Bx/ρ
        cf = sqrt(0.5f0*(a2 + b2 + sqrt(max(0f0, (a2 + b2)*(a2 + b2) - 4f0*a2*bx2))))
        max(abs(u) + cf, p.ch)              # fast magnetosonic, floored by the cleaning speed
    end
end

export Euler, GLMMHD
