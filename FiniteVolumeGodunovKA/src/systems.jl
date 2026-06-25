# The compressible Euler system, defined entirely through the @fvsystem contract.
# This is the worked example proving the contract expresses real physics; GLM-MHD is
# the same shape with B[3] + ψ added to the tuples and a ψ-damping source.
#
# Primitive  W = (ρ, u, v, w, P)
# Conserved  U = (ρ, ρu, ρv, ρw, E)     E = P/(γ-1) + ½ρ|v|²
# Every function is branch-free and generic over the element type T.

@fvsystem Euler begin
    nvars = 5
    vidx  = (2, 3, 4)                      # momentum components rotate for y/z sweeps
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

export Euler
