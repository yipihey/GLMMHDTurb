# 3D CPU backend (scalar reference): symmetric Strang splitting x·y·z·y·x + source.
# Each sweep reuses _update_dir with the direction's perm — the z-sweep via dirperm(s,N,3),
# which swaps slot[1]↔slot[3] in every vector triple (momentum AND B for GLM-MHD). This is the
# proof the rotation machinery generalizes to all three axes with no new code.

mutable struct Grid3D{N,T,S<:FVSystem,R,RS}
    sys::S
    recon::R
    rsol::RS
    U::Array{NTuple{N,T},3}      # (nx, ny, nz)
    Ut::Array{NTuple{N,T},3}
    nx::Int; ny::Int; nz::Int
    dx::T; dy::T; dz::T
    bc::Symbol; cfl::T
end

function Grid3D(sys::FVSystem, U0::Array{NTuple{N,T},3};
               dx, dy, dz, bc::Symbol = :outflow, recon = PLM(), rsol = HLLC(),
               cfl = T(0.4)) where {N,T}
    nx, ny, nz = size(U0)
    Grid3D{N,T,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, copy(U0), similar(U0), nx, ny, nz, T(dx), T(dy), T(dz), bc, T(cfl))
end

function _sweep_x3d!(g::Grid3D{N,T}, dt) where {N,T}
    s, r, rs = g.sys, g.recon, g.rsol; λ = T(dt)/g.dx
    nx, ny, nz = g.nx, g.ny, g.nz; bc = Val(g.bc); perm = identperm(Val(N)); U = g.U
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        g.Ut[i,j,k] = _update_dir(s, r, rs, U[_gidx(i-2,nx,bc),j,k], U[_gidx(i-1,nx,bc),j,k],
                                  U[i,j,k], U[_gidx(i+1,nx,bc),j,k], U[_gidx(i+2,nx,bc),j,k], λ, perm)
    end
    g.U, g.Ut = g.Ut, g.U
end

function _sweep_y3d!(g::Grid3D{N,T}, dt) where {N,T}
    s, r, rs = g.sys, g.recon, g.rsol; λ = T(dt)/g.dy
    nx, ny, nz = g.nx, g.ny, g.nz; bc = Val(g.bc); perm = dirperm(s, N, 2); U = g.U
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        g.Ut[i,j,k] = _update_dir(s, r, rs, U[i,_gidx(j-2,ny,bc),k], U[i,_gidx(j-1,ny,bc),k],
                                  U[i,j,k], U[i,_gidx(j+1,ny,bc),k], U[i,_gidx(j+2,ny,bc),k], λ, perm)
    end
    g.U, g.Ut = g.Ut, g.U
end

function _sweep_z3d!(g::Grid3D{N,T}, dt) where {N,T}
    s, r, rs = g.sys, g.recon, g.rsol; λ = T(dt)/g.dz
    nx, ny, nz = g.nx, g.ny, g.nz; bc = Val(g.bc); perm = dirperm(s, N, 3); U = g.U
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        g.Ut[i,j,k] = _update_dir(s, r, rs, U[i,j,_gidx(k-2,nz,bc)], U[i,j,_gidx(k-1,nz,bc)],
                                  U[i,j,k], U[i,j,_gidx(k+1,nz,bc)], U[i,j,_gidx(k+2,nz,bc)], λ, perm)
    end
    g.U, g.Ut = g.Ut, g.U
end

# Symmetric Strang (2nd order): x(dt/2)·y(dt/2)·z(dt)·y(dt/2)·x(dt/2), then the source.
function step!(g::Grid3D, dt)
    _sweep_x3d!(g, dt/2); _sweep_y3d!(g, dt/2); _sweep_z3d!(g, dt); _sweep_y3d!(g, dt/2); _sweep_x3d!(g, dt/2)
    s = g.sys
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        g.U[i,j,k] = source(s, g.U[i,j,k], dt)
    end
    return g
end

function max_wavespeed(g::Grid3D{N}) where {N}
    s = g.sys; a = zero(g.dx); py = dirperm(s, N, 2); pz = dirperm(s, N, 3)
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        W = cons2prim(s, g.U[i,j,k])
        a = max(a, fastspeed_x(s, W), fastspeed_x(s, _swap(W, py)), fastspeed_x(s, _swap(W, pz)))
    end
    return a
end

function evolve3d!(g::Grid3D, tend; maxsteps::Int = 10^7)
    t = 0f0; tend = Float32(tend); n = 0
    while t < tend && n < maxsteps
        c = max_wavespeed(g); g.sys = prestep(g.sys, c)
        dt = min(g.cfl * min(g.dx, g.dy, g.dz) / c, tend - t)
        step!(g, dt); t += dt; n += 1
    end
    return g
end

primitives(g::Grid3D{N}) where {N} = [cons2prim(g.sys, g.U[i,j,k]) for i in 1:g.nx, j in 1:g.ny, k in 1:g.nz]
