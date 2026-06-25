# 3D CUDA backend: one thread per cell, symmetric Strang x·y·z·y·x + source. Each sweep reuses
# _update_dir with the direction's perm — the SAME code as the scalar 3D backend. Device state is
# an (nx,ny,nz,N) array, double-buffered.

@inline _readcell3d(U, i, j, k, ::Val{N}) where {N} = ntuple(c -> @inbounds(U[i, j, k, c]), Val(N))
@inline function _writecell3d!(U, i, j, k, v::NTuple{N}) where {N}
    ntuple(c -> (@inbounds(U[i, j, k, c] = v[c]); nothing), Val(N)); nothing
end
@inline _tid3d() = ((blockIdx().x-1)*blockDim().x + threadIdx().x,
                    (blockIdx().y-1)*blockDim().y + threadIdx().y,
                    (blockIdx().z-1)*blockDim().z + threadIdx().z)

function _sweepx3d_kernel!(Unew, U, s, r, rs, λ, nx, ny, nz, ::Val{N}, bc, perm) where {N}
    i, j, k = _tid3d()
    if i <= nx && j <= ny && k <= nz
        _writecell3d!(Unew, i, j, k, _update_dir(s, r, rs,
            _readcell3d(U,_gidx(i-2,nx,bc),j,k,Val(N)), _readcell3d(U,_gidx(i-1,nx,bc),j,k,Val(N)),
            _readcell3d(U,i,j,k,Val(N)),
            _readcell3d(U,_gidx(i+1,nx,bc),j,k,Val(N)), _readcell3d(U,_gidx(i+2,nx,bc),j,k,Val(N)), λ, perm))
    end
    return nothing
end
function _sweepy3d_kernel!(Unew, U, s, r, rs, λ, nx, ny, nz, ::Val{N}, bc, perm) where {N}
    i, j, k = _tid3d()
    if i <= nx && j <= ny && k <= nz
        _writecell3d!(Unew, i, j, k, _update_dir(s, r, rs,
            _readcell3d(U,i,_gidx(j-2,ny,bc),k,Val(N)), _readcell3d(U,i,_gidx(j-1,ny,bc),k,Val(N)),
            _readcell3d(U,i,j,k,Val(N)),
            _readcell3d(U,i,_gidx(j+1,ny,bc),k,Val(N)), _readcell3d(U,i,_gidx(j+2,ny,bc),k,Val(N)), λ, perm))
    end
    return nothing
end
function _sweepz3d_kernel!(Unew, U, s, r, rs, λ, nx, ny, nz, ::Val{N}, bc, perm) where {N}
    i, j, k = _tid3d()
    if i <= nx && j <= ny && k <= nz
        _writecell3d!(Unew, i, j, k, _update_dir(s, r, rs,
            _readcell3d(U,i,j,_gidx(k-2,nz,bc),Val(N)), _readcell3d(U,i,j,_gidx(k-1,nz,bc),Val(N)),
            _readcell3d(U,i,j,k,Val(N)),
            _readcell3d(U,i,j,_gidx(k+1,nz,bc),Val(N)), _readcell3d(U,i,j,_gidx(k+2,nz,bc),Val(N)), λ, perm))
    end
    return nothing
end
function _source3d_kernel!(U, s, dt, nx, ny, nz, ::Val{N}) where {N}
    i, j, k = _tid3d()
    (i <= nx && j <= ny && k <= nz) && _writecell3d!(U, i, j, k, source(s, _readcell3d(U,i,j,k,Val(N)), dt))
    return nothing
end
function _speed3d_kernel!(spd, U, s, nx, ny, nz, ::Val{N}, py, pz) where {N}
    i, j, k = _tid3d()
    if i <= nx && j <= ny && k <= nz
        W = cons2prim(s, _readcell3d(U, i, j, k, Val(N)))
        @inbounds spd[i, j, k] = max(fastspeed_x(s, W), fastspeed_x(s, _swap(W, py)), fastspeed_x(s, _swap(W, pz)))
    end
    return nothing
end

mutable struct Grid3DCU{N,S<:FVSystem,R,RS}
    sys::S; recon::R; rsol::RS
    U::CuArray{Float32,4}; Unew::CuArray{Float32,4}; spd::CuArray{Float32,3}
    nx::Int; ny::Int; nz::Int; dx::Float32; dy::Float32; dz::Float32; bc::Symbol; cfl::Float32
end

function Grid3DCU(sys::FVSystem, U0::Array{NTuple{N,T},3};
                  dx, dy, dz, bc::Symbol = :outflow, recon = PLM(), rsol = HLLC(), cfl = 0.4f0) where {N,T}
    nx, ny, nz = size(U0)
    Uh = Array{Float32,4}(undef, nx, ny, nz, N)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx, c in 1:N; Uh[i,j,k,c] = Float32(U0[i,j,k][c]); end
    U = CuArray(Uh)
    Grid3DCU{N,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, U, similar(U), CUDA.zeros(Float32, nx, ny, nz),
        nx, ny, nz, Float32(dx), Float32(dy), Float32(dz), bc, Float32(cfl))
end

@inline _cfg3d(nx, ny, nz) = ((8,8,4), (cld(nx,8), cld(ny,8), cld(nz,4)))

function step!(g::Grid3DCU{N}, dt; rev::Bool = false) where {N}
    thr, blk = _cfg3d(g.nx, g.ny, g.nz); bc = Val(g.bc)
    px = identperm(Val(N)); py = dirperm(g.sys, N, 2); pz = dirperm(g.sys, N, 3)
    sweep(kern, λ, perm) = (@cuda threads=thr blocks=blk kern(g.Unew, g.U, g.sys, g.recon, g.rsol,
        λ, g.nx, g.ny, g.nz, Val(N), bc, perm); (g.U, g.Unew) = (g.Unew, g.U))
    if rev
        sweep(_sweepz3d_kernel!, Float32(dt)/g.dz, pz); sweep(_sweepy3d_kernel!, Float32(dt)/g.dy, py); sweep(_sweepx3d_kernel!, Float32(dt)/g.dx, px)
    else
        sweep(_sweepx3d_kernel!, Float32(dt)/g.dx, px); sweep(_sweepy3d_kernel!, Float32(dt)/g.dy, py); sweep(_sweepz3d_kernel!, Float32(dt)/g.dz, pz)
    end
    @cuda threads=thr blocks=blk _source3d_kernel!(g.U, g.sys, Float32(dt), g.nx, g.ny, g.nz, Val(N))
    return g
end

function max_wavespeed(g::Grid3DCU{N}) where {N}
    thr, blk = _cfg3d(g.nx, g.ny, g.nz)
    @cuda threads=thr blocks=blk _speed3d_kernel!(g.spd, g.U, g.sys, g.nx, g.ny, g.nz, Val(N),
                                                  dirperm(g.sys, N, 2), dirperm(g.sys, N, 3))
    return maximum(g.spd)
end

function evolve3d!(g::Grid3DCU, tend; maxsteps::Int = 10^7)
    t = 0f0; tend = Float32(tend); n = 0
    while t < tend && n < maxsteps
        c = max_wavespeed(g); g.sys = prestep(g.sys, c)
        dt = min(g.cfl * min(g.dx, g.dy, g.dz) / c, tend - t)
        step!(g, dt; rev = isodd(n)); t += dt; n += 1
    end
    return g
end

function primitives(g::Grid3DCU{N}) where {N}
    Uh = Array(g.U)
    [cons2prim(g.sys, ntuple(c -> Uh[i,j,k,c], Val(N))) for i in 1:g.nx, j in 1:g.ny, k in 1:g.nz]
end
