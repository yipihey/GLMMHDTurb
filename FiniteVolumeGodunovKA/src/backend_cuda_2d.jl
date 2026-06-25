# 2D CUDA backend: one GPU thread per cell, Strang dimensional splitting (x·y·x) + the
# operator-split source. Each sweep kernel reuses `_update_dir` with the direction's perm —
# the SAME code as the CPU 2D backend. Device state is an (nx,ny,N) array, double-buffered.

@inline _readcell2d(U, i, j, ::Val{N}) where {N} = ntuple(k -> @inbounds(U[i, j, k]), Val(N))
@inline function _writecell2d!(U, i, j, v::NTuple{N}) where {N}
    ntuple(k -> (@inbounds(U[i, j, k] = v[k]); nothing), Val(N)); nothing
end

function _sweepx2d_kernel!(Unew, U, s, r, rs, λ, nx, ny, ::Val{N}, bc, perm) where {N}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= nx && j <= ny
        im2 = _readcell2d(U, _gidx(i-2, nx, bc), j, Val(N)); im1 = _readcell2d(U, _gidx(i-1, nx, bc), j, Val(N))
        i0  = _readcell2d(U, i, j, Val(N))
        ip1 = _readcell2d(U, _gidx(i+1, nx, bc), j, Val(N)); ip2 = _readcell2d(U, _gidx(i+2, nx, bc), j, Val(N))
        _writecell2d!(Unew, i, j, _update_dir(s, r, rs, im2, im1, i0, ip1, ip2, λ, perm))
    end
    return nothing
end

function _sweepy2d_kernel!(Unew, U, s, r, rs, λ, nx, ny, ::Val{N}, bc, perm) where {N}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= nx && j <= ny
        jm2 = _readcell2d(U, i, _gidx(j-2, ny, bc), Val(N)); jm1 = _readcell2d(U, i, _gidx(j-1, ny, bc), Val(N))
        j0  = _readcell2d(U, i, j, Val(N))
        jp1 = _readcell2d(U, i, _gidx(j+1, ny, bc), Val(N)); jp2 = _readcell2d(U, i, _gidx(j+2, ny, bc), Val(N))
        _writecell2d!(Unew, i, j, _update_dir(s, r, rs, jm2, jm1, j0, jp1, jp2, λ, perm))
    end
    return nothing
end

function _source2d_kernel!(U, s, dt, nx, ny, ::Val{N}) where {N}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= nx && j <= ny
        _writecell2d!(U, i, j, source(s, _readcell2d(U, i, j, Val(N)), dt))
    end
    return nothing
end

function _speed2d_kernel!(spd, U, s, nx, ny, ::Val{N}, py) where {N}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= nx && j <= ny
        W = cons2prim(s, _readcell2d(U, i, j, Val(N)))
        @inbounds spd[i, j] = max(fastspeed_x(s, W), fastspeed_x(s, _swap(W, py)))
    end
    return nothing
end

mutable struct Grid2DCU{N,S<:FVSystem,R,RS}
    sys::S
    recon::R
    rsol::RS
    U::CuArray{Float32,3}        # (nx, ny, N)
    Unew::CuArray{Float32,3}
    spd::CuMatrix{Float32}
    nx::Int
    ny::Int
    dx::Float32
    dy::Float32
    bc::Symbol
    cfl::Float32
end

function Grid2DCU(sys::FVSystem, U0::Matrix{NTuple{N,T}};
                  dx, dy, bc::Symbol = :outflow, recon = PLM(), rsol = HLLC(),
                  cfl = 0.4f0) where {N,T}
    nx, ny = size(U0)
    Uh = Array{Float32,3}(undef, nx, ny, N)
    @inbounds for j in 1:ny, i in 1:nx, k in 1:N; Uh[i, j, k] = Float32(U0[i, j][k]); end
    U = CuArray(Uh)
    Grid2DCU{N,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, U, similar(U), CUDA.zeros(Float32, nx, ny), nx, ny,
        Float32(dx), Float32(dy), bc, Float32(cfl))
end

@inline _cfg2d(nx, ny) = ((16, 16), (cld(nx, 16), cld(ny, 16)))

# 2 full-dt sweeps + alternating order across steps (rev); source skipped when none. 3 passes → 2.
function step!(g::Grid2DCU{N}, dt; rev::Bool = false) where {N}
    thr, blk = _cfg2d(g.nx, g.ny); bc = Val(g.bc)
    px = identperm(Val(N)); py = dirperm(g.sys, N, 2)
    swx() = (@cuda threads=thr blocks=blk _sweepx2d_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt)/g.dx, g.nx, g.ny, Val(N), bc, px); (g.U, g.Unew) = (g.Unew, g.U))
    swy() = (@cuda threads=thr blocks=blk _sweepy2d_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt)/g.dy, g.nx, g.ny, Val(N), bc, py); (g.U, g.Unew) = (g.Unew, g.U))
    rev ? (swy(); swx()) : (swx(); swy())
    has_source(g.sys) && @cuda threads=thr blocks=blk _source2d_kernel!(g.U, g.sys, Float32(dt), g.nx, g.ny, Val(N))
    return g
end

function max_wavespeed(g::Grid2DCU{N}) where {N}
    thr, blk = _cfg2d(g.nx, g.ny)
    @cuda threads=thr blocks=blk _speed2d_kernel!(g.spd, g.U, g.sys, g.nx, g.ny, Val(N),
                                                  dirperm(g.sys, N, 2))
    return maximum(g.spd)   # max fast SPEED over cells & directions
end

function evolve2d!(g::Grid2DCU, tend; maxsteps::Int = 10^7)
    t = 0f0; tend = Float32(tend); n = 0
    while t < tend && n < maxsteps
        c = max_wavespeed(g)
        g.sys = prestep(g.sys, c)                          # dynamic cleaning speed
        dt = min(g.cfl * min(g.dx, g.dy) / c, tend - t)    # exact for dx=dy
        step!(g, dt; rev = isodd(n))
        t += dt; n += 1
    end
    return g
end

function primitives(g::Grid2DCU{N}) where {N}
    Uh = Array(g.U)
    [cons2prim(g.sys, ntuple(k -> Uh[i, j, k], Val(N))) for i in 1:g.nx, j in 1:g.ny]
end
