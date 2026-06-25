# CUDA backend (v0): one GPU thread per cell, fused per-cell recompute.
#
# Reuses the contract physics VERBATIM (cons2prim/faces/riemann/_halfstep) with T = Float32
# on a thread — the same code the scalar and SIMD CPU backends run. Each thread reads its
# 5-cell stencil from global memory and recomputes the MUSCL-Hancock half-steps (the
# register-light fused structure our .cu work found best for cheap recompute). Boundaries are
# handled in-kernel by index mapping (periodic = mod1, outflow = clamp) — no padding or
# ghost-fill pass. State is a device (nx, N) matrix, double-buffered.
#
# Packaging note: CUDA is a direct dependency for v0; moving it to a weakdep + package
# extension (so CPU-only installs don't pull CUDA artifacts) is a follow-up.

@inline _gidx(i, nx, ::Val{:periodic}) = mod1(i, nx)
@inline _gidx(i, nx, ::Val{:outflow})  = clamp(i, 1, nx)

@inline _readcell(U, i, ::Val{N}) where {N} = ntuple(k -> @inbounds(U[i, k]), Val(N))
@inline function _writecell!(U, i, v::NTuple{N}) where {N}
    ntuple(k -> (@inbounds(U[i, k] = v[k]); nothing), Val(N)); nothing
end

# Fused per-cell update: half-steps at i-1, i, i+1, two Riemann fluxes, conservative update.
@inline function _update_cell(s, r, rs, im2, im1, i0, ip1, ip2, λ)
    WRm      = _halfstep(s, r, im2, im1, i0, λ)[2]   # right face of cell i-1
    WL0, WR0 = _halfstep(s, r, im1, i0, ip1, λ)      # both faces of cell i
    WLp      = _halfstep(s, r, i0, ip1, ip2, λ)[1]   # left face of cell i+1
    Fl = riemann(rs, s, WRm, WL0)                    # interface i-1/2
    Fr = riemann(rs, s, WR0, WLp)                    # interface i+1/2
    return i0 .- λ .* (Fr .- Fl)
end

function _cuda_step_kernel!(Unew, U, s, r, rs, λ, nx, ::Val{N}, bc) where {N}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= nx
        im2 = _readcell(U, _gidx(i-2, nx, bc), Val(N))
        im1 = _readcell(U, _gidx(i-1, nx, bc), Val(N))
        i0  = _readcell(U, _gidx(i,   nx, bc), Val(N))
        ip1 = _readcell(U, _gidx(i+1, nx, bc), Val(N))
        ip2 = _readcell(U, _gidx(i+2, nx, bc), Val(N))
        _writecell!(Unew, i, _update_cell(s, r, rs, im2, im1, i0, ip1, ip2, λ))
    end
    return nothing
end

function _cuda_speed_kernel!(spd, U, s, nx, ::Val{N}) where {N}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= nx
        @inbounds spd[i] = maxspeed_x(s, cons2prim(s, _readcell(U, i, Val(N))))
    end
    return nothing
end

mutable struct Grid1DCU{N,S<:FVSystem,R,RS}
    sys::S
    recon::R
    rsol::RS
    U::CuMatrix{Float32}        # (nx, N) conserved
    Unew::CuMatrix{Float32}     # double buffer
    spd::CuVector{Float32}      # per-cell wavespeed scratch
    nx::Int
    dx::Float32
    bc::Symbol
    cfl::Float32
end

function Grid1DCU(sys::FVSystem, U0::Vector{NTuple{N,Float32}};
                  dx, bc::Symbol = :outflow, recon = PLM(), rsol = HLLC(),
                  cfl = 0.4f0) where {N}
    nx = length(U0); Uh = Matrix{Float32}(undef, nx, N)
    @inbounds for i in 1:nx, k in 1:N; Uh[i, k] = U0[i][k]; end
    U = CuArray(Uh)
    Grid1DCU{N,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, U, similar(U), CUDA.zeros(Float32, nx), nx, Float32(dx), bc, Float32(cfl))
end

@inline _launch(nx) = (256, cld(nx, 256))

function step!(g::Grid1DCU{N}, dt) where {N}
    thr, blk = _launch(g.nx)
    @cuda threads=thr blocks=blk _cuda_step_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
                                                    Float32(dt) / g.dx, g.nx, Val(N), Val(g.bc))
    g.U, g.Unew = g.Unew, g.U
    return g
end

function max_wavespeed(g::Grid1DCU{N}) where {N}
    thr, blk = _launch(g.nx)
    @cuda threads=thr blocks=blk _cuda_speed_kernel!(g.spd, g.U, g.sys, g.nx, Val(N))
    return maximum(g.spd)
end

function evolve_cuda!(g::Grid1DCU, tend; maxsteps::Int = 10^7)
    t = 0f0; tend = Float32(tend); n = 0
    while t < tend && n < maxsteps
        dt = min(g.cfl * g.dx / max_wavespeed(g), tend - t)
        step!(g, dt)
        t += dt; n += 1
    end
    return g
end

primitives_cuda(g::Grid1DCU{N}) where {N} =
    (Uh = Array(g.U); [cons2prim(g.sys, ntuple(k -> Uh[i, k], Val(N))) for i in 1:g.nx])
