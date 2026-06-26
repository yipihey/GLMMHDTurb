# ============================================================================================
# Metal (Apple GPU) backend for FiniteVolumeGodunovKA — 1D / 2D / 3D, at parity with the CUDA backends.
#
# ⚠️  UNTESTED — written on a Linux/NVIDIA host where Metal.jl cannot be installed or run.
#     Metal.jl is macOS/Apple-Silicon only. Each kernel BODY here is identical to the corresponding
#     CUDA backend (src/backend_cuda{,_2d,_3d}.jl): the same branch-free `_update_dir` physics compiles
#     for Metal via GPUCompiler in Float32. Only three things differ from CUDA: the launch macro
#     (`Metal.@metal threads= groups=`), the thread-index intrinsics (`thread_position_in_grid_*d`),
#     and the array type (`MtlArray`). The dimensional-split scheme, the alternating-Strang `rev`, the
#     `has_source` guard, and the dynamic-`ch` `prestep` are all mirrored exactly.
#
#     The transpile-to-nvcc backend (`Grid3DCuMarch`) has NO Metal analog — it shells out to `nvcc` to
#     build a CUDA-C `.so`. The Metal equivalent would transpile to MSL and build via `metallib`; that is
#     left as future work (and is the "does Metal.jl have a CUDA-sized codegen gap?" question below).
#
# Validate on a Mac:
#         ] add Metal
#         using FiniteVolumeGodunovKA, Metal
#         include("metal/metal.jl")
#         metal_selfcheck_1d(); metal_selfcheck_2d(); metal_selfcheck_3d()   # all must be max|Δ| = 0
#
# This is intentionally NOT a package dependency (it would break the Linux package's resolution).
# Productionization on the Mac: move it to a weakdep + package extension `ext/MetalExt.jl`.
#
# OPEN DESIGN QUESTION (from DESIGN_fvkernel.md): compare this Metal.jl-native backend to the Metal
# bandwidth roofline (and a hand-MSL kernel if you write one). If Metal.jl is close to native → keep it;
# if it has a CUDA-sized codegen gap → the transpile escape hatch must target MSL too, not just CUDA-C.
# ============================================================================================

using FiniteVolumeGodunovKA, Metal
const FV = FiniteVolumeGodunovKA
import FiniteVolumeGodunovKA: _update_dir, _gidx, _swap, identperm, dirperm, has_source,
                              cons2prim, source, fastspeed_x, maxspeed_x, FVSystem, PLM, HLLC

# Metal thread-index intrinsics return 1-based global grid positions.
@inline _mtid1() = thread_position_in_grid_1d()
@inline _mtid2() = thread_position_in_grid_2d()
@inline _mtid3() = thread_position_in_grid_3d()

# =================================== 1D ===================================
# Mirrors src/backend_cuda.jl: one x-sweep per step (1D has no dimensional splitting), maxspeed CFL.

@inline _mread1(U, i, ::Val{N}) where {N} = ntuple(k -> @inbounds(U[i, k]), Val(N))
@inline _mwrite1!(U, i, v::NTuple{N}) where {N} = (ntuple(k -> (@inbounds(U[i, k] = v[k]); nothing), Val(N)); nothing)

function _mstep1_kernel!(Unew, U, s, r, rs, λ, nx, ::Val{N}, bc, perm) where {N}
    i = _mtid1()
    if i <= nx
        _mwrite1!(Unew, i, _update_dir(s, r, rs,
            _mread1(U,_gidx(i-2,nx,bc),Val(N)), _mread1(U,_gidx(i-1,nx,bc),Val(N)), _mread1(U,i,Val(N)),
            _mread1(U,_gidx(i+1,nx,bc),Val(N)), _mread1(U,_gidx(i+2,nx,bc),Val(N)), λ, perm))
    end
    return
end
function _mspeed1_kernel!(spd, U, s, nx, ::Val{N}) where {N}
    i = _mtid1()
    (i <= nx) && (@inbounds spd[i] = maxspeed_x(s, cons2prim(s, _mread1(U, i, Val(N)))))
    return
end

mutable struct Grid1DMtl{N,S<:FVSystem,R,RS}
    sys::S; recon::R; rsol::RS
    U::MtlArray{Float32,2}; Unew::MtlArray{Float32,2}; spd::MtlArray{Float32,1}
    nx::Int; dx::Float32; bc::Symbol; cfl::Float32
end
function Grid1DMtl(sys::FVSystem, U0::Vector{NTuple{N,T}};
                   dx, bc::Symbol = :outflow, recon = PLM(), rsol = HLLC(), cfl = 0.4f0) where {N,T}
    nx = length(U0); Uh = Matrix{Float32}(undef, nx, N)
    @inbounds for i in 1:nx, k in 1:N; Uh[i, k] = Float32(U0[i][k]); end
    U = MtlArray(Uh)
    Grid1DMtl{N,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, U, similar(U), Metal.zeros(Float32, nx), nx, Float32(dx), bc, Float32(cfl))
end
@inline _mcfg1(nx) = (256, cld(nx, 256))

function mstep1d!(g::Grid1DMtl{N}, dt) where {N}
    thr, grp = _mcfg1(g.nx)
    Metal.@metal threads=thr groups=grp _mstep1_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt)/g.dx, g.nx, Val(N), Val(g.bc), identperm(Val(N)))
    g.U, g.Unew = g.Unew, g.U
    return g
end
function mmax_wavespeed_1d(g::Grid1DMtl{N}) where {N}
    thr, grp = _mcfg1(g.nx)
    Metal.@metal threads=thr groups=grp _mspeed1_kernel!(g.spd, g.U, g.sys, g.nx, Val(N))
    return maximum(g.spd)
end
function mevolve1d!(g::Grid1DMtl, tend; maxsteps::Int = 10^7)
    t = 0f0; tend = Float32(tend); n = 0
    while t < tend && n < maxsteps
        c = mmax_wavespeed_1d(g); g.sys = FV.prestep(g.sys, c)
        dt = min(g.cfl * g.dx / c, tend - t); mstep1d!(g, dt); t += dt; n += 1
    end
    return g
end
mprimitives_1d(g::Grid1DMtl{N}) where {N} =
    (Uh = Array(g.U); [cons2prim(g.sys, ntuple(k -> Uh[i, k], Val(N))) for i in 1:g.nx])

# =================================== 2D ===================================
# Mirrors src/backend_cuda_2d.jl: 2 full-dt sweeps, alternating order (rev), source skipped when none.

@inline _mread2(U, i, j, ::Val{N}) where {N} = ntuple(c -> @inbounds(U[i, j, c]), Val(N))
@inline _mwrite2!(U, i, j, v::NTuple{N}) where {N} = (ntuple(c -> (@inbounds(U[i, j, c] = v[c]); nothing), Val(N)); nothing)

function _msweepx2_kernel!(Unew, U, s, r, rs, λ, nx, ny, ::Val{N}, bc, perm) where {N}
    i, j = _mtid2()
    if i <= nx && j <= ny
        _mwrite2!(Unew, i, j, _update_dir(s, r, rs,
            _mread2(U,_gidx(i-2,nx,bc),j,Val(N)), _mread2(U,_gidx(i-1,nx,bc),j,Val(N)), _mread2(U,i,j,Val(N)),
            _mread2(U,_gidx(i+1,nx,bc),j,Val(N)), _mread2(U,_gidx(i+2,nx,bc),j,Val(N)), λ, perm))
    end
    return
end
function _msweepy2_kernel!(Unew, U, s, r, rs, λ, nx, ny, ::Val{N}, bc, perm) where {N}
    i, j = _mtid2()
    if i <= nx && j <= ny
        _mwrite2!(Unew, i, j, _update_dir(s, r, rs,
            _mread2(U,i,_gidx(j-2,ny,bc),Val(N)), _mread2(U,i,_gidx(j-1,ny,bc),Val(N)), _mread2(U,i,j,Val(N)),
            _mread2(U,i,_gidx(j+1,ny,bc),Val(N)), _mread2(U,i,_gidx(j+2,ny,bc),Val(N)), λ, perm))
    end
    return
end
function _msource2_kernel!(U, s, dt, nx, ny, ::Val{N}) where {N}
    i, j = _mtid2()
    (i <= nx && j <= ny) && _mwrite2!(U, i, j, source(s, _mread2(U,i,j,Val(N)), dt))
    return
end
function _mspeed2_kernel!(spd, U, s, nx, ny, ::Val{N}, py) where {N}
    i, j = _mtid2()
    if i <= nx && j <= ny
        W = cons2prim(s, _mread2(U, i, j, Val(N)))
        @inbounds spd[i, j] = max(fastspeed_x(s, W), fastspeed_x(s, _swap(W, py)))
    end
    return
end

mutable struct Grid2DMtl{N,S<:FVSystem,R,RS}
    sys::S; recon::R; rsol::RS
    U::MtlArray{Float32,3}; Unew::MtlArray{Float32,3}; spd::MtlArray{Float32,2}
    nx::Int; ny::Int; dx::Float32; dy::Float32; bc::Symbol; cfl::Float32
end
function Grid2DMtl(sys::FVSystem, U0::Matrix{NTuple{N,T}};
                   dx, dy, bc::Symbol = :outflow, recon = PLM(), rsol = HLLC(), cfl = 0.4f0) where {N,T}
    nx, ny = size(U0); Uh = Array{Float32,3}(undef, nx, ny, N)
    @inbounds for j in 1:ny, i in 1:nx, c in 1:N; Uh[i,j,c] = Float32(U0[i,j][c]); end
    U = MtlArray(Uh)
    Grid2DMtl{N,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, U, similar(U), Metal.zeros(Float32, nx, ny), nx, ny, Float32(dx), Float32(dy), bc, Float32(cfl))
end
@inline _mcfg2(nx, ny) = ((16, 16), (cld(nx, 16), cld(ny, 16)))

function mstep2d!(g::Grid2DMtl{N}, dt; rev::Bool = false) where {N}
    thr, grp = _mcfg2(g.nx, g.ny); bc = Val(g.bc); px = identperm(Val(N)); py = dirperm(g.sys, N, 2)
    swx() = (Metal.@metal threads=thr groups=grp _msweepx2_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt)/g.dx, g.nx, g.ny, Val(N), bc, px); (g.U, g.Unew) = (g.Unew, g.U))
    swy() = (Metal.@metal threads=thr groups=grp _msweepy2_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt)/g.dy, g.nx, g.ny, Val(N), bc, py); (g.U, g.Unew) = (g.Unew, g.U))
    rev ? (swy(); swx()) : (swx(); swy())
    has_source(g.sys) && Metal.@metal threads=thr groups=grp _msource2_kernel!(g.U, g.sys, Float32(dt), g.nx, g.ny, Val(N))
    return g
end
function mmax_wavespeed_2d(g::Grid2DMtl{N}) where {N}
    thr, grp = _mcfg2(g.nx, g.ny)
    Metal.@metal threads=thr groups=grp _mspeed2_kernel!(g.spd, g.U, g.sys, g.nx, g.ny, Val(N), dirperm(g.sys, N, 2))
    return maximum(g.spd)
end
function mevolve2d!(g::Grid2DMtl, tend; maxsteps::Int = 10^7)
    t = 0f0; tend = Float32(tend); n = 0
    while t < tend && n < maxsteps
        c = mmax_wavespeed_2d(g); g.sys = FV.prestep(g.sys, c)
        dt = min(g.cfl * min(g.dx, g.dy) / c, tend - t); mstep2d!(g, dt; rev = isodd(n)); t += dt; n += 1
    end
    return g
end
mprimitives_2d(g::Grid2DMtl{N}) where {N} =
    (Uh = Array(g.U); [cons2prim(g.sys, ntuple(c -> Uh[i,j,c], Val(N))) for i in 1:g.nx, j in 1:g.ny])

# =================================== 3D ===================================
# Mirrors src/backend_cuda_3d.jl: symmetric Strang x·y·z (rev → z·y·x) + source skipped when none.

@inline _mread3(U, i, j, k, ::Val{N}) where {N} = ntuple(c -> @inbounds(U[i, j, k, c]), Val(N))
@inline _mwrite3!(U, i, j, k, v::NTuple{N}) where {N} = (ntuple(c -> (@inbounds(U[i, j, k, c] = v[c]); nothing), Val(N)); nothing)

function _msweepx3_kernel!(Unew, U, s, r, rs, λ, nx, ny, nz, ::Val{N}, bc, perm) where {N}
    i, j, k = _mtid3()
    if i <= nx && j <= ny && k <= nz
        _mwrite3!(Unew, i, j, k, _update_dir(s, r, rs,
            _mread3(U,_gidx(i-2,nx,bc),j,k,Val(N)), _mread3(U,_gidx(i-1,nx,bc),j,k,Val(N)), _mread3(U,i,j,k,Val(N)),
            _mread3(U,_gidx(i+1,nx,bc),j,k,Val(N)), _mread3(U,_gidx(i+2,nx,bc),j,k,Val(N)), λ, perm))
    end
    return
end
function _msweepy3_kernel!(Unew, U, s, r, rs, λ, nx, ny, nz, ::Val{N}, bc, perm) where {N}
    i, j, k = _mtid3()
    if i <= nx && j <= ny && k <= nz
        _mwrite3!(Unew, i, j, k, _update_dir(s, r, rs,
            _mread3(U,i,_gidx(j-2,ny,bc),k,Val(N)), _mread3(U,i,_gidx(j-1,ny,bc),k,Val(N)), _mread3(U,i,j,k,Val(N)),
            _mread3(U,i,_gidx(j+1,ny,bc),k,Val(N)), _mread3(U,i,_gidx(j+2,ny,bc),k,Val(N)), λ, perm))
    end
    return
end
function _msweepz3_kernel!(Unew, U, s, r, rs, λ, nx, ny, nz, ::Val{N}, bc, perm) where {N}
    i, j, k = _mtid3()
    if i <= nx && j <= ny && k <= nz
        _mwrite3!(Unew, i, j, k, _update_dir(s, r, rs,
            _mread3(U,i,j,_gidx(k-2,nz,bc),Val(N)), _mread3(U,i,j,_gidx(k-1,nz,bc),Val(N)), _mread3(U,i,j,k,Val(N)),
            _mread3(U,i,j,_gidx(k+1,nz,bc),Val(N)), _mread3(U,i,j,_gidx(k+2,nz,bc),Val(N)), λ, perm))
    end
    return
end
function _msource3_kernel!(U, s, dt, nx, ny, nz, ::Val{N}) where {N}
    i, j, k = _mtid3()
    (i <= nx && j <= ny && k <= nz) && _mwrite3!(U, i, j, k, source(s, _mread3(U,i,j,k,Val(N)), dt))
    return
end
function _mspeed3_kernel!(spd, U, s, nx, ny, nz, ::Val{N}, py, pz) where {N}
    i, j, k = _mtid3()
    if i <= nx && j <= ny && k <= nz
        W = cons2prim(s, _mread3(U, i, j, k, Val(N)))
        @inbounds spd[i, j, k] = max(fastspeed_x(s, W), fastspeed_x(s, _swap(W, py)), fastspeed_x(s, _swap(W, pz)))
    end
    return
end

mutable struct Grid3DMtl{N,S<:FVSystem,R,RS}
    sys::S; recon::R; rsol::RS
    U::MtlArray{Float32,4}; Unew::MtlArray{Float32,4}; spd::MtlArray{Float32,3}
    nx::Int; ny::Int; nz::Int; dx::Float32; dy::Float32; dz::Float32; bc::Symbol; cfl::Float32
end
function Grid3DMtl(sys::FVSystem, U0::Array{NTuple{N,T},3};
                   dx, dy, dz, bc::Symbol = :outflow, recon = PLM(), rsol = HLLC(), cfl = 0.4f0) where {N,T}
    nx, ny, nz = size(U0); Uh = Array{Float32,4}(undef, nx, ny, nz, N)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx, c in 1:N; Uh[i,j,k,c] = Float32(U0[i,j,k][c]); end
    U = MtlArray(Uh)
    Grid3DMtl{N,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, U, similar(U), Metal.zeros(Float32, nx, ny, nz),
        nx, ny, nz, Float32(dx), Float32(dy), Float32(dz), bc, Float32(cfl))
end
@inline _mcfg3(nx, ny, nz) = ((8, 8, 4), (cld(nx, 8), cld(ny, 8), cld(nz, 4)))

function mstep3d!(g::Grid3DMtl{N}, dt; rev::Bool = false) where {N}
    thr, grp = _mcfg3(g.nx, g.ny, g.nz); bc = Val(g.bc)
    px = identperm(Val(N)); py = dirperm(g.sys, N, 2); pz = dirperm(g.sys, N, 3)
    sweep(kern, λ, perm) = (Metal.@metal threads=thr groups=grp kern(g.Unew, g.U, g.sys, g.recon, g.rsol,
        λ, g.nx, g.ny, g.nz, Val(N), bc, perm); (g.U, g.Unew) = (g.Unew, g.U))
    if rev
        sweep(_msweepz3_kernel!, Float32(dt)/g.dz, pz); sweep(_msweepy3_kernel!, Float32(dt)/g.dy, py); sweep(_msweepx3_kernel!, Float32(dt)/g.dx, px)
    else
        sweep(_msweepx3_kernel!, Float32(dt)/g.dx, px); sweep(_msweepy3_kernel!, Float32(dt)/g.dy, py); sweep(_msweepz3_kernel!, Float32(dt)/g.dz, pz)
    end
    has_source(g.sys) && Metal.@metal threads=thr groups=grp _msource3_kernel!(g.U, g.sys, Float32(dt), g.nx, g.ny, g.nz, Val(N))
    return g
end
function mmax_wavespeed_3d(g::Grid3DMtl{N}) where {N}
    thr, grp = _mcfg3(g.nx, g.ny, g.nz)
    Metal.@metal threads=thr groups=grp _mspeed3_kernel!(g.spd, g.U, g.sys, g.nx, g.ny, g.nz, Val(N),
                                                         dirperm(g.sys, N, 2), dirperm(g.sys, N, 3))
    return maximum(g.spd)
end
function mevolve3d!(g::Grid3DMtl, tend; maxsteps::Int = 10^7)
    t = 0f0; tend = Float32(tend); n = 0
    while t < tend && n < maxsteps
        c = mmax_wavespeed_3d(g); g.sys = FV.prestep(g.sys, c)
        dt = min(g.cfl * min(g.dx, g.dy, g.dz) / c, tend - t); mstep3d!(g, dt; rev = isodd(n)); t += dt; n += 1
    end
    return g
end
mprimitives_3d(g::Grid3DMtl{N}) where {N} =
    (Uh = Array(g.U); [cons2prim(g.sys, ntuple(c -> Uh[i,j,k,c], Val(N))) for i in 1:g.nx, j in 1:g.ny, k in 1:g.nz])

# =============================== validation (run on a Mac) ===============================
# Each Metal backend must be bit-identical to its scalar reference (same branch-free physics, Float32).

function metal_selfcheck_1d()
    s = FV.Euler(γ = 1.4f0); n = 256; d = 1f0/n
    U0 = [FV.prim2cons(s, (1f0+0.3f0*sinpi(2f0*(i-1f0)/n), 0.5f0,0f0,0f0, 1f0)) for i in 1:n]
    gsc = FV.Grid1D(s, copy(U0); dx=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    gm  = Grid1DMtl(s, copy(U0); dx=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    for _ in 1:20; FV.step!(gsc, 0.1f0*d); mstep1d!(gm, 0.1f0*d); end
    Wc = FV.primitives(gsc); Wm = mprimitives_1d(gm)
    md = maximum(maximum(abs.(Wc[i] .- Wm[i])) for i in 1:n)
    println("Metal ≡ scalar Grid1D max|Δ| = ", md, md == 0f0 ? "  ✓ bit-identical" : "  (investigate)"); md
end

function metal_selfcheck_2d()
    s = FV.Euler(γ = 1.4f0); n = 64; d = 1f0/n
    U0 = [FV.prim2cons(s, (1f0+0.3f0*sinpi(2f0*(i+j-1f0)/n), 0.5f0,0.3f0,0f0, 1f0)) for i in 1:n, j in 1:n]
    gsc = FV.Grid2D(s, copy(U0); dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    gm  = Grid2DMtl(s, copy(U0); dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    for nn in 0:14; FV.step!(gsc, 0.1f0*d; rev=isodd(nn)); mstep2d!(gm, 0.1f0*d; rev=isodd(nn)); end
    Wc = FV.primitives(gsc); Wm = mprimitives_2d(gm)
    md = maximum(maximum(abs.(Wc[i,j] .- Wm[i,j])) for i in 1:n, j in 1:n)
    println("Metal ≡ scalar Grid2D max|Δ| = ", md, md == 0f0 ? "  ✓ bit-identical" : "  (investigate)"); md
end

function metal_selfcheck_3d()
    s = FV.Euler(γ = 1.4f0); n = 32; d = 1f0/n
    U0 = [FV.prim2cons(s, (1f0+0.3f0*sinpi(2f0*(i+j+k-1f0)/n), 0.5f0,0.3f0,0.2f0, 1f0)) for i in 1:n, j in 1:n, k in 1:n]
    gsc = FV.Grid3D(s, copy(U0); dx=d, dy=d, dz=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    gm  = Grid3DMtl(s, copy(U0); dx=d, dy=d, dz=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    for nn in 0:9; FV.step!(gsc, 0.1f0*d; rev=isodd(nn)); mstep3d!(gm, 0.1f0*d; rev=isodd(nn)); end
    Wc = FV.primitives(gsc); Wm = mprimitives_3d(gm)
    md = maximum(maximum(abs.(Wc[i,j,k] .- Wm[i,j,k])) for i in 1:n, j in 1:n, k in 1:n)
    println("Metal ≡ scalar Grid3D max|Δ| = ", md, md == 0f0 ? "  ✓ bit-identical" : "  (investigate)"); md
end
