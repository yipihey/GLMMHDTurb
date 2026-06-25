# ============================================================================================
# Metal (Apple GPU) 2D backend for FiniteVolumeGodunovKA.
#
# ⚠️  UNTESTED — written on a Linux/NVIDIA host where Metal.jl cannot be installed or run.
#     Metal.jl is macOS/Apple-Silicon only. This file is a faithful transcription of the CUDA
#     2D backend (src/backend_cuda_2d.jl): the kernel BODIES are identical (the same branch-free
#     `_update_dir` physics compiles for Metal via GPUCompiler, in Float32) — only the launch
#     macro, the thread-index intrinsics, and the array type differ. Validate on a Mac:
#
#         ] add Metal
#         using FiniteVolumeGodunovKA, Metal
#         include("metal/metal_2d.jl")
#         # then run the bit-identical check against Grid2D (scalar) below.
#
# This is intentionally NOT a package dependency (it would break the Linux package's resolution).
# Productionization on the Mac: move it to a weakdep + package extension `ext/MetalExt.jl`.
#
# ALSO RUN THE METAL-GAP EXPERIMENT (the open design question from DESIGN_fvkernel.md): compare
# this Metal.jl-native backend to the Metal bandwidth roofline (and, if you write one, a hand-MSL
# kernel). If Metal.jl is close to native → keep it; if it has a CUDA-sized codegen gap → the
# transpile escape hatch must target MSL too, not just CUDA-C.
# ============================================================================================

using FiniteVolumeGodunovKA, Metal
const FV = FiniteVolumeGodunovKA
import FiniteVolumeGodunovKA: _update_dir, _gidx, _swap, identperm, dirperm,
                              cons2prim, source, fastspeed_x, FVSystem, PLM, HLLC

@inline _mreadcell(U, i, j, ::Val{N}) where {N} = ntuple(c -> @inbounds(U[i, j, c]), Val(N))
@inline function _mwritecell!(U, i, j, v::NTuple{N}) where {N}
    ntuple(c -> (@inbounds(U[i, j, c] = v[c]); nothing), Val(N)); nothing
end
@inline _mtid() = thread_position_in_grid_2d()   # Metal: (x, y) 1-based grid position

function _msweepx_kernel!(Unew, U, s, r, rs, λ, nx, ny, ::Val{N}, bc, perm) where {N}
    i, j = _mtid()
    if i <= nx && j <= ny
        _mwritecell!(Unew, i, j, _update_dir(s, r, rs,
            _mreadcell(U,_gidx(i-2,nx,bc),j,Val(N)), _mreadcell(U,_gidx(i-1,nx,bc),j,Val(N)),
            _mreadcell(U,i,j,Val(N)),
            _mreadcell(U,_gidx(i+1,nx,bc),j,Val(N)), _mreadcell(U,_gidx(i+2,nx,bc),j,Val(N)), λ, perm))
    end
    return
end
function _msweepy_kernel!(Unew, U, s, r, rs, λ, nx, ny, ::Val{N}, bc, perm) where {N}
    i, j = _mtid()
    if i <= nx && j <= ny
        _mwritecell!(Unew, i, j, _update_dir(s, r, rs,
            _mreadcell(U,i,_gidx(j-2,ny,bc),Val(N)), _mreadcell(U,i,_gidx(j-1,ny,bc),Val(N)),
            _mreadcell(U,i,j,Val(N)),
            _mreadcell(U,i,_gidx(j+1,ny,bc),Val(N)), _mreadcell(U,i,_gidx(j+2,ny,bc),Val(N)), λ, perm))
    end
    return
end
function _msource_kernel!(U, s, dt, nx, ny, ::Val{N}) where {N}
    i, j = _mtid()
    (i <= nx && j <= ny) && _mwritecell!(U, i, j, source(s, _mreadcell(U,i,j,Val(N)), dt))
    return
end
function _mspeed_kernel!(spd, U, s, nx, ny, ::Val{N}, py) where {N}
    i, j = _mtid()
    if i <= nx && j <= ny
        W = cons2prim(s, _mreadcell(U, i, j, Val(N)))
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

@inline _mcfg(nx, ny) = ((16, 16), (cld(nx, 16), cld(ny, 16)))

function mstep!(g::Grid2DMtl{N}, dt) where {N}
    thr, grp = _mcfg(g.nx, g.ny); bc = Val(g.bc); px = identperm(Val(N)); py = dirperm(g.sys, N, 2)
    Metal.@metal threads=thr groups=grp _msweepx_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt/2)/g.dx, g.nx, g.ny, Val(N), bc, px); g.U, g.Unew = g.Unew, g.U
    Metal.@metal threads=thr groups=grp _msweepy_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt)/g.dy, g.nx, g.ny, Val(N), bc, py);   g.U, g.Unew = g.Unew, g.U
    Metal.@metal threads=thr groups=grp _msweepx_kernel!(g.Unew, g.U, g.sys, g.recon, g.rsol,
        Float32(dt/2)/g.dx, g.nx, g.ny, Val(N), bc, px); g.U, g.Unew = g.Unew, g.U
    Metal.@metal threads=thr groups=grp _msource_kernel!(g.U, g.sys, Float32(dt), g.nx, g.ny, Val(N))
    return g
end

function mmax_wavespeed(g::Grid2DMtl{N}) where {N}
    thr, grp = _mcfg(g.nx, g.ny)
    Metal.@metal threads=thr groups=grp _mspeed_kernel!(g.spd, g.U, g.sys, g.nx, g.ny, Val(N), dirperm(g.sys, N, 2))
    return maximum(g.spd)
end

function mevolve2d!(g::Grid2DMtl, tend; maxsteps::Int = 10^7)
    t = 0f0; tend = Float32(tend); n = 0
    while t < tend && n < maxsteps
        c = mmax_wavespeed(g); g.sys = FV.prestep(g.sys, c)
        dt = min(g.cfl * min(g.dx, g.dy) / c, tend - t); mstep!(g, dt); t += dt; n += 1
    end
    return g
end

mprimitives(g::Grid2DMtl{N}) where {N} = (Uh = Array(g.U); [cons2prim(g.sys, ntuple(c -> Uh[i,j,c], Val(N))) for i in 1:g.nx, j in 1:g.ny])

# --- validation to run on the Mac: must be bit-identical to the scalar Grid2D ---
function metal_selfcheck()
    s = FV.Euler(γ = 1.4f0); n = 64; d = 1f0/n
    U0 = [FV.prim2cons(s, (1f0+0.3f0*sinpi(2f0*(i+j-1f0)/n), 0.5f0,0.3f0,0f0, 1f0)) for i in 1:n, j in 1:n]
    gsc = FV.Grid2D(s, copy(U0); dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    gm  = Grid2DMtl(s, copy(U0); dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    for _ in 1:15; FV.step!(gsc, 0.1f0*d); mstep!(gm, 0.1f0*d); end
    Wc = FV.primitives(gsc); Wm = mprimitives(gm)
    md = maximum(maximum(abs.(Wc[i,j] .- Wm[i,j])) for i in 1:n, j in 1:n)
    println("Metal ≡ scalar Grid2D max|Δ| = ", md, md == 0f0 ? "  ✓ bit-identical" : "  (investigate)")
    md
end
