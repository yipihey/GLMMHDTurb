# SIMD CPU backend (v0): single-thread, 1D, vectorized along the grid.
#
# The performance counterpart to the scalar reference backend. State is
# Structure-of-Arrays (one padded Float32 vector per conserved variable); the stencil
# neighbours come from *shifted* vector loads (vload at p-1 / p / p+1). The per-cell
# physics — cons2prim / faces / riemann — is reused VERBATIM with the element type
# T = Vec{W,Float32}: W consecutive cells per lane. A scalar tail (T = Float32) handles
# the remainder, again the same functions. This is the cpu_simd.jl lane pattern.
#
# Threads + cache-blocking are the next increment; this establishes correctness and the
# single-core vectorization win against the scalar backend.

const _W = 8  # lane width (Vec{8,Float32} = 256-bit; bump to 16 on AVX-512 hosts)

struct Grid1DSoA{N,S<:FVSystem,R,RS}
    sys::S
    recon::R
    rsol::RS
    U::NTuple{N,Vector{Float32}}     # conserved, padded length nx + 2*_NG
    WL::NTuple{N,Vector{Float32}}    # scratch: half-stepped left-face primitives
    WR::NTuple{N,Vector{Float32}}    # scratch: half-stepped right-face primitives
    nx::Int
    dx::Float32
    bc::Symbol
    cfl::Float32
end

function Grid1DSoA(sys::FVSystem, U0::Vector{NTuple{N,Float32}};
                   dx, bc::Symbol = :outflow, recon = PLM(), rsol = HLLC(),
                   cfl = 0.4f0) where {N}
    nx = length(U0); npad = nx + 2_NG
    mk() = ntuple(_ -> Vector{Float32}(undef, npad), Val(N))
    U = mk()
    @inbounds for k in 1:N, i in 1:nx
        U[k][i + _NG] = U0[i][k]
    end
    Grid1DSoA{N,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, U, mk(), mk(), nx, Float32(dx), bc, Float32(cfl))
end

# --- SoA tuple load/store helpers (vector and scalar) ---
@inline loadv(a::NTuple{N,Vector{Float32}}, ::Val{Wd}, p) where {N,Wd} =
    ntuple(k -> vload(Vec{Wd,Float32}, a[k], p), Val(N))
@inline loads(a::NTuple{N,Vector{Float32}}, p) where {N} =
    ntuple(k -> @inbounds(a[k][p]), Val(N))
@inline function storev!(a::NTuple{N,Vector{Float32}}, v::NTuple{N,Vec{Wd,Float32}}, p) where {N,Wd}
    ntuple(k -> (vstore(v[k], a[k], p); nothing), Val(N)); nothing
end
@inline function stores!(a::NTuple{N,Vector{Float32}}, v::NTuple{N,Float32}, p) where {N}
    ntuple(k -> (@inbounds(a[k][p] = v[k]); nothing), Val(N)); nothing
end

function fillghosts!(g::Grid1DSoA{N}) where {N}
    ng, nx = _NG, g.nx
    @inbounds for a in g.U
        if g.bc === :periodic
            for j in 1:ng
                a[j]            = a[nx + j]
                a[nx + ng + j]  = a[ng + j]
            end
        elseif g.bc === :outflow
            for j in 1:ng
                a[j]            = a[ng + 1]
                a[nx + ng + j]  = a[nx + ng]
            end
        else
            error("Grid1DSoA: unknown bc $(g.bc)")
        end
    end
end

function step!(g::Grid1DSoA{N}, dt) where {N}
    s, r, rs = g.sys, g.recon, g.rsol
    λ    = Float32(dt) / g.dx
    npad = length(g.U[1])
    fillghosts!(g)

    # Pass 1 — half-stepped faces for every cell with both neighbours (2 … npad-1).
    p = 2
    @inbounds while p + _W - 1 <= npad - 1
        WLh, WRh = _halfstep(s, r, loadv(g.U, Val(_W), p-1), loadv(g.U, Val(_W), p),
                             loadv(g.U, Val(_W), p+1), λ)
        storev!(g.WL, WLh, p); storev!(g.WR, WRh, p)
        p += _W
    end
    @inbounds while p <= npad - 1
        WLh, WRh = _halfstep(s, r, loads(g.U, p-1), loads(g.U, p), loads(g.U, p+1), λ)
        stores!(g.WL, WLh, p); stores!(g.WR, WRh, p)
        p += 1
    end

    # Pass 2 — Riemann flux per interface, conservative update in place (each cell reads
    # only its own U + the face scratch, so in-place is safe).
    p = _NG + 1; pend = _NG + g.nx
    @inbounds while p + _W - 1 <= pend
        Fl = riemann(rs, s, loadv(g.WR, Val(_W), p-1), loadv(g.WL, Val(_W), p))
        Fr = riemann(rs, s, loadv(g.WR, Val(_W), p),   loadv(g.WL, Val(_W), p+1))
        U0 = loadv(g.U, Val(_W), p)
        storev!(g.U, U0 .- λ .* (Fr .- Fl), p)
        p += _W
    end
    @inbounds while p <= pend
        Fl = riemann(rs, s, loads(g.WR, p-1), loads(g.WL, p))
        Fr = riemann(rs, s, loads(g.WR, p),   loads(g.WL, p+1))
        stores!(g.U, loads(g.U, p) .- λ .* (Fr .- Fl), p)
        p += 1
    end
    return g
end

function max_wavespeed(g::Grid1DSoA)
    s = g.sys; a = 0f0
    @inbounds for i in 1:g.nx
        a = max(a, maxspeed_x(s, cons2prim(s, loads(g.U, i + _NG))))
    end
    return a
end

function evolve_simd!(g::Grid1DSoA, tend; maxsteps::Int = 10^7)
    t = 0f0; tend = Float32(tend); n = 0
    while t < tend && n < maxsteps
        dt = min(g.cfl * g.dx / max_wavespeed(g), tend - t)
        step!(g, dt)
        t += dt; n += 1
    end
    return g
end

primitives_soa(g::Grid1DSoA{N}) where {N} =
    [cons2prim(g.sys, loads(g.U, i + _NG)) for i in 1:g.nx]

function conserved_total(g::Grid1DSoA{N}) where {N}
    acc = ntuple(k -> 0.0, Val(N))
    @inbounds for i in 1:g.nx
        u = loads(g.U, i + _NG)
        acc = acc .+ ntuple(k -> Float64(u[k]), Val(N))
    end
    return acc .* Float64(g.dx)
end
