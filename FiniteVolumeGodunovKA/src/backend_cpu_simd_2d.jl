# 2D SIMD CPU backend: Strang splitting (x·y·x) + source, vectorized ALONG x for both sweeps.
# SoA flat storage (column-major, x contiguous): the x-sweep uses shifted x-loads; the y-sweep
# uses aligned x-blocks at rows j±1/j±2 (no cross-lane shift). Reuses _update_dir + the perm with
# T = Vec{8,Float32} (+ a scalar tail) — the same physics as the scalar 2D backend. Bit-identical.

mutable struct Grid2DSoA{N,S<:FVSystem,R,RS}
    sys::S
    recon::R
    rsol::RS
    U::NTuple{N,Vector{Float32}}     # flat (nxp*nyp), idx(i,j) = (j-1)*nxp + i
    Ut::NTuple{N,Vector{Float32}}
    nx::Int; ny::Int; nxp::Int; nyp::Int
    dx::Float32; dy::Float32; bc::Symbol; cfl::Float32
end

function Grid2DSoA(sys::FVSystem, U0::Matrix{NTuple{N,Float32}};
                   dx, dy, bc::Symbol = :outflow, recon = PLM(), rsol = HLLC(),
                   cfl = 0.4f0) where {N}
    nx, ny = size(U0); ng = _NG; nxp = nx + 2ng; nyp = ny + 2ng
    mk() = ntuple(_ -> zeros(Float32, nxp*nyp), Val(N))
    U = mk()
    @inbounds for jj in 1:ny, ii in 1:nx
        b = (jj-1+ng)*nxp + (ii+ng)
        for k in 1:N; U[k][b] = U0[ii, jj][k]; end
    end
    Grid2DSoA{N,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, U, mk(), nx, ny, nxp, nyp, Float32(dx), Float32(dy), bc, Float32(cfl))
end

function _fillghosts2d!(g::Grid2DSoA{N}) where {N}
    ng, nxp, nyp, nx, ny = _NG, g.nxp, g.nyp, g.nx, g.ny; per = g.bc === :periodic
    @inbounds for a in g.U
        for j in 1:nyp, m in 1:ng                         # x ghosts (every row)
            rb = (j-1)*nxp
            a[rb + (ng+1-m)]  = per ? a[rb + (ng+nx+1-m)] : a[rb + (ng+1)]
            a[rb + (ng+nx+m)] = per ? a[rb + (ng+m)]      : a[rb + (ng+nx)]
        end
        for m in 1:ng                                     # y ghosts (every column)
            jt = ng+1-m;   st = per ? ng+ny+1-m : ng+1
            jb = ng+ny+m;  sb = per ? ng+m      : ng+ny
            for i in 1:nxp
                a[(jt-1)*nxp + i] = a[(st-1)*nxp + i]
                a[(jb-1)*nxp + i] = a[(sb-1)*nxp + i]
            end
        end
    end
end

# One directional sweep, vectorized along x. `stride` selects the stencil axis: 1 for the
# x-sweep (neighbours ±1 in memory), nxp for the y-sweep (neighbours ±a full row).
@inline function _sweep_simd!(g::Grid2DSoA{N}, dt, d, stride, perm) where {N}
    s, r, rs = g.sys, g.recon, g.rsol; λ = Float32(dt) / d
    ng, nxp = _NG, g.nxp; s2 = 2*stride
    @inbounds for jj in 1:g.ny
        rb = (jj-1+ng)*nxp; p = ng+1; pend = ng+g.nx
        while p + _W - 1 <= pend
            b = rb + p
            v = _update_dir(s, r, rs, loadv(g.U,Val(_W),b-s2), loadv(g.U,Val(_W),b-stride),
                            loadv(g.U,Val(_W),b), loadv(g.U,Val(_W),b+stride),
                            loadv(g.U,Val(_W),b+s2), λ, perm)
            storev!(g.Ut, v, b); p += _W
        end
        while p <= pend
            b = rb + p
            v = _update_dir(s, r, rs, loads(g.U,b-s2), loads(g.U,b-stride), loads(g.U,b),
                            loads(g.U,b+stride), loads(g.U,b+s2), λ, perm)
            stores!(g.Ut, v, b); p += 1
        end
    end
    g.U, g.Ut = g.Ut, g.U
end

function step!(g::Grid2DSoA{N}, dt) where {N}
    px = identperm(Val(N)); py = dirperm(g.sys, N, 2)
    _fillghosts2d!(g); _sweep_simd!(g, dt/2, g.dx, 1, px)
    _fillghosts2d!(g); _sweep_simd!(g, dt,   g.dy, g.nxp, py)
    _fillghosts2d!(g); _sweep_simd!(g, dt/2, g.dx, 1, px)
    s = g.sys; ng, nxp = _NG, g.nxp                       # operator-split source
    @inbounds for jj in 1:g.ny, ii in 1:g.nx
        b = (jj-1+ng)*nxp + (ii+ng); stores!(g.U, source(s, loads(g.U, b), Float32(dt)), b)
    end
    return g
end

function max_wavespeed(g::Grid2DSoA{N}) where {N}
    s = g.sys; a = 0f0; ng, nxp = _NG, g.nxp; py = dirperm(s, N, 2)
    @inbounds for jj in 1:g.ny, ii in 1:g.nx
        W = cons2prim(s, loads(g.U, (jj-1+ng)*nxp + (ii+ng)))
        a = max(a, fastspeed_x(s, W), fastspeed_x(s, _swap(W, py)))
    end
    return a
end

function evolve_simd2d!(g::Grid2DSoA, tend; maxsteps::Int = 10^7)
    t = 0f0; tend = Float32(tend); n = 0
    while t < tend && n < maxsteps
        c = max_wavespeed(g); g.sys = prestep(g.sys, c)
        dt = min(g.cfl * min(g.dx, g.dy) / c, tend - t)
        step!(g, dt); t += dt; n += 1
    end
    return g
end

function primitives_soa(g::Grid2DSoA{N}) where {N}
    ng, nxp = _NG, g.nxp
    [cons2prim(g.sys, loads(g.U, (jj-1+ng)*nxp + (ii+ng))) for ii in 1:g.nx, jj in 1:g.ny]
end
