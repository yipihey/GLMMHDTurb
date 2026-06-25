# 3D SIMD CPU backend: symmetric Strang x·y·z·y·x + source, vectorized ALONG x for every sweep.
# SoA flat (column-major, x contiguous): idx(i,j,k) = ((k-1)*nyp + (j-1))*nxp + i. The sweep
# stencil stride is 1 (x), nxp (y), or nxp*nyp (z) — all giving aligned x-block loads. Reuses
# _update_dir + perm with Vec{8} (+ scalar tail). Bit-identical to the scalar 3D backend.

mutable struct Grid3DSoA{N,S<:FVSystem,R,RS}
    sys::S; recon::R; rsol::RS
    U::NTuple{N,Vector{Float32}}; Ut::NTuple{N,Vector{Float32}}
    nx::Int; ny::Int; nz::Int; nxp::Int; nyp::Int; nzp::Int
    dx::Float32; dy::Float32; dz::Float32; bc::Symbol; cfl::Float32
end

function Grid3DSoA(sys::FVSystem, U0::Array{NTuple{N,Float32},3};
                   dx, dy, dz, bc::Symbol = :outflow, recon = PLM(), rsol = HLLC(), cfl = 0.4f0) where {N}
    nx, ny, nz = size(U0); ng = _NG; nxp = nx+2ng; nyp = ny+2ng; nzp = nz+2ng
    mk() = ntuple(_ -> zeros(Float32, nxp*nyp*nzp), Val(N))
    U = mk()
    @inbounds for kk in 1:nz, jj in 1:ny, ii in 1:nx
        b = ((kk-1+ng)*nyp + (jj-1+ng))*nxp + (ii+ng)
        for c in 1:N; U[c][b] = U0[ii,jj,kk][c]; end
    end
    Grid3DSoA{N,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, U, mk(), nx, ny, nz, nxp, nyp, nzp, Float32(dx), Float32(dy), Float32(dz), bc, Float32(cfl))
end

@inline _lin3(nxp, nyp, i, j, k) = ((k-1)*nyp + (j-1))*nxp + i

function _fillghosts3d!(g::Grid3DSoA{N}) where {N}
    ng, nxp, nyp, nzp = _NG, g.nxp, g.nyp, g.nzp; nx, ny, nz = g.nx, g.ny, g.nz; per = g.bc === :periodic
    L(i,j,k) = _lin3(nxp, nyp, i, j, k)
    @inbounds for a in g.U
        for k in 1:nzp, j in 1:nyp, m in 1:ng
            a[L(ng+1-m,j,k)] = per ? a[L(ng+nx+1-m,j,k)] : a[L(ng+1,j,k)]
            a[L(ng+nx+m,j,k)] = per ? a[L(ng+m,j,k)]     : a[L(ng+nx,j,k)]
        end
        for k in 1:nzp, i in 1:nxp, m in 1:ng
            a[L(i,ng+1-m,k)] = per ? a[L(i,ng+ny+1-m,k)] : a[L(i,ng+1,k)]
            a[L(i,ng+ny+m,k)] = per ? a[L(i,ng+m,k)]     : a[L(i,ng+ny,k)]
        end
        for j in 1:nyp, i in 1:nxp, m in 1:ng
            a[L(i,j,ng+1-m)] = per ? a[L(i,j,ng+nz+1-m)] : a[L(i,j,ng+1)]
            a[L(i,j,ng+nz+m)] = per ? a[L(i,j,ng+m)]     : a[L(i,j,ng+nz)]
        end
    end
end

@inline function _sweep_simd3d!(g::Grid3DSoA{N}, dt, d, stride, perm) where {N}
    s, r, rs = g.sys, g.recon, g.rsol; λ = Float32(dt) / d; ng, nxp, nyp = _NG, g.nxp, g.nyp; s2 = 2*stride
    @inbounds for kk in 1:g.nz, jj in 1:g.ny
        base = ((kk-1+ng)*nyp + (jj-1+ng))*nxp; p = ng+1; pend = ng+g.nx
        while p + _W - 1 <= pend
            b = base + p
            v = _update_dir(s, r, rs, loadv(g.U,Val(_W),b-s2), loadv(g.U,Val(_W),b-stride),
                            loadv(g.U,Val(_W),b), loadv(g.U,Val(_W),b+stride), loadv(g.U,Val(_W),b+s2), λ, perm)
            storev!(g.Ut, v, b); p += _W
        end
        while p <= pend
            b = base + p
            v = _update_dir(s, r, rs, loads(g.U,b-s2), loads(g.U,b-stride), loads(g.U,b),
                            loads(g.U,b+stride), loads(g.U,b+s2), λ, perm)
            stores!(g.Ut, v, b); p += 1
        end
    end
    g.U, g.Ut = g.Ut, g.U
end

function step!(g::Grid3DSoA{N}, dt) where {N}
    px = identperm(Val(N)); py = dirperm(g.sys, N, 2); pz = dirperm(g.sys, N, 3); sxy = g.nxp*g.nyp
    _fillghosts3d!(g); _sweep_simd3d!(g, dt/2, g.dx, 1, px)
    _fillghosts3d!(g); _sweep_simd3d!(g, dt/2, g.dy, g.nxp, py)
    _fillghosts3d!(g); _sweep_simd3d!(g, dt,   g.dz, sxy, pz)
    _fillghosts3d!(g); _sweep_simd3d!(g, dt/2, g.dy, g.nxp, py)
    _fillghosts3d!(g); _sweep_simd3d!(g, dt/2, g.dx, 1, px)
    s = g.sys; ng, nxp, nyp = _NG, g.nxp, g.nyp
    @inbounds for kk in 1:g.nz, jj in 1:g.ny, ii in 1:g.nx
        b = ((kk-1+ng)*nyp + (jj-1+ng))*nxp + (ii+ng); stores!(g.U, source(s, loads(g.U, b), Float32(dt)), b)
    end
    return g
end

function max_wavespeed(g::Grid3DSoA{N}) where {N}
    s = g.sys; a = 0f0; ng, nxp, nyp = _NG, g.nxp, g.nyp; py = dirperm(s, N, 2); pz = dirperm(s, N, 3)
    @inbounds for kk in 1:g.nz, jj in 1:g.ny, ii in 1:g.nx
        W = cons2prim(s, loads(g.U, ((kk-1+ng)*nyp + (jj-1+ng))*nxp + (ii+ng)))
        a = max(a, fastspeed_x(s, W), fastspeed_x(s, _swap(W, py)), fastspeed_x(s, _swap(W, pz)))
    end
    return a
end

function evolve_simd3d!(g::Grid3DSoA, tend; maxsteps::Int = 10^7)
    t = 0f0; tend = Float32(tend); n = 0
    while t < tend && n < maxsteps
        c = max_wavespeed(g); g.sys = prestep(g.sys, c)
        dt = min(g.cfl * min(g.dx, g.dy, g.dz) / c, tend - t)
        step!(g, dt); t += dt; n += 1
    end
    return g
end

function primitives_soa(g::Grid3DSoA{N}) where {N}
    ng, nxp, nyp = _NG, g.nxp, g.nyp
    [cons2prim(g.sys, loads(g.U, ((kk-1+ng)*nyp + (jj-1+ng))*nxp + (ii+ng))) for ii in 1:g.nx, jj in 1:g.ny, kk in 1:g.nz]
end
