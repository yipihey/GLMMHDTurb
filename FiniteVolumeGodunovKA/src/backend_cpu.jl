# Reference CPU backend (v0): scalar, 1D, single-stage MUSCL-Hancock.
#
# This is the correctness anchor for the contract — deliberately simple (allocates
# per step, no SIMD/threads yet). The SIMD-CPU backend reuses the IDENTICAL physics
# with the element type `T = Vec{W,Float32}` over lane-packed cells; the CUDA backend
# reuses it with `T = Float32` in a staged shared-memory cube. Only this driver loop
# changes per backend; `cons2prim`/`faces`/`riemann` are shared verbatim.

const _NG = 2  # ghost layers (PLM slope stencil + interface reach)

mutable struct Grid1D{N,T,S<:FVSystem,R,RS}
    sys::S
    recon::R
    rsol::RS
    U::Vector{NTuple{N,T}}
    nx::Int
    dx::T
    bc::Symbol      # :periodic | :outflow
    cfl::T
end

function Grid1D(sys::FVSystem, U0::Vector{NTuple{N,T}};
                dx, bc::Symbol=:outflow, recon=PLM(), rsol=HLLC(), cfl=T(0.4)) where {N,T}
    Grid1D{N,T,typeof(sys),typeof(recon),typeof(rsol)}(
        sys, recon, rsol, copy(U0), length(U0), T(dx), bc, T(cfl))
end

primitives(g::Grid1D) = [cons2prim(g.sys, u) for u in g.U]

function conserved_total(g::Grid1D{N,T}) where {N,T}
    acc = g.U[1]
    @inbounds for i in 2:g.nx
        acc = acc .+ g.U[i]
    end
    return acc .* g.dx
end

function max_wavespeed(g::Grid1D)
    s = g.sys
    a = fastspeed_x(s, cons2prim(s, g.U[1]))
    @inbounds for i in 2:g.nx
        a = max(a, fastspeed_x(s, cons2prim(s, g.U[i])))
    end
    return a
end

# Fill a padded primitive array (length nx + 2*_NG) with ghosts per the BC.
function _padded_primitives(g::Grid1D{N,T}) where {N,T}
    nx, s = g.nx, g.sys
    Wp = Vector{NTuple{N,T}}(undef, nx + 2_NG)
    @inbounds for i in 1:nx
        Wp[i + _NG] = cons2prim(s, g.U[i])
    end
    if g.bc === :periodic
        @inbounds for k in 1:_NG
            Wp[k]              = Wp[nx + k]          # left ghosts ← right interior
            Wp[nx + _NG + k]   = Wp[_NG + k]         # right ghosts ← left interior
        end
    elseif g.bc === :outflow
        @inbounds for k in 1:_NG
            Wp[k]              = Wp[_NG + 1]         # zero-gradient
            Wp[nx + _NG + k]   = Wp[nx + _NG]
        end
    else
        error("Grid1D: unknown bc $(g.bc)")
    end
    return Wp
end

function step!(g::Grid1D{N,T}, dt) where {N,T}
    s, dx, nx, ng = g.sys, g.dx, g.nx, _NG
    λ = T(dt) / dx
    Wp = _padded_primitives(g)

    # MUSCL-Hancock predictor: half-step the limited face states by dt/2.
    np  = nx + 2ng
    WLh = Vector{NTuple{N,T}}(undef, np)
    WRh = Vector{NTuple{N,T}}(undef, np)
    @inbounds for j in 2:np-1
        WL, WR = faces(g.recon, Wp[j-1], Wp[j], Wp[j+1])
        FL, FR = physflux_x(s, WL), physflux_x(s, WR)
        dUh = (T(0.5) * λ) .* (FR .- FL)
        WLh[j] = cons2prim(s, prim2cons(s, WL) .- dUh)
        WRh[j] = cons2prim(s, prim2cons(s, WR) .- dUh)
    end

    # Godunov corrector: Riemann flux at each interface, conservative update.
    Unew = Vector{NTuple{N,T}}(undef, nx)
    @inbounds for i in 1:nx
        j  = i + ng
        Fl = riemann(g.rsol, s, WRh[j-1], WLh[j])    # interface i-1/2
        Fr = riemann(g.rsol, s, WRh[j],   WLh[j+1])  # interface i+1/2
        Unew[i] = g.U[i] .- λ .* (Fr .- Fl)
    end
    if has_source(s)
        @inbounds for i in 1:nx
            Unew[i] = source(s, Unew[i], dt)    # operator-split source
        end
    end
    copyto!(g.U, Unew)
    return g
end

"""
    evolve!(g, tend; maxsteps=10^7) -> g

Advance to `tend` with CFL-limited steps. Returns the grid (state in `g.U`).
"""
function evolve!(g::Grid1D{N,T}, tend; maxsteps::Int = 10^7) where {N,T}
    t = zero(T); tend = T(tend); n = 0
    while t < tend && n < maxsteps
        c = max_wavespeed(g)
        g.sys = prestep(g.sys, c)          # dynamic cleaning speed (no-op unless GLM)
        dt = min(g.cfl * g.dx / c, tend - t)
        step!(g, dt)
        t += dt; n += 1
    end
    return g
end
