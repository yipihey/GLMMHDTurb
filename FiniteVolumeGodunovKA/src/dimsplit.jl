# Dimensional-splitting machinery — the "library rotates for y/z" design win.
#
# The user writes ONLY physflux_x. To get the y/z fluxes we rotate a cell's state into the
# sweep direction's normal frame by SWAPPING the marked vector components, call the same
# x-direction physics, then swap the flux back. `vidx(s)` is a tuple of 3-component vector
# triples (Euler: ((2,3,4),) for momentum; GLM-MHD: ((2,3,4),(6,7,8)) for momentum AND B).
# The y-sweep swaps slot[1]↔slot[2] within EVERY triple — swapping two vectors is an even
# permutation (a proper 90° rotation), which is exactly why no pseudovector sign flip is needed.

# Boundary index map (shared by all backends).
@inline _gidx(i, n, ::Val{:periodic}) = mod1(i, n)
@inline _gidx(i, n, ::Val{:outflow})  = clamp(i, 1, n)

# Reindex a state by a compile-time permutation P (an NTuple{N,Int}). The direction swap is
# an involution, so the SAME perm rotates state in and flux out.
@inline _swap(t::NTuple{N,T}, ::Val{P}) where {N,T,P} = ntuple(k -> @inbounds(t[P[k]]), Val(N))

# Identity permutation (x-direction, no rotation).
@inline identperm(::Val{N}) where {N} = Val(ntuple(identity, Val(N)))

# Direction-d permutation: swap slot[1]↔slot[d] within each vector triple of vidx(s). Built on
# the host (once per sweep / passed as an isbits Val kernel arg), so its instability is irrelevant
# to the hot loop, where `_swap` unrolls on the compile-time perm.
function dirperm(s::FVSystem, N::Integer, d::Integer)
    p = collect(1:Int(N))
    @inbounds for tr in vidx(s)
        a, b = tr[1], tr[d]
        p[a], p[b] = p[b], p[a]
    end
    return Val(Tuple(p))
end

# Normal-frame flux balance (Fr - Fl) for a 5-cell stencil; fused per-cell recompute (the
# register-light structure). Inputs are already in the normal frame.
@inline function _fluxdiff(s, r, rs, im2, im1, i0, ip1, ip2, λ)
    WRm      = _halfstep(s, r, im2, im1, i0, λ)[2]   # right face of cell i-1
    WL0, WR0 = _halfstep(s, r, im1, i0, ip1, λ)      # both faces of cell i
    WLp      = _halfstep(s, r, i0, ip1, ip2, λ)[1]   # left face of cell i+1
    return riemann(rs, s, WR0, WLp) .- riemann(rs, s, WRm, WL0)
end

# One cell's directional update: rotate the lab-frame stencil into the normal frame (perm),
# compute the flux balance with the x-direction physics, rotate it back, apply. `perm` is the
# identity Val for x → reduces to the plain 1D update.
@inline function _update_dir(s, r, rs, im2, im1, i0, ip1, ip2, λ, perm::Val)
    fd = _fluxdiff(s, r, rs, _swap(im2, perm), _swap(im1, perm), _swap(i0, perm),
                   _swap(ip1, perm), _swap(ip2, perm), λ)
    return i0 .- λ .* _swap(fd, perm)
end
