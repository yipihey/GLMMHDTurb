# Dimensional-splitting machinery — the "library rotates for y/z" design win.
#
# The user writes ONLY physflux_x (and eig_x/maxspeed_x, all x-direction). To get the y/z
# fluxes, we rotate a cell's state into the sweep direction's normal frame by SWAPPING the
# marked vector components (vidx), call the same x-direction physics, then swap the resulting
# flux back. For Euler vidx=(2,3,4): the y-sweep swaps components 2↔3 (u↔v / ρu↔ρv), so
# physflux_x of the swapped state IS the y-flux (in swapped order); swapping back recovers it.
# Swap is its own inverse, so the same operation rotates state in and flux out.

# Boundary index map (shared by all backends).
@inline _gidx(i, n, ::Val{:periodic}) = mod1(i, n)
@inline _gidx(i, n, ::Val{:outflow})  = clamp(i, 1, n)

# Swap two tuple components; Val(0),Val(0) is the identity (x-direction, no rotation).
@inline _swap(t::NTuple{N,T}, ::Val{0}, ::Val{0}) where {N,T} = t
@inline _swap(t::NTuple{N,T}, ::Val{a}, ::Val{b}) where {N,T,a,b} =
    ntuple(k -> k == a ? t[b] : (k == b ? t[a] : t[k]), Val(N))

# The swap (as a pair of Vals) that rotates a sweep direction into the normal frame.
@inline xperm(::FVSystem) = (Val(0), Val(0))
@inline yperm(s::FVSystem) = (Val(vidx(s)[1]), Val(vidx(s)[2]))
@inline zperm(s::FVSystem) = (Val(vidx(s)[1]), Val(vidx(s)[3]))

# Normal-frame flux balance (Fr - Fl) for a 5-cell stencil; fused per-cell recompute
# (the register-light structure). Inputs are already in the normal frame.
@inline function _fluxdiff(s, r, rs, im2, im1, i0, ip1, ip2, λ)
    WRm      = _halfstep(s, r, im2, im1, i0, λ)[2]   # right face of cell i-1
    WL0, WR0 = _halfstep(s, r, im1, i0, ip1, λ)      # both faces of cell i
    WLp      = _halfstep(s, r, i0, ip1, ip2, λ)[1]   # left face of cell i+1
    return riemann(rs, s, WR0, WLp) .- riemann(rs, s, WRm, WL0)
end

# One cell's directional update: rotate the lab-frame stencil into the normal frame, compute
# the flux balance with the x-direction physics, rotate the balance back, apply. (va,vb) is
# the direction's swap — Val(0),Val(0) for x reduces this to the plain 1D update.
@inline function _update_dir(s, r, rs, im2, im1, i0, ip1, ip2, λ, va::Val, vb::Val)
    fd = _fluxdiff(s, r, rs, _swap(im2, va, vb), _swap(im1, va, vb), _swap(i0, va, vb),
                   _swap(ip1, va, vb), _swap(ip2, va, vb), λ)
    return i0 .- λ .* _swap(fd, va, vb)
end
