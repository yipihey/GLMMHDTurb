# Library-owned reconstruction. Operates componentwise on primitive `NTuple`s.
# Branch-free (ifelse/min/max) so it vectorizes under the SIMD/GPU element type.
#
# `PCM`  — piecewise constant (1st order), for debugging / monotone reference.
# `PLM`  — piecewise linear (2nd order). `PLM()` uses the MC limiter (TVD, for
#          shocks); `PLM(:none)` uses unlimited centered slopes (clean 2nd-order
#          convergence on smooth flows, where TVD limiters clip extrema to ~1st).

struct PCM end
struct PLM{Lim} end
PLM()              = PLM{:mc}()
PLM(lim::Symbol)   = PLM{lim}()

# Monotonized-central limiter on a left/right difference pair (branch-free).
@inline function mc_limit(a::T, b::T) where {T}
    s = ifelse(a * b <= zero(T), zero(T),
               sign(a) * min(T(0.5) * abs(a + b), T(2) * abs(a), T(2) * abs(b)))
    return s
end

# Limited slope for one component, dispatched on the reconstruction kind.
@inline slope(::PLM{:mc},   am, a0, ap) = mc_limit(a0 - am, ap - a0)
@inline slope(::PLM{:none}, am, a0, ap) = (ap - am) * oftype(a0, 0.5)

# Face primitive states (WL at the cell's left face, WR at its right face) from the
# three-cell stencil (Wm, W0, Wp). `map` keeps it componentwise over the tuple.
@inline function faces(r::PLM, Wm::NTuple{N,T}, W0::NTuple{N,T}, Wp::NTuple{N,T}) where {N,T}
    d = map((am, a0, ap) -> slope(r, am, a0, ap), Wm, W0, Wp)
    half = oftype(first(W0), 0.5)
    (W0 .- half .* d, W0 .+ half .* d)
end
@inline faces(::PCM, Wm, W0::NTuple{N,T}, Wp) where {N,T} = (W0, W0)
