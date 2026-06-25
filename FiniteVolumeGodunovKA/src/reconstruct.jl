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
# Value-based zero (not `T(...)`) so it vectorizes for T = Vec as well as scalars.
@inline function mc_limit(a, b)
    z = zero(a)
    ifelse(a * b <= z, z, sign(a) * min(0.5f0 * abs(a + b), 2f0 * abs(a), 2f0 * abs(b)))
end

# Limited slope for one component, dispatched on the reconstruction kind.
@inline slope(::PLM{:mc},   am, a0, ap) = mc_limit(a0 - am, ap - a0)
@inline slope(::PLM{:none}, am, a0, ap) = (ap - am) * 0.5f0

# Face primitive states (WL at the cell's left face, WR at its right face) from the
# three-cell stencil (Wm, W0, Wp). `map` keeps it componentwise over the tuple.
@inline function faces(r::PLM, Wm::NTuple{N,T}, W0::NTuple{N,T}, Wp::NTuple{N,T}) where {N,T}
    d = map((am, a0, ap) -> slope(r, am, a0, ap), Wm, W0, Wp)
    (W0 .- 0.5f0 .* d, W0 .+ 0.5f0 .* d)
end
@inline faces(::PCM, Wm, W0::NTuple{N,T}, Wp) where {N,T} = (W0, W0)
