"""
    FiniteVolumeGodunovKA

A write-once, multi-backend finite-volume Godunov DSL. The user declares a *system*
(conserved variables + a handful of pure, branch-free per-cell physics functions over a
generic element type `T`); the library owns all the structure — reconstruction, the
Riemann combination, the conservative update, tiling, occupancy tuning, and backend
dispatch (CPU-SIMD / CUDA / Metal).

This is the v0 scaffold: the `@fvsystem` contract, the library-owned PLM reconstruction and
LLF/HLL/HLLC Riemann solvers, a reference **CPU (scalar) backend** for 1D, and the `Euler`
system defined through the contract. The element-generic physics already vectorizes (T =
Float32 scalar on a thread, `Vec{W}` on a CPU lane); the SIMD-CPU and CUDA backends drop in
against the *same* system definition.

See `DESIGN_fvkernel.md` for the contract rationale and the three locked design decisions.
"""
module FiniteVolumeGodunovKA

using SIMD
using CUDA

# SIMD.jl deliberately leaves `Base.ifelse` on Vec masks to downstream packages, so
# we provide it here. This is what makes the branch-free physics (ifelse/min/max)
# element-type-generic: the IDENTICAL code runs as Float32 scalars (one thread) or
# `Vec{W,Float32}` lanes (one CPU core).
@inline Base.ifelse(m::Vec{N,Bool}, a::Vec{N,T}, b::Vec{N,T}) where {N,T} = vifelse(m, a, b)

export @fvsystem, FVSystem
export PLM, PCM, LLF, HLL, HLLC, HLLD
export Grid1D, step!, evolve!, primitives, conserved_total
export Grid1DSoA, evolve_simd!, primitives_soa
export Grid1DCU, evolve_cuda!, primitives_cuda
export Grid2D, evolve2d!, Grid2DCU

# ---------------------------------------------------------------------------
# The contract. A system is a `<: FVSystem` value; the per-cell physics are
# methods on these generic functions, supplied by the user via `@fvsystem`.
# All operate on `NTuple{N,T}` states and MUST be branch-free on field values
# (use `ifelse`/`min`/`max`, never `if x > 0`) so the identical code runs as
# Float32 scalars (GPU thread) or `Vec{W,Float32}` lanes (CPU SIMD).
# ---------------------------------------------------------------------------

"""Abstract supertype for all finite-volume systems (Euler, GLM-MHD, …)."""
abstract type FVSystem end

"Number of conserved variables (the tuple width)."
function nconserved end
"Indices of the rotating vector components in the conserved/primitive tuple (e.g. momentum `(2,3,4)`)."
function vidx end
"`U -> W`: conserved → primitive (both `NTuple{N,T}`)."
function cons2prim end
"`W -> U`: primitive → conserved."
function prim2cons end
"`W -> F`: physical flux in the **x** direction (y/z obtained by component rotation)."
function physflux_x end
"`W -> |u_x| + c`: max signal speed in x (CFL + LLF). Required."
function maxspeed_x end
"`W -> (u_x, c)`: normal velocity and signal speed in x (HLL/HLLC). Optional."
function eig_x end
"`(U, dt) -> U`: optional operator-split source (e.g. GLM ψ-damping). Default: identity."
function source end
@inline source(::FVSystem, U, dt) = U
"`W -> |u_x| + c`: physical fast signal speed, WITHOUT any cleaning floor — used to set a global
cleaning speed. Default = `maxspeed_x` (correct for systems with no cleaning wave)."
function fastspeed_x end
@inline fastspeed_x(s::FVSystem, W) = maxspeed_x(s, W)
"`(s, cmax) -> s`: optional per-step system update given the global max fast speed (e.g. GLM sets
ch = cmax each step). Default: identity."
function prestep end
@inline prestep(s::FVSystem, cmax) = s

include("macro.jl")
include("reconstruct.jl")
include("riemann.jl")
include("dimsplit.jl")
include("backend_cpu.jl")
include("backend_cpu_simd.jl")
include("backend_cuda.jl")
include("backend_cpu_2d.jl")
include("backend_cuda_2d.jl")
include("systems.jl")
include("riemann_mhd.jl")

end # module
