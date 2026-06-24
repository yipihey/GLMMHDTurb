# MarchBridge — drive the nvcc 2.5D-march godunov kernel (mini-ramses-metal spike_25d.cu,
# compiled by build_march.sh) directly from CUDA.jl, over SHARED device memory.
#
# Why: CUDA.jl emits ~96 regs for the same light HANCOCK1D scheme where nvcc emits ~80, so the
# Julia-native `integrator_hydro_lmarch!` runs ~4-5x slower on the A6000 (occupancy + per-inst
# codegen, not the algorithm — see project_glmmhd_julia_solver memory). This module gives the
# best of both: nvcc's throughput AND Julia's test rig (CUFFT spectra, conservation checks) on
# the very same CuArrays, with zero host round-trip in the timed loop.
#
# State layout: NV=5 fp32 SoA conserved planes [rho, mx, my, mz, E], each an N×N×N CuArray.
# Periodic BCs, fixed dt (set via `set_dtdx`). Build a matching .so first:
#     bash march_bridge/build_march.sh 480
#
# Usage:
#     using CUDA; include("march_bridge/MarchBridge.jl"); using .MarchBridge
#     m  = MarchBridge.open_lib("march_bridge/libmarch480.so")
#     q  = [CUDA.zeros(Float32, m.N, m.N, m.N) for _ in 1:m.NV]   # fill IC...
#     o  = [similar(x) for x in q]
#     MarchBridge.set_dtdx(m, 0.02f0)
#     ms = MarchBridge.run!(m, q, o, 30)        # result left in q; CUFFT/analysis on q here
module MarchBridge

using CUDA, Libdl

struct Lib
    h::Ptr{Cvoid}
    f_dt::Ptr{Cvoid}
    f_run::Ptr{Cvoid}
    f_glm::Ptr{Cvoid}   # march_set_glm; C_NULL for the hydro lib (NV=5)
    NV::Int
    N::Int
end

function open_lib(path::AbstractString)
    h = dlopen(path)
    NV = Int(ccall(dlsym(h, :march_nv), Cint, ()))
    N  = Int(ccall(dlsym(h, :march_nx), Cint, ()))
    fglm = dlsym(h, :march_set_glm; throw_error=false)  # only the GLM-MHD lib exports it
    Lib(h, dlsym(h, :march_set_dtdx), dlsym(h, :march_run_dev),
        fglm === nothing ? C_NULL : fglm, NV, N)
end

close_lib(m::Lib) = dlclose(m.h)

"Set the kernel's fixed dt/dx (a __constant__ in the .cu)."
set_dtdx(m::Lib, v::Real) = ccall(m.f_dt, Cvoid, (Cfloat,), Float32(v))

"GLM-MHD only: set the Dedner cleaning speed `ch` and the psi-decay factor `glmfac`
(applied as `psi *= glmfac` each step). No-op for the hydro lib."
function set_glm(m::Lib, ch::Real, glmfac::Real)
    m.f_glm === C_NULL && error("this lib (NV=$(m.NV)) has no march_set_glm — not a GLM-MHD build")
    ccall(m.f_glm, Cvoid, (Cfloat, Cfloat), Float32(ch), Float32(glmfac))
end

# Build a host vector of device addresses (one per SoA plane) for the float** ABI.
_addrs(planes) = [reinterpret(Ptr{Float32}, UInt(UInt64(pointer(p)))) for p in planes]

"""
    run!(m, q, o, nsteps) -> elapsed_ms

Run `nsteps` periodic marches. `q`,`o` are length-NV vectors of N×N×N fp32 CuArrays
(input/scratch). The result is always left in `q` (a device-to-device copy if `nsteps`
is odd). Returns kernel time in ms (CUDA events; excludes host/device transfer).
"""
function run!(m::Lib, q::AbstractVector{<:CuArray}, o::AbstractVector{<:CuArray}, nsteps::Integer)
    @assert length(q) == m.NV == length(o) "expected $(m.NV) SoA planes"
    @assert all(size(x) == (m.N, m.N, m.N) for x in q) "planes must be $(m.N)^3"
    qa = _addrs(q); oa = _addrs(o)
    GC.@preserve q o qa oa begin
        ccall(m.f_run, Cdouble,
              (Ptr{Ptr{Float32}}, Ptr{Ptr{Float32}}, Cint), qa, oa, Cint(nsteps))
    end
end

end # module
