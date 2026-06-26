# Prototype: CUDA backend for FiniteVolumeGodunovKA, reusing the contract physics VERBATIM
# (cons2prim/faces/riemann/_halfstep) with T = Float32 on a GPU thread. Fused per-cell
# recompute (each thread reads its 5-cell stencil from global, recomputes the half-steps) —
# the register-light structure our .cu work found best for cheap recompute. BC handled
# in-kernel via index mapping (no padding / ghost-fill pass).
#
# Validate vs the CPU backend on Sod, then benchmark. If clean → integrate into the package.

using FiniteVolumeGodunovKA, CUDA
const FV = FiniteVolumeGodunovKA
import FiniteVolumeGodunovKA: cons2prim, prim2cons, physflux_x, maxspeed_x, faces, riemann, _halfstep

@inline gidx(i, nx, ::Val{:periodic}) = mod1(i, nx)
@inline gidx(i, nx, ::Val{:outflow})  = clamp(i, 1, nx)

@inline readcell(U, i, ::Val{N}) where {N} = ntuple(k -> @inbounds(U[i, k]), Val(N))
@inline function writecell!(U, i, v::NTuple{N}) where {N}
    ntuple(k -> (@inbounds(U[i, k] = v[k]); nothing), Val(N)); nothing
end

# Fused per-cell update: half-steps at i-1, i, i+1, two Riemann fluxes, conservative update.
@inline function update_cell(s, r, rs, im2, im1, i0, ip1, ip2, λ)
    WRm     = _halfstep(s, r, im2, im1, i0, λ)[2]   # right face of cell i-1
    WL0, WR0 = _halfstep(s, r, im1, i0, ip1, λ)     # both faces of cell i
    WLp     = _halfstep(s, r, i0, ip1, ip2, λ)[1]   # left face of cell i+1
    Fl = riemann(rs, s, WRm, WL0)                   # interface i-1/2
    Fr = riemann(rs, s, WR0, WLp)                   # interface i+1/2
    return i0 .- λ .* (Fr .- Fl)
end

function _step_kernel!(Unew, U, s, r, rs, λ, nx, ::Val{N}, bc) where {N}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= nx
        im2 = readcell(U, gidx(i-2, nx, bc), Val(N))
        im1 = readcell(U, gidx(i-1, nx, bc), Val(N))
        i0  = readcell(U, gidx(i,   nx, bc), Val(N))
        ip1 = readcell(U, gidx(i+1, nx, bc), Val(N))
        ip2 = readcell(U, gidx(i+2, nx, bc), Val(N))
        writecell!(Unew, i, update_cell(s, r, rs, im2, im1, i0, ip1, ip2, λ))
    end
    return nothing
end

function _speed_kernel!(spd, U, s, nx, ::Val{N}) where {N}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= nx
        @inbounds spd[i] = maxspeed_x(s, cons2prim(s, readcell(U, i, Val(N))))
    end
    return nothing
end

function to_device(U0::Vector{NTuple{N,Float32}}) where {N}
    nx = length(U0); Uh = Matrix{Float32}(undef, nx, N)
    @inbounds for i in 1:nx, k in 1:N; Uh[i, k] = U0[i][k]; end
    CuArray(Uh)
end
from_device(U::CuMatrix{Float32}, ::Val{N}) where {N} =
    (Uh = Array(U); [ntuple(k -> Uh[i, k], Val(N)) for i in 1:size(U, 1)])

function cuda_evolve(s, U0::Vector{NTuple{N,Float32}}, dx, tend;
                     bc = :outflow, recon = PLM(), rsol = HLLC(), cfl = 0.4f0) where {N}
    nx = length(U0); dx = Float32(dx); bcv = Val(bc)
    U = to_device(U0); Unew = similar(U); spd = CUDA.zeros(Float32, nx)
    thr = 256; blk = cld(nx, thr)
    t = 0f0; tend = Float32(tend)
    while t < tend
        @cuda threads=thr blocks=blk _speed_kernel!(spd, U, s, nx, Val(N))
        dt = min(cfl * dx / maximum(spd), tend - t); λ = dt / dx
        @cuda threads=thr blocks=blk _step_kernel!(Unew, U, s, recon, rsol, λ, nx, Val(N), bcv)
        U, Unew = Unew, U
        t += dt
    end
    return from_device(U, Val(N))
end

# Fixed-dt stepping for throughput (no host sync per step beyond the launch).
function cuda_bench(s, U0::Vector{NTuple{N,Float32}}, dx, dt, nsteps;
                    bc = :periodic, recon = PLM(), rsol = HLL()) where {N}
    nx = length(U0); bcv = Val(bc); λ = Float32(dt) / Float32(dx)
    U = to_device(U0); Unew = similar(U)
    thr = 256; blk = cld(nx, thr)
    run1() = (@cuda threads=thr blocks=blk _step_kernel!(Unew, U, s, recon, rsol, λ, nx, Val(N), bcv); (U, Unew) = (Unew, U))
    run1(); CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:nsteps; run1(); end
    CUDA.synchronize()
    nx * nsteps / ((time_ns() - t0) / 1e9) / 1e6
end

# ---- validate vs CPU on Sod ----
println("CUDA functional: ", CUDA.functional(), " | ", CUDA.name(CUDA.device()))
s = Euler(γ = 1.4f0); nx = 4000; dx = 1f0 / nx
xs = Float32[(i - 0.5f0) * dx for i in 1:nx]
ic(x) = prim2cons(s, (x < 0.5f0 ? 1f0 : 0.125f0, 0f0, 0f0, 0f0, x < 0.5f0 ? 1f0 : 0.1f0))
U0 = [ic(x) for x in xs]

gcpu = Grid1D(s, copy(U0); dx = dx, bc = :outflow, recon = PLM(), rsol = HLLC(), cfl = 0.4f0)
FV.evolve!(gcpu, 0.2f0); Wcpu = FV.primitives(gcpu)
Ugpu = cuda_evolve(s, U0, dx, 0.2f0; bc = :outflow, recon = PLM(), rsol = HLLC(), cfl = 0.4f0)
Wgpu = [cons2prim(s, U) for U in Ugpu]    # GPU returns CONSERVED; compare in primitives

maxabs = maximum(maximum(abs.(Wcpu[i] .- Wgpu[i])) for i in 1:nx)
relrho = maximum(abs(Wcpu[i][1] - Wgpu[i][1]) for i in 1:nx)
println("Sod CPU-vs-GPU  max|Δ| (all prim vars) = ", maxabs, "   max|Δρ| = ", relrho)
i = argmin(abs.(xs .- 0.75f0))
println("GPU post-shock ρ=", round(Wgpu[i][1], digits = 4), " (exact 0.2656)  P=", round(Wgpu[i][5], digits = 4), " (exact 0.3031)")

# ---- throughput ----
ρ0(x) = 1f0 + 0.2f0 * sinpi(2f0 * x)
function throughput()
    for nx in (65536, 1048576, 8388608)
        dxs = 1f0 / nx
        U0s = [prim2cons(s, (ρ0((i - 0.5f0) * dxs), 1f0, 0f0, 0f0, 1f0)) for i in 1:nx]
        m = cuda_bench(s, U0s, dxs, 0.2f0 * dxs, 200; bc = :periodic, recon = PLM(), rsol = HLL())
        println("nx=$nx  GPU = $(round(m, digits = 1)) Mcell/s")
    end
end
throughput()
