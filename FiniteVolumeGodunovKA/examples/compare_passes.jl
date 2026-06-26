# Same-process comparison of 3-pass (current step!) vs 5-pass symmetric Strang, to control for
# GPU thermal state. Interleaved + repeated so both see the same clock conditions.
using FiniteVolumeGodunovKA, CUDA
const FV = FiniteVolumeGodunovKA
import FiniteVolumeGodunovKA: prim2cons, identperm, dirperm, _cfg3d,
                              _sweepx3d_kernel!, _sweepy3d_kernel!, _sweepz3d_kernel!, _source3d_kernel!

function step5!(g::FV.Grid3DCU{N}, dt) where {N}                 # symmetric 5-sweep (the old form)
    thr, blk = _cfg3d(g.nx, g.ny, g.nz); bc = Val(g.bc)
    px = identperm(Val(N)); py = dirperm(g.sys, N, 2); pz = dirperm(g.sys, N, 3)
    sw(kern, λ, perm) = (CUDA.@cuda threads=thr blocks=blk kern(g.Unew, g.U, g.sys, g.recon, g.rsol,
        λ, g.nx, g.ny, g.nz, Val(N), bc, perm); (g.U, g.Unew) = (g.Unew, g.U))
    sw(_sweepx3d_kernel!, Float32(dt/2)/g.dx, px); sw(_sweepy3d_kernel!, Float32(dt/2)/g.dy, py)
    sw(_sweepz3d_kernel!, Float32(dt)/g.dz, pz)
    sw(_sweepy3d_kernel!, Float32(dt/2)/g.dy, py); sw(_sweepx3d_kernel!, Float32(dt/2)/g.dx, px)
    CUDA.@cuda threads=thr blocks=blk _source3d_kernel!(g.U, g.sys, Float32(dt), g.nx, g.ny, g.nz, Val(N))
    return g
end

s = Euler(γ=1.4f0); n = 384; d = 1f0/n
ph(i,j,k) = 0.001f0*Float32(mod(i*7+j*13+k*17, 911))
U0 = [prim2cons(s, (1f0+ph(i,j,k), 0.3f0,0.2f0,0.1f0, 1f0+ph(i,j,k))) for i in 1:n, j in 1:n, k in 1:n]
mk() = Grid3DCU(s, U0; dx=d, dy=d, dz=d, bc=:periodic, recon=PLM(), rsol=HLLC())
timeit(g, f, ns) = (f(g, 1f-6); CUDA.synchronize(); t=time_ns(); for _ in 1:ns; f(g, 1f-6); end; CUDA.synchronize(); n^3*ns/((time_ns()-t)/1e9)/1e6)

g3 = mk(); g5 = mk()
for rep in 1:3                                                   # interleave to share thermal state
    m3 = timeit(g3, (g,dt)->FV.step!(g, dt), 15)
    m5 = timeit(g5, step5!, 15)
    println("rep $rep:  3-pass = $(round(m3,digits=0))   5-pass = $(round(m5,digits=0))   ratio 3/5 = $(round(m3/m5,digits=2))")
end
