# Threaded SIMD throughput at large grids (run with -t <n>).
using FiniteVolumeGodunovKA
const FV = FiniteVolumeGodunovKA
import FiniteVolumeGodunovKA: prim2cons

function bench(n, nsteps)
    s = Euler(γ=1.4f0); d = 1f0/n
    U0 = [prim2cons(s, (1f0+0.2f0*sinpi(2f0*Float32(i+j)/n), 0.5f0, 0.3f0, 0f0, 1f0)) for i in 1:n, j in 1:n]
    g = Grid2DSoA(s, U0; dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    FV.step!(g, 0.05f0*d)
    t = time_ns(); for _ in 1:nsteps; FV.step!(g, 0.05f0*d); end
    n*n*nsteps/((time_ns()-t)/1e9)/1e6
end
println("threads = ", Threads.nthreads())
for n in (512, 1024, 2048, 4096)
    ns = n <= 1024 ? 30 : 10
    println("Euler 2D nx=$n  SIMD = $(round(bench(n, ns), digits=1)) Mcell/s")
end
