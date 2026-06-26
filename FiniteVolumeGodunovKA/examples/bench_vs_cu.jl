# 3D CUDA throughput at the .cu reference grid sizes, to quantify the gap vs the hand-tuned .cu.
# .cu reference (gradient IC @480³, Mcell/s): hydro PLM 6865, GLM PLM 3175.
using FiniteVolumeGodunovKA, CUDA
const FV = FiniteVolumeGodunovKA
import FiniteVolumeGodunovKA: prim2cons

function bench(make, s, n, nsteps)
    d = 1f0/n
    # gradient IC (matches the .cu convention — avoids MonCen early-return inflation)
    ph(i,j,k) = 0.001f0*Float32(mod(i*7+j*13+k*17, 911))
    nv = FV.nconserved(s)
    U0 = if nv == 5
        [prim2cons(s, (1f0+ph(i,j,k), 0.3f0, 0.2f0, 0.1f0, 1f0+ph(i,j,k))) for i in 1:n, j in 1:n, k in 1:n]
    else
        [prim2cons(s, (1f0+ph(i,j,k), 0.3f0,0.2f0,0.1f0, 1f0+ph(i,j,k), 0.5f0,0.4f0,0.3f0, 0f0)) for i in 1:n, j in 1:n, k in 1:n]
    end
    g = make(s, U0; dx=d, dy=d, dz=d, bc=:periodic, recon=PLM(), rsol=(nv==5 ? HLLC() : LLF()))
    FV.step!(g, 1f-6); CUDA.synchronize()
    t = time_ns(); for _ in 1:nsteps; FV.step!(g, 1f-6); end; CUDA.synchronize()
    n^3*nsteps/((time_ns()-t)/1e9)/1e6
end

se = Euler(γ=1.4f0); sm = GLMMHD(γ=5f0/3f0, ch=2f0)
for n in (256, 384, 480)
    me = bench(Grid3DCU, se, n, 20)
    println("Euler 3D CUDA nx=$n : $(round(me,digits=0)) Mcell/s   ($(round(100*me/6865,digits=0))% of .cu hydro 6865)")
end
for n in (256, 384)
    mg = bench(Grid3DCU, sm, n, 20)
    println("GLM   3D CUDA nx=$n : $(round(mg,digits=0)) Mcell/s   ($(round(100*mg/3175,digits=0))% of .cu GLM 3175)")
end
