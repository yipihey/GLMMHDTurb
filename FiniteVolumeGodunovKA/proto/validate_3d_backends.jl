# 3D SIMD + CUDA backends must be bit-identical to the scalar 3D backend.
using FiniteVolumeGodunovKA, CUDA
const FV = FiniteVolumeGodunovKA
import FiniteVolumeGodunovKA: prim2cons

function bitcmp(make_other, get_other, s, U0, recon, rsol, nsteps)
    n = size(U0,1); d = 1f0/n
    gsc = Grid3D(s, copy(U0); dx=d,dy=d,dz=d, bc=:periodic, recon=recon, rsol=rsol)
    go  = make_other(s, copy(U0); dx=d,dy=d,dz=d, bc=:periodic, recon=recon, rsol=rsol)
    for _ in 1:nsteps; FV.step!(gsc, 0.1f0*d); FV.step!(go, 0.1f0*d); end
    Wc = FV.primitives(gsc); Wo = get_other(go)
    maximum(maximum(abs.(Wc[i,j,k] .- Wo[i,j,k])) for i in 1:n, j in 1:n, k in 1:n)
end

se = Euler(γ=1.4f0); n = 32
U0e = [prim2cons(se, (1f0+0.3f0*sinpi(2f0*((i+j+k-1.5f0)/n)), 0.5f0,0.3f0,0.2f0, 1f0)) for i in 1:n, j in 1:n, k in 1:n]
sm = GLMMHD(γ=5f0/3f0, ch=2f0); B0 = 1f0/sqrt(4f0*Float32(π))
icW(x,y,z) = (0.5f0, -sinpi(2f0*y), sinpi(2f0*x), 0.1f0, 0.5f0, -B0*sinpi(2f0*y), B0*sinpi(4f0*x), 0f0, 0f0)
U0m = [prim2cons(sm, icW((i-0.5f0)/n,(j-0.5f0)/n,(k-0.5f0)/n)) for i in 1:n, j in 1:n, k in 1:n]

println("3D SIMD ≡ scalar (Euler/HLLC) max|Δ| = ", bitcmp(Grid3DSoA, FV.primitives_soa, se, U0e, PLM(), HLLC(), 10))
println("3D SIMD ≡ scalar (GLM/HLLD)  max|Δ| = ", bitcmp(Grid3DSoA, FV.primitives_soa, sm, U0m, PLM(), HLLD(), 8))
println("3D CUDA ≡ scalar (Euler/HLLC) max|Δ| = ", bitcmp(Grid3DCU, FV.primitives, se, U0e, PLM(), HLLC(), 10))
println("3D CUDA ≡ scalar (GLM/HLLD)  max|Δ| = ", bitcmp(Grid3DCU, FV.primitives, sm, U0m, PLM(), HLLD(), 8))

# throughput: scalar vs SIMD vs CUDA (Euler 3D)
function bench(n, nsteps)
    s = Euler(γ=1.4f0); d = 1f0/n
    U0 = [prim2cons(s, (1f0+0.2f0*sinpi(2f0*Float32(i+j+k)/n), 0.5f0,0.3f0,0.2f0, 1f0)) for i in 1:n, j in 1:n, k in 1:n]
    gsc=Grid3D(s,copy(U0);dx=d,dy=d,dz=d,bc=:periodic,recon=PLM(),rsol=HLLC())
    gsi=Grid3DSoA(s,copy(U0);dx=d,dy=d,dz=d,bc=:periodic,recon=PLM(),rsol=HLLC())
    gg =Grid3DCU(s,copy(U0);dx=d,dy=d,dz=d,bc=:periodic,recon=PLM(),rsol=HLLC())
    bc(g)=(FV.step!(g,0.05f0*d); t=time_ns(); for _ in 1:nsteps; FV.step!(g,0.05f0*d); end; n^3*nsteps/((time_ns()-t)/1e9)/1e6)
    bg(g)=(FV.step!(g,0.05f0*d); CUDA.synchronize(); t=time_ns(); for _ in 1:nsteps; FV.step!(g,0.05f0*d); end; CUDA.synchronize(); n^3*nsteps/((time_ns()-t)/1e9)/1e6)
    (bc(gsc), bc(gsi), bg(gg))
end
let (msc,msi,mg) = bench(128, 20)
    println("Euler 3D nx=128  scalar=$(round(msc,digits=1))  SIMD=$(round(msi,digits=1))  CUDA=$(round(mg,digits=0))  Mcell/s")
end
