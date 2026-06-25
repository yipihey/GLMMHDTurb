# Validate the 2D SIMD backend: bit-identical to the scalar 2D backend + throughput.
using FiniteVolumeGodunovKA
const FV = FiniteVolumeGodunovKA
import FiniteVolumeGodunovKA: prim2cons

# bit-identical over fixed-dt steps (step! → no dynamic-ch/dt differences).
function bitcmp(s, U0, dx, bc, recon, rsol, nsteps)
    gsc = Grid2D(s, copy(U0); dx=dx, dy=dx, bc=bc, recon=recon, rsol=rsol)
    gsi = Grid2DSoA(s, copy(U0); dx=dx, dy=dx, bc=bc, recon=recon, rsol=rsol)
    for _ in 1:nsteps; FV.step!(gsc, 0.1f0*dx); FV.step!(gsi, 0.1f0*dx); end
    Wc = FV.primitives(gsc); Wi = FV.primitives_soa(gsi)
    maximum(maximum(abs.(Wc[i,j] .- Wi[i,j])) for i in 1:size(U0,1), j in 1:size(U0,2))
end

let se = Euler(γ=1.4f0), n = 128, d = 1f0/n
    ρ0(x,y) = 1f0 + 0.3f0*sinpi(2f0*(x+y))
    U0 = [prim2cons(se, (ρ0((i-0.5f0)*d,(j-0.5f0)*d), 0.5f0, 0.3f0, 0f0, 1f0)) for i in 1:n, j in 1:n]
    println("Euler 2D  SIMD ≡ scalar (20 steps, HLLC)  max|Δ| = ", bitcmp(se, U0, d, :periodic, PLM(), HLLC(), 20))
end
let sm = GLMMHD(γ=5f0/3f0, ch=2f0), n = 96, d = 1f0/n
    B0 = 1f0/sqrt(4f0*Float32(π))
    icW(x,y) = (0.5f0, -sinpi(2f0*y), sinpi(2f0*x), 0f0, 0.5f0, -B0*sinpi(2f0*y), B0*sinpi(4f0*x), 0f0, 0f0)
    U0 = [prim2cons(sm, icW((i-0.5f0)*d,(j-0.5f0)*d)) for i in 1:n, j in 1:n]
    println("GLM-MHD 2D SIMD ≡ scalar (15 steps, LLF)   max|Δ| = ", bitcmp(sm, U0, d, :periodic, PLM(), LLF(), 15))
    println("GLM-MHD 2D SIMD ≡ scalar (15 steps, HLLD)  max|Δ| = ", bitcmp(sm, U0, d, :periodic, PLM(), HLLD(), 15))
end

# throughput: scalar 2D vs SIMD 2D (Euler)
function bench(n, nsteps)
    s = Euler(γ=1.4f0); d = 1f0/n
    U0 = [prim2cons(s, (1f0+0.2f0*sinpi(2f0*((i-0.5f0)*d+(j-0.5f0)*d)), 0.5f0, 0.3f0, 0f0, 1f0)) for i in 1:n, j in 1:n]
    gsc = Grid2D(s, copy(U0); dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    gsi = Grid2DSoA(s, copy(U0); dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    b(g) = (FV.step!(g, 0.05f0*d); t=time_ns(); for _ in 1:nsteps; FV.step!(g, 0.05f0*d); end; n*n*nsteps/((time_ns()-t)/1e9)/1e6)
    (b(gsc), b(gsi))
end
for n in (256, 512)
    msc, msi = bench(n, 50); println("Euler 2D nx=$n  scalar=$(round(msc,digits=1))  SIMD=$(round(msi,digits=1))  ->  $(round(msi/msc,digits=2))x")
end
