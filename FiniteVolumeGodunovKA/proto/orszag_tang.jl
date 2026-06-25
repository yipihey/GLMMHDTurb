# 2D CUDA backend validation + Orszag-Tang vortex (GLM-MHD on the GPU).
using FiniteVolumeGodunovKA, CUDA
const FV = FiniteVolumeGodunovKA
import FiniteVolumeGodunovKA: cons2prim, prim2cons, maxspeed_x

# 1) Grid2DCU must be bit-identical to the CPU Grid2D (Euler, periodic, with rotation in y).
function check_gpu_2d()
    s = Euler(γ=1.4f0); n = 128; d = 1f0/n
    ρ0(x, y) = 1f0 + 0.3f0*sinpi(2f0*(x + y))
    U0 = [prim2cons(s, (ρ0((i-0.5f0)*d, (j-0.5f0)*d), 0.5f0, 0.3f0, 0f0, 1f0)) for i in 1:n, j in 1:n]
    gc = Grid2D(s, copy(U0); dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    gg = Grid2DCU(s, copy(U0); dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC())
    for _ in 1:20; FV.step!(gc, 0.1f0*d); FV.step!(gg, 0.1f0*d); end
    Wc = FV.primitives(gc); Wg = FV.primitives(gg)
    maximum(maximum(abs.(Wc[i,j] .- Wg[i,j])) for i in 1:n, j in 1:n)
end
println("Grid2DCU ≡ Grid2D (Euler 2D, 20 steps) max|Δ| = ", check_gpu_2d())

# 2) Orszag-Tang vortex (standard normalization).
function orszag_tang(n)
    γ = 5f0/3f0; B0 = 1f0/sqrt(4f0*Float32(π)); d = 1f0/n
    ρ = 25f0/(36f0*Float32(π)); P = 5f0/(12f0*Float32(π))
    xs = Float32[(i-0.5f0)*d for i in 1:n]
    icW(x, y) = (ρ, -sinpi(2f0*y), sinpi(2f0*x), 0f0, P, -B0*sinpi(2f0*y), B0*sinpi(4f0*x), 0f0, 0f0)
    ch = 0f0
    let s0 = GLMMHD(γ=γ, ch=0f0)
        for i in 1:n, j in 1:n; ch = max(ch, maxspeed_x(s0, icW(xs[i], xs[j]))); end
    end
    s = GLMMHD(γ=γ, ch=ch)
    U0 = [prim2cons(s, icW(xs[i], xs[j])) for i in 1:n, j in 1:n]
    g = Grid2DCU(s, U0; dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=LLF(), cfl=0.4f0)
    FV.step!(g, 1f-6)                              # warmup/compile
    g = Grid2DCU(s, U0; dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=LLF(), cfl=0.4f0)
    t0 = time_ns(); FV.evolve2d!(g, 0.5f0); el = (time_ns()-t0)/1e9
    W = FV.primitives(g)
    gp(i) = mod1(i, n); maxdiv = 0f0; rmin = 1f9; rmax = -1f9
    for i in 1:n, j in 1:n
        db = abs((W[gp(i+1),j][6]-W[gp(i-1),j][6])/(2d) + (W[i,gp(j+1)][7]-W[i,gp(j-1)][7])/(2d))
        maxdiv = max(maxdiv, db); r = W[i,j][1]; rmin = min(rmin, r); rmax = max(rmax, r)
    end
    println("OT $(n)² : ch=$(round(ch,digits=2)) wall=$(round(el,digits=2))s finite=",
            all(all(isfinite, W[i,j]) for i in 1:n, j in 1:n),
            " ρ∈($(round(rmin,digits=3)),$(round(rmax,digits=3))) max|divB|=$(round(maxdiv,digits=3))")
    dens = Float32[W[i,j][1] for i in 1:n, j in 1:n]
    open("proto/ot_density.bin", "w") do io; write(io, dens); end
    open("proto/ot_meta.txt", "w") do io; println(io, n); end
    println("saved proto/ot_density.bin ($(n)x$(n))")
end
orszag_tang(256)
