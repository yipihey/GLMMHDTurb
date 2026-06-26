# Validate GLM-MHD through the same contract: Brio-Wu shock tube + rotation isotropy
# (now TWO vectors — momentum AND B — must swap together).
using FiniteVolumeGodunovKA
const FV = FiniteVolumeGodunovKA
import FiniteVolumeGodunovKA: cons2prim, prim2cons

# Euler regression: the vidx convention change + new perm machinery didn't break Euler.
let s = Euler(γ=1.4f0), nx=400, dx=1f0/nx
    xs = Float32[(i-0.5f0)*dx for i in 1:nx]
    U0 = [prim2cons(s, (x<0.5f0 ? 1f0 : 0.125f0, 0f0,0f0,0f0, x<0.5f0 ? 1f0 : 0.1f0)) for x in xs]
    g = Grid1D(s, U0; dx=dx, bc=:outflow, recon=PLM(), rsol=HLLC()); FV.evolve!(g, 0.2f0)
    i = argmin(abs.(xs .- 0.75f0))
    println("Euler 1D Sod post-shock ρ=", round(FV.primitives(g)[i][1], digits=4), " (exact 0.2656)")
end

# Brio-Wu MHD shock tube (γ=2), 1D scalar backend + LLF. W = (ρ,u,v,w,P,Bx,By,Bz,ψ).
s = GLMMHD(γ = 2f0, ch = 2f0)
nx = 800; dx = 1f0/nx; xs = Float32[(i-0.5f0)*dx for i in 1:nx]
L = (1f0,   0f0,0f0,0f0, 1f0, 0.75f0,  1f0, 0f0, 0f0)
R = (0.125f0, 0f0,0f0,0f0, 0.1f0, 0.75f0, -1f0, 0f0, 0f0)
U0 = [prim2cons(s, x < 0.5f0 ? L : R) for x in xs]
g = Grid1D(s, U0; dx=dx, bc=:outflow, recon=PLM(), rsol=LLF(), cfl=0.4f0)
m0 = sum(u[1] for u in U0) * dx
FV.evolve!(g, 0.1f0)
W = FV.primitives(g)
println("Brio-Wu: finite=", all(all(isfinite, w) for w in W),
        "  positivity=", all(w -> w[1] > 0 && w[5] > 0, W))
println("  max|Bx-0.75| (normal field preserved) = ", round(maximum(abs(w[6] - 0.75f0) for w in W), digits=5))
println("  mass rel.err = ", round(abs(sum(u[1] for u in g.U)*dx - m0)/m0, sigdigits=3))
println("  ρ range = (", round(minimum(w[1] for w in W), digits=4), ", ", round(maximum(w[1] for w in W), digits=4), ")")

# Rotation isotropy for GLM-MHD: one x-sweep ≡ one y-sweep on the same problem laid along y
# (swap momentum AND B). Must be bit-exact — the real test of two-vector rotation.
function mhd_isotropy()
    n = 128; d = 1f0/n; mm = 4
    xL = (1f0,0f0,0f0,0f0,1f0, 0.75f0, 1f0,0f0,0f0); xR = (0.125f0,0f0,0f0,0f0,0.1f0, 0.75f0,-1f0,0f0,0f0)
    # y-version: swap (u,v) and (Bx,By)
    yL = (1f0,0f0,0f0,0f0,1f0, 1f0, 0.75f0,0f0,0f0); yR = (0.125f0,0f0,0f0,0f0,0.1f0, -1f0,0.75f0,0f0,0f0)
    xp = [prim2cons(s, i <= n÷2 ? xL : xR) for i in 1:n, _ in 1:mm]
    gx = Grid2D(s, xp; dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=LLF()); FV._sweep_x!(gx, 0.1f0*d)
    yp = [prim2cons(s, j <= n÷2 ? yL : yR) for _ in 1:mm, j in 1:n]
    gy = Grid2D(s, yp; dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=LLF()); FV._sweep_y!(gy, 0.1f0*d)
    maxd = 0f0
    for a in 1:n, b in 1:mm
        ua = gx.U[a, b]; u = gy.U[b, a]
        us = (u[1], u[3], u[2], u[4], u[5], u[7], u[6], u[8], u[9])   # undo (u↔v),(Bx↔By)
        maxd = max(maxd, maximum(abs.(ua .- us)))
    end
    maxd
end
println("GLM-MHD rotation isotropy (x-sweep ≡ y-sweep, momentum+B) max|Δ| = ", mhd_isotropy())
