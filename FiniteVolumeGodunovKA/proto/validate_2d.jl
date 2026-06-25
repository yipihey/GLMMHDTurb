# Validate the 2D backend + the rotation design (library rotates physflux_x for y).
using FiniteVolumeGodunovKA
const FV = FiniteVolumeGodunovKA
import FiniteVolumeGodunovKA: cons2prim, prim2cons

s = Euler(γ = 1.4f0)

# 0) 1D smoke — the refactor (moved _halfstep, shared _update_dir) didn't break anything.
let nx = 400, dx = 1f0/nx
    xs = Float32[(i-0.5f0)*dx for i in 1:nx]
    U0 = [prim2cons(s, (x < 0.5f0 ? 1f0 : 0.125f0, 0f0,0f0,0f0, x < 0.5f0 ? 1f0 : 0.1f0)) for x in xs]
    g = Grid1D(s, U0; dx=dx, bc=:outflow, recon=PLM(), rsol=HLLC()); FV.evolve!(g, 0.2f0)
    W = FV.primitives(g); i = argmin(abs.(xs .- 0.75f0))
    println("1D Sod post-shock ρ=", round(W[i][1], digits=4), " (exact 0.2656)")
end

# 1) Rotation isotropy: one x-sweep on an x-varying Sod vs one y-sweep on the SAME profile
# laid along y (normal velocity in v). Result must match under transpose + u↔v swap — bit-exact
# if the y-flux-via-rotation is exactly the x-flux.
function isotropy()
    n = 128; d = 1f0/n; m = 4
    xprob = [prim2cons(s, (i <= n÷2 ? 1f0 : 0.125f0, 0.3f0, 0f0, 0f0, i <= n÷2 ? 1f0 : 0.1f0)) for i in 1:n, _ in 1:m]
    gx = Grid2D(s, xprob; dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC()); FV._sweep_x!(gx, 0.1f0*d)
    yprob = [prim2cons(s, (j <= n÷2 ? 1f0 : 0.125f0, 0f0, 0.3f0, 0f0, j <= n÷2 ? 1f0 : 0.1f0)) for _ in 1:m, j in 1:n]
    gy = Grid2D(s, yprob; dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC()); FV._sweep_y!(gy, 0.1f0*d)
    maxd = 0f0
    for a in 1:n, b in 1:m
        ua = gx.U[a, b]; ub = gy.U[b, a]
        ubs = (ub[1], ub[3], ub[2], ub[4], ub[5])   # undo u↔v
        maxd = max(maxd, maximum(abs.(ua .- ubs)))
    end
    maxd
end
println("rotation isotropy (x-sweep ≡ y-sweep) max|Δ| = ", isotropy())

# 2) 2D diagonal entropy-wave convergence (Float64): ρ = 1 + 0.2 sin(2π(x+y)), u=v=1, exact at t=1.
ρ0(x, y) = 1.0 + 0.2 * sinpi(2 * (x + y))
function err2d(n)
    dx = 1.0/n
    U0 = [prim2cons(s, (ρ0((i-0.5)*dx, (j-0.5)*dx), 1.0, 1.0, 0.0, 1.0)) for i in 1:n, j in 1:n]
    g = Grid2D(s, U0; dx=dx, dy=dx, bc=:periodic, recon=PLM(:none), rsol=HLL(), cfl=0.4)
    FV.evolve2d!(g, 1.0); W = FV.primitives(g)
    e = 0.0
    for i in 1:n, j in 1:n; e += abs(W[i,j][1] - ρ0((i-0.5)*dx, (j-0.5)*dx)); end
    e * dx * dx
end
let es = [err2d(n) for n in (16, 32, 64)]
    println("2D conv L1 = ", es)
    println("2D order   = ", round(log2(es[1]/es[2]), digits=2), ", ", round(log2(es[2]/es[3]), digits=2))
end
