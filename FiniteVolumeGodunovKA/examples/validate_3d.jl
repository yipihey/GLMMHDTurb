# Validate the 3D backend: z-rotation isotropy (the new axis) + 3D convergence.
using FiniteVolumeGodunovKA
const FV = FiniteVolumeGodunovKA
import FiniteVolumeGodunovKA: prim2cons

s = Euler(γ = 1.4f0)

# z-rotation isotropy: one x-sweep on an x-varying Sod ≡ one z-sweep on the same profile along z
# (normal velocity in w). Must be bit-exact — proves dirperm(s,N,3) rotation is correct.
function z_isotropy()
    n = 64; d = 1f0/n; m = 4
    xp = [prim2cons(s, (i <= n÷2 ? 1f0 : 0.125f0, 0.3f0, 0f0, 0f0, i <= n÷2 ? 1f0 : 0.1f0)) for i in 1:n, _ in 1:m, _ in 1:m]
    gx = Grid3D(s, xp; dx=d, dy=d, dz=d, bc=:periodic, recon=PLM(), rsol=HLLC()); FV._sweep_x3d!(gx, 0.1f0*d)
    zp = [prim2cons(s, (k <= n÷2 ? 1f0 : 0.125f0, 0f0, 0f0, 0.3f0, k <= n÷2 ? 1f0 : 0.1f0)) for _ in 1:m, _ in 1:m, k in 1:n]
    gz = Grid3D(s, zp; dx=d, dy=d, dz=d, bc=:periodic, recon=PLM(), rsol=HLLC()); FV._sweep_z3d!(gz, 0.1f0*d)
    maxd = 0f0
    for a in 1:n, b in 1:m, c in 1:m
        u = gz.U[b, c, a]; us = (u[1], u[4], u[3], u[2], u[5])      # undo u↔w
        maxd = max(maxd, maximum(abs.(gx.U[a, b, c] .- us)))
    end
    maxd
end
println("3D z-rotation isotropy (x-sweep ≡ z-sweep) max|Δ| = ", z_isotropy())

# 3D diagonal entropy wave (Float64), exact at t=1 → 2nd order.
ρ0(x, y, z) = 1.0 + 0.2 * sinpi(2 * (x + y + z))
function err3d(n)
    dx = 1.0/n
    U0 = [prim2cons(s, (ρ0((i-0.5)*dx, (j-0.5)*dx, (k-0.5)*dx), 1.0, 1.0, 1.0, 1.0)) for i in 1:n, j in 1:n, k in 1:n]
    g = Grid3D(s, U0; dx=dx, dy=dx, dz=dx, bc=:periodic, recon=PLM(:none), rsol=HLL(), cfl=0.4)
    FV.evolve3d!(g, 1.0); W = FV.primitives(g)
    sum(abs(W[i,j,k][1] - ρ0((i-0.5)*dx, (j-0.5)*dx, (k-0.5)*dx)) for i in 1:n, j in 1:n, k in 1:n) * dx^3
end
let es = [err3d(n) for n in (16, 24, 32)]
    println("3D conv L1 = ", round.(es, sigdigits=3))
    println("3D order (16→24, 24→32) = ", round(log(es[1]/es[2])/log(24/16), digits=2), ", ", round(log(es[2]/es[3])/log(32/24), digits=2))
end

# GLM-MHD 3D stability smoke (a few steps, finite + positive).
let sm = GLMMHD(γ=5f0/3f0, ch=2f0), n = 24, d = 1f0/n, B0 = 1f0/sqrt(4f0*Float32(π))
    icW(x,y,z) = (0.5f0, -sinpi(2f0*y), sinpi(2f0*x), 0.1f0*sinpi(2f0*z), 0.5f0, -B0*sinpi(2f0*y), B0*sinpi(4f0*x), 0f0, 0f0)
    U0 = [prim2cons(sm, icW((i-0.5f0)*d, (j-0.5f0)*d, (k-0.5f0)*d)) for i in 1:n, j in 1:n, k in 1:n]
    g = Grid3D(sm, U0; dx=d, dy=d, dz=d, bc=:periodic, recon=PLM(), rsol=HLLD(), cfl=0.4f0)
    FV.evolve3d!(g, 0.05f0); W = FV.primitives(g)
    println("GLM-MHD 3D (HLLD, t=0.05): finite=", all(all(isfinite,w) for w in W), " pos=", all(w->w[1]>0 && w[5]>0, W))
end
