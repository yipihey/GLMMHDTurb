# Validate HLLD: Brio-Wu (stable/positive/sharper than LLF) + rotation isotropy (bit-exact).
using FiniteVolumeGodunovKA
const FV = FiniteVolumeGodunovKA
import FiniteVolumeGodunovKA: prim2cons, cons2prim

s = GLMMHD(γ = 2f0, ch = 2f0)
nx = 800; dx = 1f0/nx; xs = Float32[(i-0.5f0)*dx for i in 1:nx]
bw(i) = i <= nx÷2 ? (1f0,0f0,0f0,0f0,1f0,0.75f0,1f0,0f0,0f0) : (0.125f0,0f0,0f0,0f0,0.1f0,0.75f0,-1f0,0f0,0f0)
U0 = [prim2cons(s, bw(i)) for i in 1:nx]

function run(rsol)
    g = Grid1D(s, [u for u in U0]; dx=dx, bc=:outflow, recon=PLM(), rsol=rsol, cfl=0.4f0)
    m0 = sum(u[1] for u in U0) * dx
    FV.evolve!(g, 0.1f0); W = FV.primitives(g)
    # total variation of density = sharpness proxy (higher TV = sharper, less diffusion)
    tv = sum(abs(W[i+1][1] - W[i][1]) for i in 1:nx-1)
    (W=W, finite=all(all(isfinite,w) for w in W), pos=all(w->w[1]>0 && w[5]>0, W),
     Bx=maximum(abs(w[6]-0.75f0) for w in W), mass=abs(sum(u[1] for u in g.U)*dx - m0)/m0, tv=tv)
end

rl = run(LLF()); rd = run(HLLD())
println("Brio-Wu LLF : finite=$(rl.finite) pos=$(rl.pos) max|Bx-.75|=$(round(rl.Bx,sigdigits=2)) mass=$(round(rl.mass,sigdigits=2)) TV=$(round(rl.tv,digits=2))")
println("Brio-Wu HLLD: finite=$(rd.finite) pos=$(rd.pos) max|Bx-.75|=$(round(rd.Bx,sigdigits=2)) mass=$(round(rd.mass,sigdigits=2)) TV=$(round(rd.tv,digits=2))")
println("HLLD/LLF total-variation ratio (sharper>1) = ", round(rd.tv/rl.tv, digits=3))

# Rotation isotropy: one x-sweep ≡ one y-sweep (momentum + B swapped), bit-exact.
function isotropy()
    n = 96; d = 1f0/n; m = 4
    xL=(1f0,0f0,0f0,0f0,1f0,0.75f0,1f0,0f0,0f0); xR=(0.125f0,0f0,0f0,0f0,0.1f0,0.75f0,-1f0,0f0,0f0)
    yL=(1f0,0f0,0f0,0f0,1f0,1f0,0.75f0,0f0,0f0); yR=(0.125f0,0f0,0f0,0f0,0.1f0,-1f0,0.75f0,0f0,0f0)
    gx=Grid2D(s,[prim2cons(s, i<=n÷2 ? xL : xR) for i in 1:n, _ in 1:m]; dx=d,dy=d,bc=:periodic,recon=PLM(),rsol=HLLD()); FV._sweep_x!(gx,0.1f0*d)
    gy=Grid2D(s,[prim2cons(s, j<=n÷2 ? yL : yR) for _ in 1:m, j in 1:n]; dx=d,dy=d,bc=:periodic,recon=PLM(),rsol=HLLD()); FV._sweep_y!(gy,0.1f0*d)
    maximum(begin u=gy.U[b,a]; maximum(abs.(gx.U[a,b] .- (u[1],u[3],u[2],u[4],u[5],u[7],u[6],u[8],u[9]))) end for a in 1:n, b in 1:m)
end
println("HLLD rotation isotropy max|Δ| = ", isotropy())
