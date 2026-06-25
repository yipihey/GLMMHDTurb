using FiniteVolumeGodunovKA
using FiniteVolumeGodunovKA: cons2prim, prim2cons, nconserved, vidx
using Test

const FV = FiniteVolumeGodunovKA

@testset "contract / Euler roundtrip" begin
    s = Euler(γ = 1.4f0)
    @test nconserved(s) == 5
    @test vidx(s) == ((2, 3, 4),)
    W = (1.2f0, 0.3f0, -0.5f0, 0.1f0, 0.9f0)
    @test all(isapprox.(cons2prim(s, prim2cons(s, W)), W; rtol = 1f-5))
end

# ---------------------------------------------------------------------------
# Sod shock tube (Float32, HLLC) — exercises cons2prim/prim2cons/physflux/HLLC,
# the limiter, and the conservative update against the known star state.
# ---------------------------------------------------------------------------
@testset "Sod shock tube (Float32)" begin
    nx = 400; dx = 1f0 / nx
    xs = Float32[(i - 0.5f0) * dx for i in 1:nx]
    s  = Euler(γ = 1.4f0)
    U0 = [prim2cons(s, (x < 0.5f0 ? 1f0 : 0.125f0, 0f0, 0f0, 0f0,
                        x < 0.5f0 ? 1f0 : 0.1f0)) for x in xs]
    g  = Grid1D(s, U0; dx = dx, bc = :outflow, recon = PLM(), rsol = HLLC(), cfl = 0.4f0)

    m0 = sum(u[1] for u in U0) * dx
    FV.evolve!(g, 0.2f0)
    W  = FV.primitives(g)

    @test all(w -> w[1] > 0 && w[5] > 0, W)                 # positivity
    @test isapprox(FV.conserved_total(g)[1], m0; rtol = 1f-4)  # mass conserved

    sample(xc) = begin
        i = argmin(abs.(xs .- xc))
        mean5(f) = sum(f(W[k]) for k in i-2:i+2) / 5
        (ρ = mean5(w -> w[1]), u = mean5(w -> w[2]), P = mean5(w -> w[5]))
    end
    postshock = sample(0.75f0)   # between contact (~0.686) and shock (~0.850)
    leftstar  = sample(0.60f0)   # between rarefaction tail and contact
    @test isapprox(postshock.ρ, 0.26557; rtol = 0.05)
    @test isapprox(postshock.P, 0.30313; rtol = 0.05)
    @test isapprox(postshock.u, 0.92745; rtol = 0.05)
    @test isapprox(leftstar.ρ,  0.42632; rtol = 0.06)
    @test isapprox(leftstar.P,  0.30313; rtol = 0.05)
end

# ---------------------------------------------------------------------------
# Smooth convergence — entropy wave advected one period. Run in Float64 to
# demonstrate the SAME physics is element-type-generic; unlimited PLM → 2nd order
# (TVD limiters clip smooth extrema to ~1st, so :none is the right choice here).
# ---------------------------------------------------------------------------
@testset "entropy-wave convergence (Float64, 2nd order)" begin
    ρ0(x) = 1.0 + 0.2 * sinpi(2x)
    run(nx) = begin
        dx = 1.0 / nx
        xs = [(i - 0.5) * dx for i in 1:nx]
        s  = Euler(γ = 1.4f0)
        U0 = [prim2cons(s, (ρ0(x), 1.0, 0.0, 0.0, 1.0)) for x in xs]
        g  = Grid1D(s, U0; dx = dx, bc = :periodic, recon = PLM(:none), rsol = HLL(), cfl = 0.4)
        FV.evolve!(g, 1.0)
        W = FV.primitives(g)
        sum(abs(W[i][1] - ρ0(xs[i])) for i in 1:nx) * dx     # L1, exact = IC at t=1
    end

    ns   = [16, 32, 64, 128]
    errs = [run(n) for n in ns]
    ord  = [log2(errs[k] / errs[k+1]) for k in 1:length(ns)-1]
    @info "entropy-wave convergence" errs ord
    @test all(diff(errs) .< 0)        # errors decrease with resolution
    @test ord[end] ≥ 1.85             # 2nd-order at the finest pair
end

# ---------------------------------------------------------------------------
# SIMD CPU backend — must be BIT-IDENTICAL to the scalar backend (same physics,
# same Float32 ops, just Vec{8} lanes + a scalar tail). Bit-identity is the
# strongest possible proof the vectorized path runs the same code.
# ---------------------------------------------------------------------------
@testset "SIMD backend ≡ scalar (bit-identical)" begin
    s = Euler(γ = 1.4f0)
    cmp(U0, dx, bc; kw...) = begin
        gsc = Grid1D(s, copy(U0); dx = dx, bc = bc, kw...)
        gsi = Grid1DSoA(s, copy(U0); dx = dx, bc = bc, kw...)
        FV.evolve!(gsc, 0.2f0); FV.evolve_simd!(gsi, 0.2f0)
        Wsc, Wsi = FV.primitives(gsc), FV.primitives_soa(gsi)
        maximum(maximum(abs.(Wsc[i] .- Wsi[i])) for i in 1:length(U0))
    end
    nx = 437; dx = 1f0 / nx                       # deliberately not a multiple of 8 → tail
    xs = Float32[(i - 0.5f0) * dx for i in 1:nx]
    sod  = [prim2cons(s, (x < 0.5f0 ? 1f0 : 0.125f0, 0f0,0f0,0f0, x < 0.5f0 ? 1f0 : 0.1f0)) for x in xs]
    wave = [prim2cons(s, (1f0 + 0.2f0*sinpi(2f0*x), 1f0, 0f0, 0f0, 1f0)) for x in xs]
    @test cmp(sod,  dx, :outflow;  recon = PLM(),     rsol = HLLC()) == 0f0
    @test cmp(wave, dx, :periodic; recon = PLM(),     rsol = HLL())  == 0f0
    @test cmp(wave, dx, :periodic; recon = PLM(:none), rsol = LLF()) == 0f0
end

# ---------------------------------------------------------------------------
# 2D + rotation — the design-defining test. The user wrote only physflux_x; the
# y-flux is obtained by rotating the marked vector components. Isotropy must be
# bit-exact, and the Strang-split 2D scheme must be 2nd order.
# ---------------------------------------------------------------------------
@testset "2D rotation isotropy + convergence" begin
    s = Euler(γ = 1.4f0)

    # one x-sweep on an x-varying Sod ≡ one y-sweep on the same profile along y (u↔v).
    n = 96; d = 1f0/n; m = 4
    xprob = [prim2cons(s, (i <= n÷2 ? 1f0 : 0.125f0, 0.3f0,0f0,0f0, i <= n÷2 ? 1f0 : 0.1f0)) for i in 1:n, _ in 1:m]
    gx = Grid2D(s, xprob; dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC()); FV._sweep_x!(gx, 0.1f0*d)
    yprob = [prim2cons(s, (j <= n÷2 ? 1f0 : 0.125f0, 0f0,0.3f0,0f0, j <= n÷2 ? 1f0 : 0.1f0)) for _ in 1:m, j in 1:n]
    gy = Grid2D(s, yprob; dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC()); FV._sweep_y!(gy, 0.1f0*d)
    iso = maximum(maximum(abs.(gx.U[a,b] .- (gy.U[b,a][1], gy.U[b,a][3], gy.U[b,a][2], gy.U[b,a][4], gy.U[b,a][5])))
                  for a in 1:n, b in 1:m)
    @test iso == 0f0                                   # y-flux-via-rotation ≡ x-flux, bit-exact

    # diagonal entropy wave (Float64), exact at t=1 → 2nd order.
    ρ0(x, y) = 1.0 + 0.2 * sinpi(2 * (x + y))
    err2d(nn) = begin
        dx = 1.0/nn
        U0 = [prim2cons(s, (ρ0((i-0.5)*dx, (j-0.5)*dx), 1.0, 1.0, 0.0, 1.0)) for i in 1:nn, j in 1:nn]
        g = Grid2D(s, U0; dx=dx, dy=dx, bc=:periodic, recon=PLM(:none), rsol=HLL(), cfl=0.4)
        FV.evolve2d!(g, 1.0); W = FV.primitives(g)
        sum(abs(W[i,j][1] - ρ0((i-0.5)*dx, (j-0.5)*dx)) for i in 1:nn, j in 1:nn) * dx * dx
    end
    es = [err2d(nn) for nn in (16, 32, 64)]
    @test all(diff(es) .< 0)
    @test log2(es[2] / es[3]) ≥ 1.9                    # 2nd order at the finest pair
end

# ---------------------------------------------------------------------------
# GLM-MHD through the SAME contract — 9 vars, TWO rotating vectors (momentum + B).
# The payoff test: add variables + a param + the flux + vidx-as-two-triples, and the
# library's rotation handles y/z automatically.
# ---------------------------------------------------------------------------
@testset "GLM-MHD via the contract (Brio-Wu + rotation)" begin
    s = GLMMHD(γ = 2f0, ch = 2f0)
    nx = 800; dx = 1f0/nx; xs = Float32[(i-0.5f0)*dx for i in 1:nx]
    L = (1f0,   0f0,0f0,0f0, 1f0, 0.75f0,  1f0, 0f0, 0f0)
    R = (0.125f0, 0f0,0f0,0f0, 0.1f0, 0.75f0, -1f0, 0f0, 0f0)
    U0 = [prim2cons(s, x < 0.5f0 ? L : R) for x in xs]
    g  = Grid1D(s, U0; dx=dx, bc=:outflow, recon=PLM(), rsol=LLF(), cfl=0.4f0)
    m0 = sum(u[1] for u in U0) * dx
    FV.evolve!(g, 0.1f0); W = FV.primitives(g)
    @test all(w -> all(isfinite, w), W)                       # stable
    @test all(w -> w[1] > 0 && w[5] > 0, W)                   # positivity
    @test maximum(abs(w[6] - 0.75f0) for w in W) == 0f0       # normal field Bx exactly preserved
    @test isapprox(sum(u[1] for u in g.U)*dx, m0; rtol = 1f-4)
    # @source hook: the GLM ψ-damping decays ψ (Euler's default source is identity).
    @test FV.source(s, prim2cons(s, (1f0,0f0,0f0,0f0,1f0, 0.5f0,0.5f0,0f0, 1f0)), 0.1f0)[9] < 1f0
    @test FV.source(Euler(γ=1.4f0), (1f0,2f0,3f0,4f0,5f0), 0.1f0) == (1f0,2f0,3f0,4f0,5f0)

    # rotation isotropy with TWO vectors: one x-sweep ≡ one y-sweep (momentum + B swapped).
    n = 96; d = 1f0/n; m = 4
    xL = (1f0,0f0,0f0,0f0,1f0, 0.75f0, 1f0,0f0,0f0); xR = (0.125f0,0f0,0f0,0f0,0.1f0, 0.75f0,-1f0,0f0,0f0)
    yL = (1f0,0f0,0f0,0f0,1f0, 1f0, 0.75f0,0f0,0f0); yR = (0.125f0,0f0,0f0,0f0,0.1f0, -1f0,0.75f0,0f0,0f0)
    gx = Grid2D(s, [prim2cons(s, i<=n÷2 ? xL : xR) for i in 1:n, _ in 1:m];
                dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=LLF()); FV._sweep_x!(gx, 0.1f0*d)
    gy = Grid2D(s, [prim2cons(s, j<=n÷2 ? yL : yR) for _ in 1:m, j in 1:n];
                dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=LLF()); FV._sweep_y!(gy, 0.1f0*d)
    iso = maximum(begin
              u = gy.U[b,a]
              maximum(abs.(gx.U[a,b] .- (u[1],u[3],u[2],u[4],u[5],u[7],u[6],u[8],u[9])))
          end for a in 1:n, b in 1:m)
    @test iso == 0f0                                          # two-vector rotation, bit-exact
end

# ---------------------------------------------------------------------------
# CUDA backend — same physics, T = Float32 on a GPU thread. Must be bit-identical
# to the scalar backend. Skipped automatically when no functional GPU is present.
# ---------------------------------------------------------------------------
using CUDA
if CUDA.functional()
    @testset "CUDA backend ≡ scalar (bit-identical)" begin
        s = Euler(γ = 1.4f0)
        cmp(U0, dx, bc; kw...) = begin
            gsc = Grid1D(s, copy(U0); dx = dx, bc = bc, kw...)
            gcu = Grid1DCU(s, copy(U0); dx = dx, bc = bc, kw...)
            FV.evolve!(gsc, 0.2f0); FV.evolve_cuda!(gcu, 0.2f0)
            Wsc, Wcu = FV.primitives(gsc), FV.primitives_cuda(gcu)
            maximum(maximum(abs.(Wsc[i] .- Wcu[i])) for i in 1:length(U0))
        end
        nx = 4001; dx = 1f0 / nx
        xs = Float32[(i - 0.5f0) * dx for i in 1:nx]
        sod  = [prim2cons(s, (x < 0.5f0 ? 1f0 : 0.125f0, 0f0,0f0,0f0, x < 0.5f0 ? 1f0 : 0.1f0)) for x in xs]
        wave = [prim2cons(s, (1f0 + 0.2f0*sinpi(2f0*x), 1f0, 0f0, 0f0, 1f0)) for x in xs]
        @test cmp(sod,  dx, :outflow;  recon = PLM(), rsol = HLLC()) == 0f0
        @test cmp(wave, dx, :periodic; recon = PLM(), rsol = HLL())  == 0f0
    end

    @testset "2D CUDA ≡ 2D CPU (rotation on GPU)" begin
        s = Euler(γ = 1.4f0); n = 64; d = 1f0/n
        ρ0(x, y) = 1f0 + 0.3f0*sinpi(2f0*(x + y))
        U0 = [prim2cons(s, (ρ0((i-0.5f0)*d, (j-0.5f0)*d), 0.5f0, 0.3f0, 0f0, 1f0)) for i in 1:n, j in 1:n]
        gc = Grid2D(s, copy(U0); dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC())
        gg = Grid2DCU(s, copy(U0); dx=d, dy=d, bc=:periodic, recon=PLM(), rsol=HLLC())
        for _ in 1:15; FV.step!(gc, 0.1f0*d); FV.step!(gg, 0.1f0*d); end
        Wc = FV.primitives(gc); Wg = FV.primitives(gg)
        @test maximum(maximum(abs.(Wc[i,j] .- Wg[i,j])) for i in 1:n, j in 1:n) == 0f0
    end
else
    @info "CUDA not functional — skipping GPU backend tests"
end
