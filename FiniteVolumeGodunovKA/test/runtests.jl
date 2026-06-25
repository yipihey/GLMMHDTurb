using FiniteVolumeGodunovKA
using FiniteVolumeGodunovKA: cons2prim, prim2cons, nconserved, vidx
using Test

const FV = FiniteVolumeGodunovKA

@testset "contract / Euler roundtrip" begin
    s = Euler(γ = 1.4f0)
    @test nconserved(s) == 5
    @test vidx(s) == (2, 3, 4)
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
