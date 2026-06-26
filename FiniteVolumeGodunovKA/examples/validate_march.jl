# Validate Grid3DCuMarch as a REAL solver (not just a throughput demo):
#   1. convergence order on a smooth entropy wave (exact solution known)
#   2. conservation (periodic box: mass + energy to ~machine precision)
#   3. agreement with the validated scalar backend (same problem, scheme-level)
#
#   julia -O3 --project=. proto/validate_march.jl

using FiniteVolumeGodunovKA, CUDA, Printf
const FV = FiniteVolumeGodunovKA

s = Euler(γ = 1.4f0)
const A = 0.2f0                              # entropy-wave amplitude (uniform u=v=w=1, uniform P)
ρexact(x, y, z, t) = 1f0 + A * sinpi(2f0 * (x + y + z - 3f0 * t))

# IC as Array{NTuple{5,Float32},3}; cell centers at (i-0.5)/n
function ic(n)
    [FV.prim2cons(s, (ρexact((i-0.5f0)/n, (j-0.5f0)/n, (k-0.5f0)/n, 0f0), 1f0, 1f0, 1f0, 1f0))
     for i in 1:n, j in 1:n, k in 1:n]
end

density(g::Grid3DCuMarch) = [w[1] for w in primitives(g)]

# ---- 1. convergence ------------------------------------------------------
function l1_err(n; tend = 0.1f0)
    g = Grid3DCuMarch(s, ic(n); dx = 1f0 / n)
    evolve!(g, tend; cfl = 0.4f0, dtevery = 4)
    ρ = density(g)
    err = 0.0
    for k in 1:n, j in 1:n, i in 1:n
        err += abs(ρ[i,j,k] - ρexact((i-0.5f0)/n, (j-0.5f0)/n, (k-0.5f0)/n, tend))
    end
    err / n^3
end

println("== 1. convergence (smooth entropy wave, L1 vs exact, t=0.1) ==")
ns = (16, 24, 32, 48); errs = Float64[]
for n in ns
    e = l1_err(n); push!(errs, e)
    ord = length(errs) > 1 ? log(errs[end-1]/e) / log(n/ns[length(errs)-1]) : NaN
    @printf("  n=%-3d  L1=%.3e   order=%s\n", n, e, isnan(ord) ? "  —" : @sprintf("%.2f", ord))
end

# ---- 2. conservation -----------------------------------------------------
println("== 2. conservation (periodic, t=0.2) ==")
let n = 48
    g = Grid3DCuMarch(s, ic(n); dx = 1f0 / n)
    c0 = conserved_total(g)
    evolve!(g, 0.2f0; cfl = 0.4f0, dtevery = 4)
    c1 = conserved_total(g)
    @printf("  Δmass/mass   = %.2e\n", abs(c1[1]-c0[1])/abs(c0[1]))
    @printf("  Δenergy/energy = %.2e\n", abs(c1[5]-c0[5])/abs(c0[5]))
end

# ---- 3. scalar-backend agreement -----------------------------------------
println("== 3. agreement with scalar Grid3D (entropy wave, n=48, t=0.1) ==")
let n = 48, tend = 0.1f0
    g = Grid3DCuMarch(s, ic(n); dx = 1f0 / n);             evolve!(g, tend; cfl = 0.4f0, dtevery = 4)
    gs = Grid3D(s, ic(n); dx = 1f0/n, dy = 1f0/n, dz = 1f0/n, bc = :periodic, cfl = 0.4f0)
    evolve3d!(gs, tend)
    ρm = density(g); ρs = [w[1] for w in primitives(gs)]
    d = sum(abs, ρm .- ρs) / n^3
    @printf("  L1(march − scalar) = %.3e   (both 2nd-order; different scheme+Riemann)\n", d)
end
println("done.")
