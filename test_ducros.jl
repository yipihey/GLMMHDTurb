# test_ducros.jl — CPU stability/accuracy tests of the Ducros-switched dissipation
# (riemann_duc / flux_duc_kernel! on the portable multi-kernel path).
#
#   julia --project=. -t 4 test_ducros.jl ot   [N]          # Orszag-Tang, several configs
#   julia --project=. -t 4 test_ducros.jl turb [N] [t_end]  # Mach-10 driven turbulence + spectra
#
# Writes spectra to ducros_out/ (one CSV per config).
include("glmmhd_turb.jl")
using .GLMMHDTurb
using FFTW, Printf, Statistics
const G = GLMMHDTurb
const NSTEPMAX = parse(Int, get(ENV, "NSTEPMAX", "20000"))
const AMP = parse(Float32, get(ENV, "AMP", "130"))

# shell-averaged power spectrum of a 3-component field (N^3 each), k in units of 2pi/L
function shell_spectrum(fx, fy, fz)
    N = size(fx, 1)
    P = abs2.(fft(fx)) .+ abs2.(fft(fy)) .+ abs2.(fft(fz))
    nk = N ÷ 2
    E = zeros(Float64, nk + 1)
    kf(i) = i - 1 <= N ÷ 2 ? i - 1 : i - 1 - N
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        kk = round(Int, sqrt(kf(i)^2 + kf(j)^2 + kf(k)^2))
        kk <= nk && (E[kk+1] += P[i, j, k])
    end
    return E ./ Float64(N)^6
end

function spectra(u, p)
    rho = Float64.(u[:,:,:,1]); r = max.(rho, p.smallr)
    sq = sqrt.(r)
    # kinetic: sqrt(rho) v ;  velocity: v ;  magnetic: B
    vx = u[:,:,:,2] ./ r; vy = u[:,:,:,3] ./ r; vz = u[:,:,:,4] ./ r
    Ek = shell_spectrum(sq .* vx, sq .* vy, sq .* vz)
    Ev = shell_spectrum(vx, vy, vz)
    Eb = shell_spectrum(Float64.(u[:,:,:,6]), Float64.(u[:,:,:,7]), Float64.(u[:,:,:,8]))
    return Ek, Ev, Eb
end

# fields: base riemann, ducros on/off, phimin floor, rjump density guard, widen (4-cell
# stencil), thr (binary: phi>thr -> 1), rlow (full dissipation where rho < rlow)
C(name, riemann, ducros; phimin=0.0f0, rjump=0.0f0, widen=false, thr=0.0f0, rlow=0.0f0) =
    (; name, riemann, ducros, phimin, rjump, widen, thr, rlow)
const CONFIGS = [
    C("hll",            :hll,  false),
    C("hlld",           :hlld, false),
    C("hll_duc0",       :hll,  true),
    C("hll_duc0.1",     :hll,  true; phimin=0.1f0),
    C("hll_duc0.1_rj2", :hll,  true; phimin=0.1f0, rjump=2.0f0),
    C("hlld_duc0.1",    :hlld, true; phimin=0.1f0),
    C("hll_duc0.3",     :hll,  true; phimin=0.3f0),
    C("hll_wide0.1",    :hll,  true; phimin=0.1f0, widen=true),
    C("hll_wide_thr",   :hll,  true; phimin=0.1f0, widen=true, thr=0.2f0),
    C("hll_wide_thr_rl",:hll,  true; phimin=0.1f0, widen=true, thr=0.2f0, rlow=0.1f0),
    C("hll_rl",         :hll,  true; phimin=0.1f0, rlow=0.1f0),
]

function pick(names)
    isempty(names) && return CONFIGS
    return [c for c in CONFIGS if c.name in names]
end

function test_ot(N; names=String[], t_end=0.5f0)
    for c in pick(names)
        t0 = time()
        d = G.run_ot(N=N, t_end=t_end, backend=:cpu, pin=:none, riemann=c.riemann,
                     ducros=c.ducros, phimin=c.phimin, rjump=c.rjump)
        @printf("  -> %-16s wall=%.1fs\n", c.name, time() - t0)
    end
end

function test_turb(N, t_end; names=String[], tavg0=0.3f0, nsnap=6, outdir=get(ENV,"OUTDIR","ducros_out"))
    mkpath(outdir)
    for c in pick(names)
        snaps = collect(range(tavg0, t_end; length=nsnap)) .|> Float32
        acc = Ref{Any}(nothing); ns = Ref(0); phistats = Float64[]
        cb = function (u, t, step, sen)
            Ek, Ev, Eb = spectra(Array(u), G.Params(N=N))
            acc[] = acc[] === nothing ? (Ek, Ev, Eb) : (acc[][1] .+ Ek, acc[][2] .+ Ev, acc[][3] .+ Eb)
            ns[] += 1
            if sen !== nothing
                s = Array(sen)
                push!(phistats, mean(s .> 0.5))
            end
        end
        t0 = time()
        d = G.run_turb(N=N, t_end=t_end, amp=AMP, backend=:cpu, pin=:none, ncontrol=400,
                       riemann=c.riemann, ducros=c.ducros, phimin=c.phimin, rjump=c.rjump,
                       widen=c.widen, thr=c.thr, rlow=c.rlow, nstepmax=NSTEPMAX,
                       snap_times=snaps, snap_cb=cb)
        wall = time() - t0
        fsh = isempty(phistats) ? NaN : mean(phistats)
        @printf("  -> %-16s wall=%.1fs  Mach=%.2f finite=%s  cells(theta>0.5)=%.1f%%  nsnap=%d\n",
                c.name, wall, d.mach, d.finite, 100*fsh, ns[])
        if ns[] > 0
            Ek, Ev, Eb = acc[] .|> (x -> x ./ ns[])
            open(joinpath(outdir, "spec_N$(N)_$(c.name).csv"), "w") do io
                println(io, "k,Ekin,Evel,Emag")
                for k in eachindex(Ek)
                    @printf(io, "%d,%.6e,%.6e,%.6e\n", k - 1, Ek[k], Ev[k], Eb[k])
                end
            end
        end
    end
end

mode = length(ARGS) >= 1 ? ARGS[1] : "ot"
N    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 64
rest = length(ARGS) >= 4 ? split(ARGS[4], ",") .|> String : String[]
if mode == "ot"
    test_ot(N; names=rest, t_end=length(ARGS) >= 3 ? parse(Float32, ARGS[3]) : 0.5f0)
else
    test_turb(N, length(ARGS) >= 3 ? parse(Float32, ARGS[3]) : 0.6f0; names=rest)
end
