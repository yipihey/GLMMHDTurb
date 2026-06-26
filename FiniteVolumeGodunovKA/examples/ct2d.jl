# 2D Constrained Transport prototype (planar B, Orszag-Tang). Proves the CT property:
# div·B stays at machine zero, vs GLM cleaning's ~few %. 1st-order, periodic, unsplit.
#
# Face-staggered B: bx[i,j] = Bx on the LEFT x-face of cell i; by[i,j] = By on the BOTTOM
# y-face of cell j. Cell-centered B = face average (for the Riemann). The magnetic field is
# advanced by the curl of an edge EMF Ez (cell corners) built from the Godunov magnetic fluxes:
#   ∂Bx/∂t = -∂Ez/∂y,  ∂By/∂t = +∂Ez/∂x   ⇒ discrete div·B is preserved exactly.
using FiniteVolumeGodunovKA
const FV = FiniteVolumeGodunovKA
import FiniteVolumeGodunovKA: riemann, fastspeed_x

const S9 = GLMMHD                                    # ideal-MHD flux via ch=0 (no GLM terms)
@inline swapy(t) = (t[1],t[3],t[2],t[4],t[5],t[7],t[6],t[8],t[9])

# primitive (ρ,u,v,w,P,Bx,By,0,0) from cell conserved (ρ,ρu,ρv,ρw,E) + cell-centered B.
@inline function ctprim(γ, U, Bxc, Byc)
    ρ,mx,my,mz,E = U; iρ = 1f0/ρ; u,v,w = mx*iρ, my*iρ, mz*iρ
    P = (γ-1)*(E - 0.5f0*ρ*(u*u+v*v+w*w) - 0.5f0*(Bxc*Bxc+Byc*Byc))
    (ρ,u,v,w,P,Bxc,Byc,0f0,0f0)
end
@inline ovr(W, B) = (W[1],W[2],W[3],W[4],W[5], B, W[7],0f0,0f0)   # override normal Bx
@inline ovry(W, B) = (W[1],W[2],W[3],W[4],W[5], W[6], B,0f0,0f0)  # override normal By

function ct_step!(U, bx, by, s, dx, dy, dt, nx, ny)
    γ = s.γ; gp(i,n) = mod1(i,n)
    Bxc = [0.5f0*(bx[i,j]+bx[gp(i+1,nx),j]) for i in 1:nx, j in 1:ny]
    Byc = [0.5f0*(by[i,j]+by[i,gp(j+1,ny)]) for i in 1:nx, j in 1:ny]
    Fx = Array{NTuple{9,Float32}}(undef, nx, ny)     # flux at left x-face of cell i
    Fy = Array{NTuple{9,Float32}}(undef, nx, ny)     # flux at bottom y-face of cell j
    exf = zeros(Float32, nx, ny); eyf = zeros(Float32, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        iL = gp(i-1,nx)
        WL = ovr(ctprim(γ, U[iL,j], Bxc[iL,j], Byc[iL,j]), bx[i,j])
        WR = ovr(ctprim(γ, U[i,j],  Bxc[i,j],  Byc[i,j]),  bx[i,j])
        Fx[i,j] = riemann(LLF(), s, WL, WR); exf[i,j] = -Fx[i,j][7]   # Ez at x-face = -F[By]
        jL = gp(j-1,ny)
        YL = ovry(ctprim(γ, U[i,jL], Bxc[i,jL], Byc[i,jL]), by[i,j])
        YR = ovry(ctprim(γ, U[i,j],  Bxc[i,j],  Byc[i,j]),  by[i,j])
        Fy[i,j] = swapy(riemann(LLF(), s, swapy(YL), swapy(YR))); eyf[i,j] = Fy[i,j][6]  # Ez at y-face = F[Bx]
    end
    # corner Ez = average of the 4 adjacent face EMFs
    Ez = [0.25f0*(exf[i,j] + exf[i,gp(j-1,ny)] + eyf[i,j] + eyf[gp(i-1,nx),j]) for i in 1:nx, j in 1:ny]
    Un = similar(U)
    @inbounds for j in 1:ny, i in 1:nx                 # cell-centered hydro update (5 vars)
        fxr = Fx[gp(i+1,nx),j]; fyr = Fy[i,gp(j+1,ny)]
        Un[i,j] = ntuple(c -> U[i,j][c] - dt/dx*(fxr[c]-Fx[i,j][c]) - dt/dy*(fyr[c]-Fy[i,j][c]), 5)
    end
    @inbounds for j in 1:ny, i in 1:nx                 # CT face-B update (curl of Ez)
        bx[i,j] -= dt/dy*(Ez[i,gp(j+1,ny)] - Ez[i,j])
        by[i,j] += dt/dx*(Ez[gp(i+1,nx),j] - Ez[i,j])
    end
    copyto!(U, Un)
end

function divB(bx, by, dx, dy, nx, ny)
    gp(i,n)=mod1(i,n); m=0f0
    for j in 1:ny, i in 1:nx
        m = max(m, abs((bx[gp(i+1,nx),j]-bx[i,j])/dx + (by[i,gp(j+1,ny)]-by[i,j])/dy))
    end
    m
end

# Orszag-Tang IC with FACE-centered B (div·B = 0 exactly).
n = 192; dx = 1f0/n; dy = dx; γ = 5f0/3f0; B0 = 1f0/sqrt(4f0*Float32(π))
ρ0 = 25f0/(36f0*Float32(π)); P0 = 5f0/(12f0*Float32(π))
s = GLMMHD(γ=γ, ch=0f0)
xc(i) = (i-0.5f0)*dx; yc(j) = (j-0.5f0)*dy
bx = [(-B0*sinpi(2f0*yc(j))) for i in 1:n, j in 1:n]          # Bx on x-face (i-1/2): depends on y only
by = [( B0*sinpi(4f0*((i-1f0)*dx))) for i in 1:n, j in 1:n]   # By on y-face (j-1/2): depends on x only
U = Array{NTuple{5,Float32}}(undef, n, n)
for j in 1:n, i in 1:n
    u = -sinpi(2f0*yc(j)); v = sinpi(2f0*xc(i)); Bxc = 0.5f0*(bx[i,j]+bx[mod1(i+1,n),j]); Byc = 0.5f0*(by[i,j]+by[i,mod1(j+1,n)])
    E = P0/(γ-1) + 0.5f0*ρ0*(u*u+v*v) + 0.5f0*(Bxc*Bxc+Byc*Byc)
    U[i,j] = (ρ0, ρ0*u, ρ0*v, 0f0, E)
end
println("CT initial div·B (max) = ", divB(bx, by, dx, dy, n, n))
function run!(U, bx, by, s, dx, dy, n, tend)
    γ = s.γ; t = 0f0; nst = 0
    while t < tend && nst < 100000
        a = 0f0
        for j in 1:n, i in 1:n
            Bxc=0.5f0*(bx[i,j]+bx[mod1(i+1,n),j]); Byc=0.5f0*(by[i,j]+by[i,mod1(j+1,n)])
            W = ctprim(γ, U[i,j], Bxc, Byc)
            a = max(a, fastspeed_x(s, W), fastspeed_x(s, swapy(W)))
        end
        dt = min(0.4f0*dx/a, tend-t)
        ct_step!(U, bx, by, s, dx, dy, dt, n, n); t += dt; nst += 1
    end
    nst
end
nst = run!(U, bx, by, s, dx, dy, n, 0.5f0)
ρmin = minimum(U[i,j][1] for i in 1:n, j in 1:n); ρmax = maximum(U[i,j][1] for i in 1:n, j in 1:n)
println("CT t=0.5 ($(nst) steps): finite=", all(all(isfinite,U[i,j]) for i in 1:n,j in 1:n),
        " ρ∈($(round(ρmin,digits=3)),$(round(ρmax,digits=3)))  max|divB| = ", divB(bx, by, dx, dy, n, n))
