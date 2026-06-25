# 2D Constrained-Transport MHD backend (scalar, 1st-order, planar B). The magnetic field is
# face-staggered and advanced by the curl of an edge-centered EMF, giving div·B = 0 to machine
# precision (vs GLM cleaning's ~few %). Uses the GLMMHD physics with ch=0 as the ideal-MHD flux.
#
# Storage: bx[i,j] = Bx on the LEFT x-face of cell i; by[i,j] = By on the BOTTOM y-face of cell j.
# Cell-centered B (for the Riemann) is the face average. EMF Ez lives on cell corners:
#   ∂Bx/∂t = -∂Ez/∂y,  ∂By/∂t = +∂Ez/∂x   ⇒ discrete div·B preserved exactly.
#
# v0 is periodic, 1st-order, planar (Bz=0). The reserved @staggered/@emf CONTRACT seam (making CT
# systems user-definable like @fvsystem) is the remaining generalization; this proves the method.

mutable struct Grid2DCT{S<:FVSystem}
    sys::S                                  # GLMMHD with ch=0 (ideal MHD)
    rsol::Any
    U::Matrix{NTuple{5,Float32}}            # cell-centered (ρ,ρu,ρv,ρw,E)
    bx::Matrix{Float32}                     # Bx on x-faces
    by::Matrix{Float32}                     # By on y-faces
    nx::Int; ny::Int; dx::Float32; dy::Float32; cfl::Float32
end

function Grid2DCT(sys::FVSystem, U::Matrix{NTuple{5,Float32}}, bx::Matrix{Float32}, by::Matrix{Float32};
                  dx, dy, rsol = LLF(), cfl = 0.4f0)
    nx, ny = size(U)
    Grid2DCT{typeof(sys)}(sys, rsol, copy(U), copy(bx), copy(by), nx, ny, Float32(dx), Float32(dy), Float32(cfl))
end

@inline _ctswapy(t) = (t[1],t[3],t[2],t[4],t[5],t[7],t[6],t[8],t[9])
@inline function _ctprim(γ, U, Bxc, Byc)
    ρ,mx,my,mz,E = U; iρ = 1f0/ρ; u,v,w = mx*iρ, my*iρ, mz*iρ
    P = (γ-1)*(E - 0.5f0*ρ*(u*u+v*v+w*w) - 0.5f0*(Bxc*Bxc+Byc*Byc))
    (ρ,u,v,w,P,Bxc,Byc,0f0,0f0)
end
@inline _ctovx(W, B) = (W[1],W[2],W[3],W[4],W[5], B, W[7],0f0,0f0)
@inline _ctovy(W, B) = (W[1],W[2],W[3],W[4],W[5], W[6], B,0f0,0f0)

function step!(g::Grid2DCT, dt)
    s, rs, nx, ny, dx, dy = g.sys, g.rsol, g.nx, g.ny, g.dx, g.dy; γ = s.γ; U, bx, by = g.U, g.bx, g.by
    gp(i,n) = mod1(i,n)
    Bxc = [0.5f0*(bx[i,j]+bx[gp(i+1,nx),j]) for i in 1:nx, j in 1:ny]
    Byc = [0.5f0*(by[i,j]+by[i,gp(j+1,ny)]) for i in 1:nx, j in 1:ny]
    Fx = Matrix{NTuple{9,Float32}}(undef, nx, ny); Fy = similar(Fx)
    exf = Matrix{Float32}(undef, nx, ny); eyf = similar(exf)
    @inbounds for j in 1:ny, i in 1:nx
        iL = gp(i-1,nx)
        Fx[i,j] = riemann(rs, s, _ctovx(_ctprim(γ,U[iL,j],Bxc[iL,j],Byc[iL,j]), bx[i,j]),
                                 _ctovx(_ctprim(γ,U[i,j], Bxc[i,j], Byc[i,j]),  bx[i,j])); exf[i,j] = -Fx[i,j][7]
        jL = gp(j-1,ny)
        Fy[i,j] = _ctswapy(riemann(rs, s, _ctswapy(_ctovy(_ctprim(γ,U[i,jL],Bxc[i,jL],Byc[i,jL]), by[i,j])),
                                          _ctswapy(_ctovy(_ctprim(γ,U[i,j], Bxc[i,j], Byc[i,j]),  by[i,j])))); eyf[i,j] = Fy[i,j][6]
    end
    Ez = [0.25f0*(exf[i,j] + exf[i,gp(j-1,ny)] + eyf[i,j] + eyf[gp(i-1,nx),j]) for i in 1:nx, j in 1:ny]
    Un = similar(U)
    @inbounds for j in 1:ny, i in 1:nx
        fxr = Fx[gp(i+1,nx),j]; fyr = Fy[i,gp(j+1,ny)]
        Un[i,j] = ntuple(c -> U[i,j][c] - dt/dx*(fxr[c]-Fx[i,j][c]) - dt/dy*(fyr[c]-Fy[i,j][c]), 5)
    end
    @inbounds for j in 1:ny, i in 1:nx
        bx[i,j] -= dt/dy*(Ez[i,gp(j+1,ny)] - Ez[i,j])
        by[i,j] += dt/dx*(Ez[gp(i+1,nx),j] - Ez[i,j])
    end
    copyto!(U, Un)
    return g
end

function max_wavespeed(g::Grid2DCT)
    s = g.sys; γ = s.γ; nx, ny = g.nx, g.ny; a = 0f0; gp(i,n) = mod1(i,n)
    @inbounds for j in 1:ny, i in 1:nx
        Bxc = 0.5f0*(g.bx[i,j]+g.bx[gp(i+1,nx),j]); Byc = 0.5f0*(g.by[i,j]+g.by[i,gp(j+1,ny)])
        W = _ctprim(γ, g.U[i,j], Bxc, Byc)
        a = max(a, fastspeed_x(s, W), fastspeed_x(s, _ctswapy(W)))
    end
    return a
end

function evolve_ct!(g::Grid2DCT, tend; maxsteps::Int = 10^6)
    t = 0f0; tend = Float32(tend); n = 0
    while t < tend && n < maxsteps
        dt = min(g.cfl * min(g.dx, g.dy) / max_wavespeed(g), tend - t)
        step!(g, dt); t += dt; n += 1
    end
    return g
end

# max |div·B| over the staggered field (the CT diagnostic — stays at machine zero).
function divB_max(g::Grid2DCT)
    nx, ny = g.nx, g.ny; gp(i,n) = mod1(i,n); m = 0f0
    @inbounds for j in 1:ny, i in 1:nx
        m = max(m, abs((g.bx[gp(i+1,nx),j]-g.bx[i,j])/g.dx + (g.by[i,gp(j+1,ny)]-g.by[i,j])/g.dy))
    end
    return m
end
