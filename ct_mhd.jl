# ct_mhd.jl — LEAN constrained-transport MHD in the GLMMHDTurb prototype style.
#
# What transfers from the GLM light line-march (verbatim): cell-centered Godunov with PLM(MonCen)
# + transverse-free 1D-Hancock predictor + HLL, fp32, fast-math-friendly. What's NEW (the part that
# does NOT transfer): the induction step. B is FACE-staggered (bxf,byf,bzf), updated by the curl of
# edge-EMFs (Balsara-Spicer average of the Godunov magnetic fluxes) -> div·B = 0 to machine
# precision by construction. Normal B at a face is the single continuous face value (no Riemann on
# Bn). LEAN layout = 8 reals/cell (5 cell + 3 face), vs production CT's 22 and GLM's 9 -> fits 512³.
#
# State: U (N,N,N,5) = (rho,mx,my,mz,E_total);  bxf,byf,bzf (N,N,N) periodic face fields.
# Convention: bxf[i,j,k] = Bx on the lower-x face of cell (i,j,k) = face (i-1/2, j, k).
module CTMHD
using CUDA, Printf

const GAMMA = 5f0/3f0
const SMALLR = 1f-6
const SMALLP = 1f-6

@inline w(i,N) = mod1(i,N)                       # 1-based periodic wrap
@inline function moncen(dl,dr)
    dc=0.5f0*(dl+dr); s=ifelse(dc>=0f0,1f0,-1f0)
    ifelse(dl*dr<=0f0, 0f0, s*min(2f0*min(abs(dl),abs(dr)),abs(dc)))
end

# cell-centered primitive (rho,vx,vy,vz,p,Bx,By,Bz) from conserved + cell-centered B
@inline function cellprim(U,bccx,bccy,bccz,i,j,k,gamma)
    @inbounds begin
        rho=max(U[i,j,k,1],SMALLR); ir=1f0/rho
        vx=U[i,j,k,2]*ir; vy=U[i,j,k,3]*ir; vz=U[i,j,k,4]*ir
        bx=bccx[i,j,k]; by=bccy[i,j,k]; bz=bccz[i,j,k]
        p=max((gamma-1f0)*(U[i,j,k,5]-0.5f0*rho*(vx*vx+vy*vy+vz*vz)-0.5f0*(bx*bx+by*by+bz*bz)),SMALLP)
        (rho,vx,vy,vz,p,bx,by,bz)
    end
end
@inline psub(a::NTuple{8},b::NTuple{8},c::Float32)=ntuple(t->a[t]+c*b[t],8)
@inline pslope(L::NTuple{8},M::NTuple{8},R::NTuple{8})=ntuple(t->0.5f0*moncen(M[t]-L[t],R[t]-M[t]),8)

# ideal-MHD flux of prim q in direction d (1=x,2=y,3=z); F[Bn]=0 (CT handles normal B via EMF)
@inline function dflux(q::NTuple{8},d::Int)
    rho,vx,vy,vz,p,bx,by,bz=q
    b2=bx*bx+by*by+bz*bz; ptot=p+0.5f0*b2
    E=p/(GAMMA-1f0)+0.5f0*rho*(vx*vx+vy*vy+vz*vz)+0.5f0*b2
    vb=vx*bx+vy*by+vz*bz
    un = d==1 ? vx : d==2 ? vy : vz
    bn = d==1 ? bx : d==2 ? by : bz
    Fr=rho*un
    Fmx=rho*un*vx - bn*bx + (d==1 ? ptot : 0f0)
    Fmy=rho*un*vy - bn*by + (d==2 ? ptot : 0f0)
    Fmz=rho*un*vz - bn*bz + (d==3 ? ptot : 0f0)
    FE=(E+ptot)*un - bn*vb
    Fbx = un*bx - (d==1 ? un*bx : vx*bn)      # = un*bx - vx*bn for d!=1, 0 for d==1
    Fby = un*by - (d==2 ? un*by : vy*bn)
    Fbz = un*bz - (d==3 ? un*bz : vz*bn)
    (Fr,Fmx,Fmy,Fmz,FE,Fbx,Fby,Fbz)
end
@inline function fast_speed(q::NTuple{8},d::Int)
    rho=q[1]; bn = d==1 ? q[6] : d==2 ? q[7] : q[8]
    c2=GAMMA*q[5]/rho; b2=(q[6]*q[6]+q[7]*q[7]+q[8]*q[8])/rho; dd=0.5f0*(b2+c2)
    sqrt(dd+sqrt(max(dd*dd-c2*bn*bn/rho,0f0)))
end
@inline function toC(q::NTuple{8})
    rho,vx,vy,vz,p,bx,by,bz=q
    (rho,rho*vx,rho*vy,rho*vz, p/(GAMMA-1f0)+0.5f0*rho*(vx*vx+vy*vy+vz*vz)+0.5f0*(bx*bx+by*by+bz*bz), bx,by,bz)
end
# HLL flux (8-comp; F[Bn]≈0 since L,R share Bn). un=d.
@inline function hll(L::NTuple{8},R::NTuple{8},d::Int)
    un=q->(d==1 ? q[2] : d==2 ? q[3] : q[4])
    cfL=fast_speed(L,d); cfR=fast_speed(R,d)
    SL=min(min(un(L)-cfL,un(R)-cfR),0f0); SR=max(max(un(L)+cfL,un(R)+cfR),0f0)
    FL=dflux(L,d); FR=dflux(R,d); UL=toC(L); UR=toC(R); ih=1f0/(SR-SL)
    ntuple(t->(SR*FL[t]-SL*FR[t]+SL*SR*(UR[t]-UL[t]))*ih, 8)
end
# transverse-free 1D-Hancock half-step of the cell center in direction d
@inline function hanc1d(q::NTuple{8},s::NTuple{8},d::Int,dtdx::Float32)
    FL=dflux(psub(q,s,-0.5f0),d); FR=dflux(psub(q,s,0.5f0),d)
    U=toC(q); h=0.5f0*dtdx
    Uh=ntuple(t->U[t]-h*(FR[t]-FL[t]),8)
    rho=max(Uh[1],SMALLR); ir=1f0/rho
    vx=Uh[2]*ir; vy=Uh[3]*ir; vz=Uh[4]*ir; bx=Uh[6]; by=Uh[7]; bz=Uh[8]
    p=max((GAMMA-1f0)*(Uh[5]-0.5f0*rho*(vx*vx+vy*vy+vz*vz)-0.5f0*(bx*bx+by*by+bz*bz)),SMALLP)
    (rho,vx,vy,vz,p,bx,by,bz)
end

# --- kernels ---------------------------------------------------------------
function bcc_kernel!(bccx,bccy,bccz,bxf,byf,bzf,N)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=N*N*N
        @inbounds begin
            k=(i-1)÷(N*N)+1; j=((i-1)÷N)%N+1; ii=(i-1)%N+1
            bccx[ii,j,k]=0.5f0*(bxf[ii,j,k]+bxf[w(ii+1,N),j,k])
            bccy[ii,j,k]=0.5f0*(byf[ii,j,k]+byf[ii,w(j+1,N),k])
            bccz[ii,j,k]=0.5f0*(bzf[ii,j,k]+bzf[ii,j,w(k+1,N)])
        end
    end
    return
end

# flux at the LOWER-d face of each cell (face (i-1/2) for d=1, etc). bnf = face normal-B array.
function flux_kernel!(F,U,bccx,bccy,bccz,bnf, d::Int,dtdx::Float32,N)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=N*N*N
        @inbounds begin
            k=(i-1)÷(N*N)+1; j=((i-1)÷N)%N+1; ii=(i-1)%N+1
            o1=d==1 ? 1 : 0; o2=d==2 ? 1 : 0; o3=d==3 ? 1 : 0
            cp(a,b,c)=cellprim(U,bccx,bccy,bccz,a,b,c,GAMMA)
            m2=cp(w(ii-2o1,N),w(j-2o2,N),w(k-2o3,N))
            m1=cp(w(ii-o1,N), w(j-o2,N), w(k-o3,N))
            c0=cp(ii,j,k)
            p1=cp(w(ii+o1,N), w(j+o2,N), w(k+o3,N))
            sL=pslope(m2,m1,c0); sR=pslope(m1,c0,p1)
            mhL=hanc1d(m1,sL,d,dtdx); mhR=hanc1d(c0,sR,d,dtdx)
            L=psub(mhL,sL, 0.5f0)      # cell i-1 reconstructed to its +d edge
            R=psub(mhR,sR,-0.5f0)      # cell i   reconstructed to its -d edge
            bn=bnf[ii,j,k]
            L=(max(L[1],SMALLR),L[2],L[3],L[4],max(L[5],SMALLP), d==1 ? bn : L[6], d==2 ? bn : L[7], d==3 ? bn : L[8])
            R=(max(R[1],SMALLR),R[2],R[3],R[4],max(R[5],SMALLP), d==1 ? bn : R[6], d==2 ? bn : R[7], d==3 ? bn : R[8])
            f=hll(L,R,d)
            for t in 1:8; F[ii,j,k,t]=f[t]; end
        end
    end
    return
end

function update_hydro_kernel!(U,Fx,Fy,Fz,dtdx,N)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=N*N*N
        @inbounds begin
            k=(i-1)÷(N*N)+1; j=((i-1)÷N)%N+1; ii=(i-1)%N+1
            ip=w(ii+1,N); jp=w(j+1,N); kp=w(k+1,N)
            for t in 1:5
                U[ii,j,k,t]+=dtdx*((Fx[ii,j,k,t]-Fx[ip,j,k,t])+(Fy[ii,j,k,t]-Fy[ii,jp,k,t])+(Fz[ii,j,k,t]-Fz[ii,j,kp,t]))
            end
        end
    end
    return
end

# edge EMFs (Balsara-Spicer average of Godunov magnetic fluxes). component index in F: Bx=6,By=7,Bz=8.
function emf_kernel!(ex,ey,ez,Fx,Fy,Fz,N)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=N*N*N
        @inbounds begin
            k=(i-1)÷(N*N)+1; j=((i-1)÷N)%N+1; ii=(i-1)%N+1
            im=w(ii-1,N); jm=w(j-1,N); km=w(k-1,N)
            # ez at z-edge (i-1/2,j-1/2,k): Fy[Bx] (comp6) at (i,j),(i-1,j) ; Fx[By] (comp7) at (i,j),(i,j-1)
            ez[ii,j,k]=0.25f0*(Fy[ii,j,k,6]+Fy[im,j,k,6]-Fx[ii,j,k,7]-Fx[ii,jm,k,7])
            # ex at x-edge (i,j-1/2,k-1/2): Fz[By](7) at (i,j),(i,j-1) ; Fy[Bz](8) at (i,j),(i,j,k-1)
            ex[ii,j,k]=0.25f0*(Fz[ii,j,k,7]+Fz[ii,jm,k,7]-Fy[ii,j,k,8]-Fy[ii,j,km,8])
            # ey at y-edge (i-1/2,j,k-1/2): Fx[Bz](8) at (i,j,k),(i,j,k-1) ; Fz[Bx](6) at (i,j),(i-1,j)
            ey[ii,j,k]=0.25f0*(Fx[ii,j,k,8]+Fx[ii,j,km,8]-Fz[ii,j,k,6]-Fz[im,j,k,6])
        end
    end
    return
end

function faceB_kernel!(bxf,byf,bzf,ex,ey,ez,dtdx,N)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if i<=N*N*N
        @inbounds begin
            k=(i-1)÷(N*N)+1; j=((i-1)÷N)%N+1; ii=(i-1)%N+1
            ip=w(ii+1,N); jp=w(j+1,N); kp=w(k+1,N)
            # dBx/dt = -(dEz/dy - dEy/dz)
            bxf[ii,j,k]-=dtdx*((ez[ii,jp,k]-ez[ii,j,k])-(ey[ii,j,kp]-ey[ii,j,k]))
            # dBy/dt = -(dEx/dz - dEz/dx)
            byf[ii,j,k]-=dtdx*((ex[ii,j,kp]-ex[ii,j,k])-(ez[ip,j,k]-ez[ii,j,k]))
            # dBz/dt = -(dEy/dx - dEx/dy)
            bzf[ii,j,k]-=dtdx*((ey[ip,j,k]-ey[ii,j,k])-(ex[ii,jp,k]-ex[ii,j,k]))
        end
    end
    return
end

# one CT step (fixed dt). Scratch passed in to avoid realloc.
function step!(U,bxf,byf,bzf, bccx,bccy,bccz, Fx,Fy,Fz, ex,ey,ez, dt,dx,N)
    dtdx=dt/dx; nb=cld(N^3,256)
    @cuda threads=256 blocks=nb bcc_kernel!(bccx,bccy,bccz,bxf,byf,bzf,N)
    @cuda threads=256 blocks=nb flux_kernel!(Fx,U,bccx,bccy,bccz,bxf,1,dtdx,N)
    @cuda threads=256 blocks=nb flux_kernel!(Fy,U,bccx,bccy,bccz,byf,2,dtdx,N)
    @cuda threads=256 blocks=nb flux_kernel!(Fz,U,bccx,bccy,bccz,bzf,3,dtdx,N)
    @cuda threads=256 blocks=nb emf_kernel!(ex,ey,ez,Fx,Fy,Fz,N)
    @cuda threads=256 blocks=nb update_hydro_kernel!(U,Fx,Fy,Fz,dtdx,N)
    @cuda threads=256 blocks=nb faceB_kernel!(bxf,byf,bzf,ex,ey,ez,dtdx,N)
    return
end

scratch(N)=(CUDA.zeros(Float32,N,N,N),CUDA.zeros(Float32,N,N,N),CUDA.zeros(Float32,N,N,N),
            CUDA.zeros(Float32,N,N,N,8),CUDA.zeros(Float32,N,N,N,8),CUDA.zeros(Float32,N,N,N,8),
            CUDA.zeros(Float32,N,N,N),CUDA.zeros(Float32,N,N,N),CUDA.zeros(Float32,N,N,N))

# ============================================================================
# FUSED STAGED f16 CT kernel (CUDA.jl port of spike_ct2.cu) — CORRECT but a DOCUMENTED NEGATIVE.
# One kernel replaces the 7-kernel global-flux path. 3D tile: load prim+faceB f16 tile -> phase A
# flux once -> shared -> phase B hydro update + edge-EMF + face-B. f32 update base (div·B machine-zero,
# validated 7.2e-5 = the 7-kernel; fields match to ~2e-4). Double-buffered so halo reads don't race.
#
# RESULT (A6000, gradient IC @480): SLOWER than the 7-kernel — best 194 (6x6x6, 1 block) vs 206.
# Unlike the .cu (where the fused tile WON, 1206 vs the multi-kernel), in CUDA.jl it loses: the shared
# tile caps occupancy at 1 block (the simple 7-kernel runs many blocks), the NG=3 halo wastes ~80-94%
# of the tile, and Julia's codegen at 1 block can't overcome it the way nvcc's does. ⇒ for Julia CT the
# fused tile is a dead end (the same occupancy/halo trap the .cu spike_hys.cu hit for hydro). The 7-
# kernel (~206, traffic-bound but high-occupancy) is the Julia CT ceiling for the 3D-tile family.
# Kept here as a validated-correct negative; the only Julia path with more headroom is a z-march
# (no halo, full occupancy) — high codegen risk, future work.
# ============================================================================
@inline _cm(a,b,c)=mc(a,b,c)
function ct_fused!(::Val{OX},::Val{OY},::Val{OZ}, U,bxf,byf,bzf, oU,obxf,obyf,obzf,
                   bccx,bccy,bccz, dtdx::Float32, N::Int) where {OX,OY,OZ}
    NG=3; TX=OX+2NG; TY=OY+2NG; TZ=OZ+2NG; TT=TX*TY*TZ
    FX=OX+2; FY=OY+2; FZ=OZ+2; FT=FX*FY*FZ
    S  = CuDynamicSharedArray(Float16, 11*TT)
    FL = CuDynamicSharedArray(Float16, 21*FT, 11*TT*sizeof(Float16))
    tid=threadIdx().x
    tx=(tid-1)%OX+1; ty=((tid-1)÷OX)%OY+1; tz=(tid-1)÷(OX*OY)+1
    x0=(blockIdx().x-1)*OX; y0=(blockIdx().y-1)*OY; z0=(blockIdx().z-1)*OZ
    @inbounds begin
        # --- load tile: 5 conserved + 3 Bcc + 3 faceB, f16 ---
        c=tid
        while c<=TT
            lx=(c-1)%TX+1; ly=((c-1)÷TX)%TY+1; lz=(c-1)÷(TX*TY)+1
            gi=w(x0-NG+lx,N); gj=w(y0-NG+ly,N); gk=w(z0-NG+lz,N)
            S[c]=Float16(U[gi,gj,gk,1]); S[TT+c]=Float16(U[gi,gj,gk,2]); S[2TT+c]=Float16(U[gi,gj,gk,3])
            S[3TT+c]=Float16(U[gi,gj,gk,4]); S[4TT+c]=Float16(U[gi,gj,gk,5])
            S[5TT+c]=Float16(bccx[gi,gj,gk]); S[6TT+c]=Float16(bccy[gi,gj,gk]); S[7TT+c]=Float16(bccz[gi,gj,gk])
            S[8TT+c]=Float16(bxf[gi,gj,gk]); S[9TT+c]=Float16(byf[gi,gj,gk]); S[10TT+c]=Float16(bzf[gi,gj,gk])
            c+=OX*OY*OZ
        end
        sync_threads()
        cidx(lx,ly,lz)=lx+TX*(ly-1)+TX*TY*(lz-1)
        tprim(lx,ly,lz)=begin
            cc=cidx(lx,ly,lz); rho=max(Float32(S[cc]),SMALLR); ir=1f0/rho
            mx=Float32(S[TT+cc]); my=Float32(S[2TT+cc]); mz=Float32(S[3TT+cc]); E=Float32(S[4TT+cc])
            bx=Float32(S[5TT+cc]); by=Float32(S[6TT+cc]); bz=Float32(S[7TT+cc])
            vx=mx*ir; vy=my*ir; vz=mz*ir
            (rho,vx,vy,vz,max((GAMMA-1f0)*(E-0.5f0*rho*(vx*vx+vy*vy+vz*vz)-0.5f0*(bx*bx+by*by+bz*bz)),SMALLP),bx,by,bz)
        end
        ffl(d,lx,ly,lz)=begin
            o1=d==1 ? 1 : 0; o2=d==2 ? 1 : 0; o3=d==3 ? 1 : 0
            m2=tprim(lx-2o1,ly-2o2,lz-2o3); m1=tprim(lx-o1,ly-o2,lz-o3); c0=tprim(lx,ly,lz); p1=tprim(lx+o1,ly+o2,lz+o3)
            sL=pslope(m2,m1,c0); sR=pslope(m1,c0,p1)
            mhL=hanc1d(m1,sL,d,dtdx); mhR=hanc1d(c0,sR,d,dtdx)
            L=psub(mhL,sL,0.5f0); R=psub(mhR,sR,-0.5f0)
            bn=Float32(S[(7+d)*TT+cidx(lx,ly,lz)])
            L=(max(L[1],SMALLR),L[2],L[3],L[4],max(L[5],SMALLP), d==1 ? bn : L[6], d==2 ? bn : L[7], d==3 ? bn : L[8])
            R=(max(R[1],SMALLR),R[2],R[3],R[4],max(R[5],SMALLP), d==1 ? bn : R[6], d==2 ? bn : R[7], d==3 ? bn : R[8])
            hll(L,R,d)
        end
        # --- phase A: each lower-face flux ONCE -> shared (comp 1-5 hydro, 6=m1, 7=m2) ---
        c=tid
        while c<=FT
            fi=(c-1)%FX; fj=((c-1)÷FX)%FY; fk=(c-1)÷(FX*FY); lx=fi+NG; ly=fj+NG; lz=fk+NG
            for d in 1:3
                f=ffl(d,lx,ly,lz)
                m1 = d==1 ? f[7] : f[6]
                m2 = d==3 ? f[7] : f[8]
                base=(d-1)*FT+c
                FL[base]=Float16(f[1]); FL[3FT+base]=Float16(f[2]); FL[6FT+base]=Float16(f[3])
                FL[9FT+base]=Float16(f[4]); FL[12FT+base]=Float16(f[5]); FL[15FT+base]=Float16(m1); FL[18FT+base]=Float16(m2)
            end
            c+=OX*OY*OZ
        end
        sync_threads()
        FF(comp,d,lx,ly,lz)=Float32(FL[((comp-1)*3+(d-1))*FT + (lx-NG)+FX*(ly-NG)+FX*FY*(lz-NG)+1])
        # --- phase B: owned-cell hydro update + EMF + face-B (f32 base from global) ---
        li=tx+NG; lj=ty+NG; lk=tz+NG; gi=x0+tx; gj=y0+ty; gk=z0+tz
        for t in 1:5
            div=(FF(t,1,li,lj,lk)-FF(t,1,li+1,lj,lk))+(FF(t,2,li,lj,lk)-FF(t,2,li,lj+1,lk))+(FF(t,3,li,lj,lk)-FF(t,3,li,lj,lk+1))
            oU[gi,gj,gk,t]=U[gi,gj,gk,t]+dtdx*div
        end
        EZ(a,b,cc)=0.25f0*(FF(6,2,a,b,cc)+FF(6,2,a-1,b,cc)-FF(6,1,a,b,cc)-FF(6,1,a,b-1,cc))
        EX(a,b,cc)=0.25f0*(FF(7,3,a,b,cc)+FF(7,3,a,b-1,cc)-FF(7,2,a,b,cc)-FF(7,2,a,b,cc-1))
        EY(a,b,cc)=0.25f0*(FF(7,1,a,b,cc)+FF(7,1,a,b,cc-1)-FF(6,3,a,b,cc)-FF(6,3,a-1,b,cc))
        obxf[gi,gj,gk]=bxf[gi,gj,gk]-dtdx*((EZ(li,lj+1,lk)-EZ(li,lj,lk))-(EY(li,lj,lk+1)-EY(li,lj,lk)))
        obyf[gi,gj,gk]=byf[gi,gj,gk]-dtdx*((EX(li,lj,lk+1)-EX(li,lj,lk))-(EZ(li+1,lj,lk)-EZ(li,lj,lk)))
        obzf[gi,gj,gk]=bzf[gi,gj,gk]-dtdx*((EY(li+1,lj,lk)-EY(li,lj,lk))-(EX(li,lj+1,lk)-EX(li,lj,lk)))
    end
    return
end

# fused-CT step (double-buffered). Returns the (possibly swapped) state arrays.
function step_fused!(U,bxf,byf,bzf, oU,obxf,obyf,obzf, bccx,bccy,bccz, dt,dx,N; ox=4,oy=4,oz=4)
    dtdx=dt/dx
    @cuda threads=256 blocks=cld(N^3,256) bcc_kernel!(bccx,bccy,bccz,bxf,byf,bzf,N)
    TT=(ox+6)*(oy+6)*(oz+6); FT=(ox+2)*(oy+2)*(oz+2); shmem=(11*TT+21*FT)*sizeof(Float16)
    k=@cuda launch=false ct_fused!(Val(ox),Val(oy),Val(oz),U,bxf,byf,bzf,oU,obxf,obyf,obzf,bccx,bccy,bccz,Float32(dtdx),N)
    shmem>48*1024 && CUDA.cuFuncSetAttribute(k.fun, CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem)
    k(Val(ox),Val(oy),Val(oz),U,bxf,byf,bzf,oU,obxf,obyf,obzf,bccx,bccy,bccz,Float32(dtdx),N;
      threads=ox*oy*oz, blocks=(N÷ox,N÷oy,N÷oz), shmem=shmem)
    return oU,obxf,obyf,obzf,U,bxf,byf,bzf   # swapped
end

# face div·B for cell i = (upper face - lower face)/dx per dim; bxf[i]=lower-x face so upper=bxf[i+1].
function divB_max(bxf,byf,bzf,dx)
    db=(circshift(bxf,(-1,0,0)).-bxf .+ circshift(byf,(0,-1,0)).-byf .+ circshift(bzf,(0,0,-1)).-bzf)./dx
    Float64(maximum(abs.(db)))
end

end # module
