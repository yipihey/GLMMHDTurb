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

# face div·B for cell i = (upper face - lower face)/dx per dim; bxf[i]=lower-x face so upper=bxf[i+1].
function divB_max(bxf,byf,bzf,dx)
    db=(circshift(bxf,(-1,0,0)).-bxf .+ circshift(byf,(0,-1,0)).-byf .+ circshift(bzf,(0,0,-1)).-bzf)./dx
    Float64(maximum(abs.(db)))
end

end # module
