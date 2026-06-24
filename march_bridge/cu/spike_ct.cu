// spike_ct.cu — FUSED constrained-transport MHD, single tiled kernel (the .cu of GLMMHDTurb/ct_mhd.jl).
//
// Lean CT: 8 reals/cell global (5 cell-centered conserved + 3 face-staggered B). One fused kernel
// over a 3-D shared tile: load prim(Bcc)+faceB once, then each owned thread (a) updates the 5 hydro
// conserved from its 6 face HLL fluxes and (b) updates its 3 owned face-B from curl of the 4 edge-
// EMFs around each face — recomputing the Godunov magnetic fluxes from the tile. The recompute is
// deterministic from the same global data, so two blocks sharing an edge get the SAME EMF → div·B
// preserved at block seams for free (no shared flux store, no cross-block sync).
//
// Cell-centered Godunov = the GLM light-march machinery (PLM MonCen + transverse-free 1D-Hancock +
// HLL); face normal-B is the single continuous face value (no Riemann on Bn). Build:
//   nvcc -O3 -arch=sm_86 --use_fast_math -DAS_LIB -DNX=.. --shared -Xcompiler -fPIC -o libct.so spike_ct.cu
#include <cstdio>
#include <cuda_fp16.h>

#ifndef NX
#define NX 256
#endif
#ifndef NY
#define NY 256
#endif
#ifndef NZ
#define NZ 256
#endif
#ifndef OX
#define OX 8
#endif
#ifndef OY
#define OY 8
#endif
#ifndef OZ
#define OZ 2
#endif
#define NG 2
#define TX (OX+2*NG)
#define TY (OY+2*NG)
#define TZ (OZ+2*NG)
#define THREADS (OX*OY*OZ)
#ifndef TILET
#define TILET float            // tile precision (float for div·B debugging; __half for speed)
#endif

__constant__ float GAMMA=1.6666667f, DTDX=0.02f, SMALLR=1e-6f, SMALLP=1e-6f;

__device__ __forceinline__ int wrap(int i,int N){ int m=i%N; return m<0?m+N:m; }
__device__ __forceinline__ size_t gidx(int i,int j,int k){ return (size_t)i+(size_t)NX*((size_t)j+(size_t)NY*k); }

struct P { float r,u,v,w,p,bx,by,bz; };   // primitive; bx,by,bz = cell-centered (Bcc) unless overridden
__device__ __forceinline__ float mc(float a,float b,float c){
    float dl=b-a,dr=c-b,dc=0.5f*(c-a);
    if(dl*dr<=0.f) return 0.f;
    float s=dl>0.f?1.f:-1.f; return s*fminf(fabsf(dc),fminf(2.f*fabsf(dl),2.f*fabsf(dr)));
}
__device__ __forceinline__ P psub(P q,P s,float a){ return {q.r+a*s.r,q.u+a*s.u,q.v+a*s.v,q.w+a*s.w,q.p+a*s.p,q.bx+a*s.bx,q.by+a*s.by,q.bz+a*s.bz}; }
__device__ __forceinline__ P slope(P a,P b,P c){ return {mc(a.r,b.r,c.r),mc(a.u,b.u,c.u),mc(a.v,b.v,c.v),mc(a.w,b.w,c.w),mc(a.p,b.p,c.p),mc(a.bx,b.bx,c.bx),mc(a.by,b.by,c.by),mc(a.bz,b.bz,c.bz)}; }
// ideal-MHD flux comp in dir d (0=x,1=y,2=z); returns the 8 components (Bn flux=0)
struct F8 { float r,mx,my,mz,E,bx,by,bz; };
__device__ __forceinline__ F8 dflux(P q,int d){
    float b2=q.bx*q.bx+q.by*q.by+q.bz*q.bz, ptot=q.p+0.5f*b2;
    float E=q.p/(GAMMA-1.f)+0.5f*q.r*(q.u*q.u+q.v*q.v+q.w*q.w)+0.5f*b2;
    float vb=q.u*q.bx+q.v*q.by+q.w*q.bz;
    float un=d==0?q.u:d==1?q.v:q.w, bn=d==0?q.bx:d==1?q.by:q.bz;
    F8 f;
    f.r=q.r*un;
    f.mx=q.r*un*q.u-bn*q.bx+(d==0?ptot:0.f);
    f.my=q.r*un*q.v-bn*q.by+(d==1?ptot:0.f);
    f.mz=q.r*un*q.w-bn*q.bz+(d==2?ptot:0.f);
    f.E=(E+ptot)*un-bn*vb;
    f.bx=(d==0)?0.f:un*q.bx-q.u*bn;
    f.by=(d==1)?0.f:un*q.by-q.v*bn;
    f.bz=(d==2)?0.f:un*q.bz-q.w*bn;
    return f;
}
__device__ __forceinline__ F8 toC(P q){
    return {q.r,q.r*q.u,q.r*q.v,q.r*q.w, q.p/(GAMMA-1.f)+0.5f*q.r*(q.u*q.u+q.v*q.v+q.w*q.w)+0.5f*(q.bx*q.bx+q.by*q.by+q.bz*q.bz), q.bx,q.by,q.bz};
}
__device__ __forceinline__ float fast_speed(P q,int d){
    float bn=d==0?q.bx:d==1?q.by:q.bz;
    float c2=GAMMA*q.p/q.r,b2=(q.bx*q.bx+q.by*q.by+q.bz*q.bz)/q.r,dd=0.5f*(b2+c2);
    return sqrtf(dd+sqrtf(fmaxf(dd*dd-c2*bn*bn/q.r,0.f)));
}
__device__ __forceinline__ F8 hll(P L,P R,int d){
    float uL=d==0?L.u:d==1?L.v:L.w, uR=d==0?R.u:d==1?R.v:R.w;
    float cL=fast_speed(L,d),cR=fast_speed(R,d);
    float SL=fminf(fminf(uL-cL,uR-cR),0.f), SR=fmaxf(fmaxf(uL+cL,uR+cR),0.f);
    F8 FL=dflux(L,d),FR=dflux(R,d),UL=toC(L),UR=toC(R),F; float ih=1.f/(SR-SL);
    F.r =(SR*FL.r -SL*FR.r +SL*SR*(UR.r -UL.r ))*ih;
    F.mx=(SR*FL.mx-SL*FR.mx+SL*SR*(UR.mx-UL.mx))*ih;
    F.my=(SR*FL.my-SL*FR.my+SL*SR*(UR.my-UL.my))*ih;
    F.mz=(SR*FL.mz-SL*FR.mz+SL*SR*(UR.mz-UL.mz))*ih;
    F.E =(SR*FL.E -SL*FR.E +SL*SR*(UR.E -UL.E ))*ih;
    F.bx=(SR*FL.bx-SL*FR.bx+SL*SR*(UR.bx-UL.bx))*ih;
    F.by=(SR*FL.by-SL*FR.by+SL*SR*(UR.by-UL.by))*ih;
    F.bz=(SR*FL.bz-SL*FR.bz+SL*SR*(UR.bz-UL.bz))*ih;
    return F;
}
// transverse-free 1D Hancock half-step of a cell center in dir d
__device__ __forceinline__ P hanc1d(P q,P s,int d){
    F8 FL=dflux(psub(q,s,-0.5f),d),FR=dflux(psub(q,s,0.5f),d); F8 U=toC(q); float h=0.5f*DTDX;
    float r=fmaxf(U.r-h*(FR.r-FL.r),SMALLR), ir=1.f/r;
    float mx=U.mx-h*(FR.mx-FL.mx),my=U.my-h*(FR.my-FL.my),mz=U.mz-h*(FR.mz-FL.mz),E=U.E-h*(FR.E-FL.E);
    float bx=U.bx-h*(FR.bx-FL.bx),by=U.by-h*(FR.by-FL.by),bz=U.bz-h*(FR.bz-FL.bz);
    float vx=mx*ir,vy=my*ir,vz=mz*ir;
    float p=fmaxf((GAMMA-1.f)*(E-0.5f*r*(vx*vx+vy*vy+vz*vz)-0.5f*(bx*bx+by*by+bz*bz)),SMALLP);
    return {r,vx,vy,vz,p,bx,by,bz};
}

// ---- precompute Bcc (cell-centered) from face B ----
struct Ptrs { float* U[5]; float* bx; float* by; float* bz; float* cx; float* cy; float* cz; };
__global__ void bcc_kernel(Ptrs q){
    size_t n=(size_t)NX*NY*NZ,i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    int k=i/((size_t)NX*NY), j=(i/NX)%NY, ii=i%NX;
    q.cx[i]=0.5f*(q.bx[i]+q.bx[gidx(wrap(ii+1,NX),j,k)]);
    q.cy[i]=0.5f*(q.by[i]+q.by[gidx(ii,wrap(j+1,NY),k)]);
    q.cz[i]=0.5f*(q.bz[i]+q.bz[gidx(ii,j,wrap(k+1,NZ))]);
}

// shared tile: 11 values/cell: rho,mx,my,mz,E, Bccx,Bccy,Bccz, bxf,byf,bzf
#define SI(lx,ly,lz) ((lx)+TX*((ly)+TY*(lz)))
#define TV(S,v,lx,ly,lz) ((float)(S)[(v)*TX*TY*TZ + SI(lx,ly,lz)])
// cell prim from tile (uses Bcc slots 5,6,7)
__device__ __forceinline__ P cp(TILET*S,int lx,int ly,int lz){
    float r=fmaxf(TV(S,0,lx,ly,lz),SMALLR), ir=1.f/r;
    float mx=TV(S,1,lx,ly,lz),my=TV(S,2,lx,ly,lz),mz=TV(S,3,lx,ly,lz),E=TV(S,4,lx,ly,lz);
    float bx=TV(S,5,lx,ly,lz),by=TV(S,6,lx,ly,lz),bz=TV(S,7,lx,ly,lz);
    float vx=mx*ir,vy=my*ir,vz=mz*ir;
    float p=fmaxf((GAMMA-1.f)*(E-0.5f*r*(vx*vx+vy*vy+vz*vz)-0.5f*(bx*bx+by*by+bz*bz)),SMALLP);
    return {r,vx,vy,vz,p,bx,by,bz};
}
// HLL flux at the lower-d face of tile cell (lx,ly,lz); Bn from face value
__device__ __forceinline__ F8 faceflux(TILET*S,int d,int lx,int ly,int lz){
    int ox=d==0,oy=d==1,oz=d==2;
    P m2=cp(S,lx-2*ox,ly-2*oy,lz-2*oz), m1=cp(S,lx-ox,ly-oy,lz-oz), c0=cp(S,lx,ly,lz), p1=cp(S,lx+ox,ly+oy,lz+oz);
    P sL=slope(m2,m1,c0), sR=slope(m1,c0,p1);
    P L=psub(hanc1d(m1,sL,d),sL, 0.5f);    // cell (lx-1) → +d edge
    P R=psub(hanc1d(c0,sR,d),sR,-0.5f);    // cell (lx)   → -d edge
    float bn=TV(S,8+d,lx,ly,lz);
    L.r=fmaxf(L.r,SMALLR);L.p=fmaxf(L.p,SMALLP);R.r=fmaxf(R.r,SMALLR);R.p=fmaxf(R.p,SMALLP);
    if(d==0){L.bx=bn;R.bx=bn;} else if(d==1){L.by=bn;R.by=bn;} else {L.bz=bn;R.bz=bn;}
    return hll(L,R,d);
}
// edge EMFs (Balsara-Spicer) at tile-edge indices, recomputed from face fluxes
// ez at z-edge (lx-½,ly-½,lz): Fy[Bx](lx,ly)&(lx-1,ly) ; Fx[By](lx,ly)&(lx,ly-1)
__device__ __forceinline__ float emfz(TILET*S,int lx,int ly,int lz){
    return 0.25f*( faceflux(S,1,lx,ly,lz).bx + faceflux(S,1,lx-1,ly,lz).bx
                 - faceflux(S,0,lx,ly,lz).by - faceflux(S,0,lx,ly-1,lz).by );
}
// ex at x-edge (lx,ly-½,lz-½): Fz[By](lx,ly)&(lx,ly-1) ; Fy[Bz](lx,ly)&(lx,ly,lz-1)
__device__ __forceinline__ float emfx(TILET*S,int lx,int ly,int lz){
    return 0.25f*( faceflux(S,2,lx,ly,lz).by + faceflux(S,2,lx,ly-1,lz).by
                 - faceflux(S,1,lx,ly,lz).bz - faceflux(S,1,lx,ly,lz-1).bz );
}
// ey at y-edge (lx-½,ly,lz-½): Fx[Bz](lx,ly,lz)&(lx,ly,lz-1) ; Fz[Bx](lx,ly)&(lx-1,ly)
__device__ __forceinline__ float emfy(TILET*S,int lx,int ly,int lz){
    return 0.25f*( faceflux(S,0,lx,ly,lz).bz + faceflux(S,0,lx,ly,lz-1).bz
                 - faceflux(S,2,lx,ly,lz).bx - faceflux(S,2,lx-1,ly,lz).bx );
}

__global__ void __launch_bounds__(THREADS) ct_step(Ptrs q, Ptrs o){
    extern __shared__ TILET sh[];
    TILET* S=sh;
    const int tid=threadIdx.x;
    const int tx=tid%OX, ty=(tid/OX)%OY, tz=tid/(OX*OY);
    const int x0=blockIdx.x*OX, y0=blockIdx.y*OY, z0=blockIdx.z*OZ;
    // cooperative load of the (TX,TY,TZ) tile, 11 values
    for(int c=tid;c<TX*TY*TZ;c+=THREADS){
        int lx=c%TX, ly=(c/TX)%TY, lz=c/(TX*TY);
        int gi=wrap(x0-NG+lx,NX), gj=wrap(y0-NG+ly,NY), gk=wrap(z0-NG+lz,NZ);
        size_t g=gidx(gi,gj,gk);
        S[0*TX*TY*TZ+c]=(TILET)q.U[0][g]; S[1*TX*TY*TZ+c]=(TILET)q.U[1][g];
        S[2*TX*TY*TZ+c]=(TILET)q.U[2][g]; S[3*TX*TY*TZ+c]=(TILET)q.U[3][g]; S[4*TX*TY*TZ+c]=(TILET)q.U[4][g];
        S[5*TX*TY*TZ+c]=(TILET)q.cx[g];   S[6*TX*TY*TZ+c]=(TILET)q.cy[g];   S[7*TX*TY*TZ+c]=(TILET)q.cz[g];
        S[8*TX*TY*TZ+c]=(TILET)q.bx[g];   S[9*TX*TY*TZ+c]=(TILET)q.by[g];   S[10*TX*TY*TZ+c]=(TILET)q.bz[g];
    }
    __syncthreads();
    int li=tx+NG, lj=ty+NG, lk=tz+NG;
    size_t g=gidx(x0+tx,y0+ty,z0+tz);
    // --- hydro update: U += DTDX*(Flo - Fhi) per dir ---
    F8 fxl=faceflux(S,0,li,lj,lk),  fxh=faceflux(S,0,li+1,lj,lk);
    F8 fyl=faceflux(S,1,li,lj,lk),  fyh=faceflux(S,1,li,lj+1,lk);
    F8 fzl=faceflux(S,2,li,lj,lk),  fzh=faceflux(S,2,li,lj,lk+1);
    float dr =(fxl.r -fxh.r )+(fyl.r -fyh.r )+(fzl.r -fzh.r );
    float dmx=(fxl.mx-fxh.mx)+(fyl.mx-fyh.mx)+(fzl.mx-fzh.mx);
    float dmy=(fxl.my-fxh.my)+(fyl.my-fyh.my)+(fzl.my-fzh.my);
    float dmz=(fxl.mz-fxh.mz)+(fyl.mz-fyh.mz)+(fzl.mz-fzh.mz);
    float dE =(fxl.E -fxh.E )+(fyl.E -fyh.E )+(fzl.E -fzh.E );
    o.U[0][g]=TV(S,0,li,lj,lk)+DTDX*dr;  o.U[1][g]=TV(S,1,li,lj,lk)+DTDX*dmx;
    o.U[2][g]=TV(S,2,li,lj,lk)+DTDX*dmy; o.U[3][g]=TV(S,3,li,lj,lk)+DTDX*dmz; o.U[4][g]=TV(S,4,li,lj,lk)+DTDX*dE;
    // --- face-B update via curl of edge EMFs ---
    // dBx/dt = -(dEz/dy - dEy/dz)
    o.bx[g]=TV(S,8,li,lj,lk) - DTDX*((emfz(S,li,lj+1,lk)-emfz(S,li,lj,lk)) - (emfy(S,li,lj,lk+1)-emfy(S,li,lj,lk)));
    // dBy/dt = -(dEx/dz - dEz/dx)
    o.by[g]=TV(S,9,li,lj,lk) - DTDX*((emfx(S,li,lj,lk+1)-emfx(S,li,lj,lk)) - (emfz(S,li+1,lj,lk)-emfz(S,li,lj,lk)));
    // dBz/dt = -(dEy/dx - dEx/dy)
    o.bz[g]=TV(S,10,li,lj,lk) - DTDX*((emfy(S,li+1,lj,lk)-emfy(S,li,lj,lk)) - (emfx(S,li,lj+1,lk)-emfx(S,li,lj,lk)));
}

#ifdef AS_LIB
extern "C" {
int march_nv(){ return 8; }
int march_nx(){ return NX; }
void march_set_dtdx(float v){ cudaMemcpyToSymbol(DTDX,&v,sizeof(float)); }
void march_set_gamma(float g){ cudaMemcpyToSymbol(GAMMA,&g,sizeof(float)); }
// qv/ov: 8 device planes [U0..U4, bxf,byf,bzf]; cc: 3 scratch planes [Bccx,Bccy,Bccz]
double ct_run(float* const* qv, float* const* ov, float* const* cc, int nsteps){
    size_t n=(size_t)NX*NY*NZ;
    Ptrs Q,O;
    for(int v=0;v<5;v++){Q.U[v]=qv[v];O.U[v]=ov[v];}
    Q.bx=qv[5];Q.by=qv[6];Q.bz=qv[7]; O.bx=ov[5];O.by=ov[6];O.bz=ov[7];
    Q.cx=cc[0];Q.cy=cc[1];Q.cz=cc[2]; O.cx=cc[0];O.cy=cc[1];O.cz=cc[2];
    dim3 grid(NX/OX,NY/OY,NZ/OZ);
    size_t shmem=sizeof(TILET)*11*TX*TY*TZ;
    cudaFuncSetAttribute(ct_step, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    cudaEvent_t t0,t1; cudaEventCreate(&t0);cudaEventCreate(&t1); cudaEventRecord(t0);
    for(int s=0;s<nsteps;s++){
        bcc_kernel<<<(n+255)/256,256>>>(Q);
        ct_step<<<grid,THREADS,shmem>>>(Q,O);
        Ptrs t=Q;Q=O;O=t;
    }
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    float ms=0;cudaEventElapsedTime(&ms,t0,t1);
    if(Q.U[0]!=qv[0]){ for(int v=0;v<5;v++)cudaMemcpy(qv[v],Q.U[v],n*4,cudaMemcpyDeviceToDevice);
        cudaMemcpy(qv[5],Q.bx,n*4,cudaMemcpyDeviceToDevice);cudaMemcpy(qv[6],Q.by,n*4,cudaMemcpyDeviceToDevice);cudaMemcpy(qv[7],Q.bz,n*4,cudaMemcpyDeviceToDevice);}
    return (double)ms;
}
int ct_regs(){ cudaFuncAttributes fa; cudaFuncGetAttributes(&fa,ct_step); return fa.numRegs; }
}
#endif

#ifndef AS_LIB
int main(){ printf("spike_ct: build with -DAS_LIB for the Julia bridge\n"); return 0; }
#endif
