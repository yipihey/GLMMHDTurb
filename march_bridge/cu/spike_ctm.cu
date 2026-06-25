// spike_ctm.cu — 2.5D z-STREAMING CT MARCH (attempted; WIP/negative). See BENCHMARKS.
//
// Idea: each block owns a full z-column (no z block-seam) -> stream z through a 5-plane prim ring
// + 3-plane mag-flux ring; inline hydro, face-B updated with lag 2. Shared is small (22KB @OX=8 ->
// 4-block potential), the structural hope for beating the 3D-tile ct3 (1501, 2 blocks).
//
// BLOCKER (why this NaNs / is not the win): the PERIODIC z-wrap breaks the forward pipeline. Unlike
// the GLM march (no cross-plane flux coupling -> periodicity was trivial wrapped loads), CT's EMF
// couples planes: plane 0's face-B needs magflux(NZ-1) (computed LAST in a forward sweep) and plane
// NZ-1 needs magflux(0) (already evicted from the 3-plane ring). BOTH boundary planes have wrap
// dependencies the small ring cannot hold -> needs persistent boundary-plane storage or a 2nd sweep.
// Plus the lag forces mag and hydro fluxes to be computed at different steps (recompute redundancy,
// the ct4 trap), and good thread counts (256) need a bigger tile -> back to ~1 block. Net: the 2.5D
// streaming win that made GLM fast does NOT transfer to CT. ct3 (3D tile, 1501 Mcell/s) stands.
//
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
#define NG 3
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

// ===== 2.5D z-STREAMING CT MARCH: each block owns a full z-column (no z-seam -> z div·B auto).
// prim ring (5 planes) + mag-flux ring (3 planes) in f16; inline hydro; face-B updated with lag 2. =====
#undef THREADS
#define THREADS (OX*OY)
#define PT (TX*TY)
#define R5(p) (((p)%5+5)%5)
#define R3(p) (((p)%3+3)%3)
#define PRV(PR,v,s,lx,ly) ((float)(PR)[(v)*5*PT + (s)*PT + (lx)+TX*(ly)])
#define MRV(MR,c,kp,lx,ly) ((float)(MR)[(c)*3*PT + R3(kp)*PT + (lx)+TX*(ly)])
__device__ __forceinline__ P cpm(__half*PR,int s,int lx,int ly){
    float r=fmaxf(PRV(PR,0,s,lx,ly),SMALLR),ir=1.f/r;
    float mx=PRV(PR,1,s,lx,ly),my=PRV(PR,2,s,lx,ly),mz=PRV(PR,3,s,lx,ly),E=PRV(PR,4,s,lx,ly);
    float bx=PRV(PR,5,s,lx,ly),by=PRV(PR,6,s,lx,ly),bz=PRV(PR,7,s,lx,ly);
    float vx=mx*ir,vy=my*ir,vz=mz*ir;
    float p=fmaxf((GAMMA-1.f)*(E-0.5f*r*(vx*vx+vy*vy+vz*vz)-0.5f*(bx*bx+by*by+bz*bz)),SMALLP);
    return {r,vx,vy,vz,p,bx,by,bz};
}
__device__ __forceinline__ F8 fflm(__half*PR,Ptrs q,int d,int kp,int lx,int ly,int x0,int y0){
    P m2,m1,c0,p1;
    if(d<2){ int s=R5(kp),ox=d==0,oy=d==1;
        m2=cpm(PR,s,lx-2*ox,ly-2*oy);m1=cpm(PR,s,lx-ox,ly-oy);c0=cpm(PR,s,lx,ly);p1=cpm(PR,s,lx+ox,ly+oy);
    } else { m2=cpm(PR,R5(kp-2),lx,ly);m1=cpm(PR,R5(kp-1),lx,ly);c0=cpm(PR,R5(kp),lx,ly);p1=cpm(PR,R5(kp+1),lx,ly); }
    P sL=slope(m2,m1,c0),sR=slope(m1,c0,p1);
    P L=psub(hanc1d(m1,sL,d),sL,0.5f),R=psub(hanc1d(c0,sR,d),sR,-0.5f);
    size_t gb=gidx(wrap(x0-NG+lx,NX),wrap(y0-NG+ly,NY),wrap(kp,NZ));
    float bn=d==0?q.bx[gb]:d==1?q.by[gb]:q.bz[gb];
    L.r=fmaxf(L.r,SMALLR);L.p=fmaxf(L.p,SMALLP);R.r=fmaxf(R.r,SMALLR);R.p=fmaxf(R.p,SMALLP);
    if(d==0){L.bx=bn;R.bx=bn;}else if(d==1){L.by=bn;R.by=bn;}else{L.bz=bn;R.bz=bn;}
    return hll(L,R,d);
}
#define EZ(MR,a,b,kp) (0.25f*( MRV(MR,2,kp,a,b)+MRV(MR,2,kp,(a)-1,b) - MRV(MR,0,kp,a,b)-MRV(MR,0,kp,a,(b)-1) ))
#define EX(MR,a,b,kp) (0.25f*( MRV(MR,5,kp,a,b)+MRV(MR,5,kp,a,(b)-1) - MRV(MR,3,kp,a,b)-MRV(MR,3,(kp)-1,a,b) ))
#define EY(MR,a,b,kp) (0.25f*( MRV(MR,1,kp,a,b)+MRV(MR,1,(kp)-1,a,b) - MRV(MR,4,kp,a,b)-MRV(MR,4,kp,(a)-1,b) ))

__global__ void __launch_bounds__(THREADS) ct_march(Ptrs q, Ptrs o){
    extern __shared__ __half sm[];
    __half* PR=sm; __half* MR=sm + 8*5*PT;
    int tid=threadIdx.x, tx=tid%OX, ty=tid/OX, x0=blockIdx.x*OX, y0=blockIdx.y*OY, li=tx+NG, lj=ty+NG;
    auto loadp=[&](int kp){ int s=R5(kp);     // load prim plane kp (periodic wrap) into ring slot
        for(int c=tid;c<PT;c+=THREADS){ int lx=c%TX,ly=c/TX;
            size_t g=gidx(wrap(x0-NG+lx,NX),wrap(y0-NG+ly,NY),wrap(kp,NZ));
            PR[0*5*PT+s*PT+c]=(__half)q.U[0][g];PR[1*5*PT+s*PT+c]=(__half)q.U[1][g];PR[2*5*PT+s*PT+c]=(__half)q.U[2][g];
            PR[3*5*PT+s*PT+c]=(__half)q.U[3][g];PR[4*5*PT+s*PT+c]=(__half)q.U[4][g];
            PR[5*5*PT+s*PT+c]=(__half)q.cx[g];PR[6*5*PT+s*PT+c]=(__half)q.cy[g];PR[7*5*PT+s*PT+c]=(__half)q.cz[g]; } };
    // PRIME: pre-load the periodic wrap planes -3,-2,-1 (= NZ-3..NZ-1) so the first magflux's z-stencil
    // and plane-0's update see real periodic data — converts the wrap-dependency to a linear sweep.
    loadp(-3); loadp(-2); loadp(-1); __syncthreads();
    for(int L=0; L<NZ+2; L++){
        if(L<NZ) loadp(L);
        __syncthreads();
        { int m=L-1, s=R3(m);                  // mag fluxes of plane m (m=-1 at L=0: the wrap plane,
                                               // valid thanks to priming) over the owned±1 halo
            for(int c=tid;c<(OX+2)*(OY+2);c+=THREADS){ int lx=c%(OX+2)+NG-1, ly=c/(OX+2)+NG-1; int idx=lx+TX*ly;
                F8 fx=fflm(PR,q,0,m,lx,ly,x0,y0),fy=fflm(PR,q,1,m,lx,ly,x0,y0),fz=fflm(PR,q,2,m,lx,ly,x0,y0);
                MR[0*3*PT+s*PT+idx]=(__half)fx.by;MR[1*3*PT+s*PT+idx]=(__half)fx.bz;
                MR[2*3*PT+s*PT+idx]=(__half)fy.bx;MR[3*3*PT+s*PT+idx]=(__half)fy.bz;
                MR[4*3*PT+s*PT+idx]=(__half)fz.bx;MR[5*3*PT+s*PT+idx]=(__half)fz.by; } }
        __syncthreads();
        if(L>=2){ int k=L-2; size_t g=gidx(x0+tx,y0+ty,k);   // update plane k (hydro inline + face-B)
            F8 fxl=fflm(PR,q,0,k,li,lj,x0,y0),fxh=fflm(PR,q,0,k,li+1,lj,x0,y0);
            F8 fyl=fflm(PR,q,1,k,li,lj,x0,y0),fyh=fflm(PR,q,1,k,li,lj+1,x0,y0);
            F8 fzl=fflm(PR,q,2,k,li,lj,x0,y0),fzh=fflm(PR,q,2,k+1,li,lj,x0,y0);
            o.U[0][g]=q.U[0][g]+DTDX*((fxl.r-fxh.r)+(fyl.r-fyh.r)+(fzl.r-fzh.r));
            o.U[1][g]=q.U[1][g]+DTDX*((fxl.mx-fxh.mx)+(fyl.mx-fyh.mx)+(fzl.mx-fzh.mx));
            o.U[2][g]=q.U[2][g]+DTDX*((fxl.my-fxh.my)+(fyl.my-fyh.my)+(fzl.my-fzh.my));
            o.U[3][g]=q.U[3][g]+DTDX*((fxl.mz-fxh.mz)+(fyl.mz-fyh.mz)+(fzl.mz-fzh.mz));
            o.U[4][g]=q.U[4][g]+DTDX*((fxl.E-fxh.E)+(fyl.E-fyh.E)+(fzl.E-fzh.E));
            o.bx[g]=q.bx[g]-DTDX*((EZ(MR,li,lj+1,k)-EZ(MR,li,lj,k))-(EY(MR,li,lj,k+1)-EY(MR,li,lj,k)));
            o.by[g]=q.by[g]-DTDX*((EX(MR,li,lj,k+1)-EX(MR,li,lj,k))-(EZ(MR,li+1,lj,k)-EZ(MR,li,lj,k)));
            o.bz[g]=q.bz[g]-DTDX*((EY(MR,li+1,lj,k)-EY(MR,li,lj,k))-(EX(MR,li,lj+1,k)-EX(MR,li,lj,k))); }
        __syncthreads();
    }
}
#ifdef AS_LIB
extern "C" {
int march_nv(){return 8;} int march_nx(){return NX;}
void march_set_dtdx(float v){cudaMemcpyToSymbol(DTDX,&v,sizeof(float));}
void march_set_gamma(float g){cudaMemcpyToSymbol(GAMMA,&g,sizeof(float));}
double ct_run(float* const* qv,float* const* ov,float* const* cc,int nsteps){
    size_t n=(size_t)NX*NY*NZ; Ptrs Q,O;
    for(int v=0;v<5;v++){Q.U[v]=qv[v];O.U[v]=ov[v];}
    Q.bx=qv[5];Q.by=qv[6];Q.bz=qv[7];O.bx=ov[5];O.by=ov[6];O.bz=ov[7];
    Q.cx=cc[0];Q.cy=cc[1];Q.cz=cc[2];O.cx=cc[0];O.cy=cc[1];O.cz=cc[2];
    dim3 grid(NX/OX,NY/OY,1); size_t shmem=sizeof(__half)*(8*5*PT+6*3*PT);
    cudaFuncSetAttribute(ct_march,cudaFuncAttributeMaxDynamicSharedMemorySize,shmem);
    cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);cudaEventRecord(t0);
    for(int s=0;s<nsteps;s++){ bcc_kernel<<<(n+255)/256,256>>>(Q); ct_march<<<grid,THREADS,shmem>>>(Q,O); Ptrs t=Q;Q=O;O=t; }
    cudaEventRecord(t1);cudaEventSynchronize(t1); float ms=0;cudaEventElapsedTime(&ms,t0,t1);
    if(Q.U[0]!=qv[0]){for(int v=0;v<5;v++)cudaMemcpy(qv[v],Q.U[v],n*4,cudaMemcpyDeviceToDevice);
        cudaMemcpy(qv[5],Q.bx,n*4,cudaMemcpyDeviceToDevice);cudaMemcpy(qv[6],Q.by,n*4,cudaMemcpyDeviceToDevice);cudaMemcpy(qv[7],Q.bz,n*4,cudaMemcpyDeviceToDevice);}
    return (double)ms;
}
int ct_regs(){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,ct_march);return fa.numRegs;}
int ct_shmem(){return (int)(sizeof(__half)*(8*5*PT+6*3*PT));}
}
#endif
#ifndef AS_LIB
int main(){printf("spike_ctm: build -DAS_LIB\n");return 0;}
#endif
