// spike_hys.cu — STAGED two-sweep hydro (5-var Euler) = DOCUMENTED NEGATIVE. Phase A: each face flux
// computed ONCE into a shared f16 flux tile (no fused 2x face recompute). Phase B: the update reads it.
// -DPPM = lean parabolic-edge recon; default PLM(MonCen). Both +1D-Hancock.
//
// RESULT (A6000, 480^3): the two-sweep DOES keep registers low (PLM 48, PPM 56-62 regs vs the fused
// march) and conserves to machine precision — but it is SLOWER than the fused 2.5D march at every tile:
//   staged PLM 4967  vs fused march 6865   |   staged PPM 3745  vs fused march 3995.
// WHY: the 3D-tile halo redundancy (an 8^3-owned tile in a 12^3 NG=2 tile recomputes ~70% halo fluxes
// per block) costs MORE than the 2x face-recompute it removes. The fused 2.5D march z-STREAMS (zero
// z-halo, only ~1.7x x-y over-read), so its register-light 2x-recompute beats the tile's halo. Lesson:
// staging removes register inflation but the 3D-tile structure reintroduces it as halo compute. The
// two-sweep would only WIN if PPM were heavy enough to drop the FUSED kernel to 1 block (the full-
// characteristic PPM, 201 regs) — the lean parabolic PPM is already register-fine fused, so staging
// only adds halo overhead. The remaining (untried, uncertain) structure that could win: a staged MARCH
// (z-stream + flux ring -> no halo AND each face once), but the flux ring may itself drop occupancy.
#include <cstdio>
#include <cuda_fp16.h>
#ifndef NX
#define NX 480
#endif
#ifndef NY
#define NY 480
#endif
#ifndef NZ
#define NZ 480
#endif
#ifndef OX
#define OX 8
#endif
#ifndef OY
#define OY 8
#endif
#ifndef OZ
#define OZ 4
#endif
#define NG 2
#define TX (OX+2*NG)
#define TY (OY+2*NG)
#define TZ (OZ+2*NG)
#define THREADS (OX*OY*OZ)
__constant__ float GAMMA=1.4f, DTDX=0.02f, SMALLR=1e-6f, SMALLP=1e-6f;
__device__ __forceinline__ int wrap(int i,int N){ int m=i%N; return m<0?m+N:m; }
__device__ __forceinline__ size_t gidx(int i,int j,int k){ return (size_t)i+(size_t)NX*((size_t)j+(size_t)NY*k); }
struct P { float r,u,v,w,p; };
struct F5 { float r,mx,my,mz,E; };
__device__ __forceinline__ float mc(float a,float b,float c){
    float dl=b-a,dr=c-b,dc=0.5f*(c-a); if(dl*dr<=0.f) return 0.f;
    float s=dl>0.f?1.f:-1.f; return s*fminf(fabsf(dc),fminf(2.f*fabsf(dl),2.f*fabsf(dr)));
}
__device__ __forceinline__ P psub(P q,P s,float a){ return {q.r+a*s.r,q.u+a*s.u,q.v+a*s.v,q.w+a*s.w,q.p+a*s.p}; }
__device__ __forceinline__ P slope(P a,P b,P c){ return {mc(a.r,b.r,c.r),mc(a.u,b.u,c.u),mc(a.v,b.v,c.v),mc(a.w,b.w,c.w),mc(a.p,b.p,c.p)}; }
__device__ __forceinline__ F5 dflux(P q,int d){
    float un=d==0?q.u:d==1?q.v:q.w, E=q.p/(GAMMA-1.f)+0.5f*q.r*(q.u*q.u+q.v*q.v+q.w*q.w);
    return {q.r*un, q.r*un*q.u+(d==0?q.p:0.f), q.r*un*q.v+(d==1?q.p:0.f), q.r*un*q.w+(d==2?q.p:0.f), (E+q.p)*un};
}
__device__ __forceinline__ F5 toC(P q){ return {q.r,q.r*q.u,q.r*q.v,q.r*q.w, q.p/(GAMMA-1.f)+0.5f*q.r*(q.u*q.u+q.v*q.v+q.w*q.w)}; }
__device__ __forceinline__ F5 hll(P L,P R,int d){
    float uL=d==0?L.u:d==1?L.v:L.w, uR=d==0?R.u:d==1?R.v:R.w;
    float aL=sqrtf(GAMMA*L.p/L.r),aR=sqrtf(GAMMA*R.p/R.r);
    float SL=fminf(fminf(uL-aL,uR-aR),0.f), SR=fmaxf(fmaxf(uL+aL,uR+aR),0.f);
    F5 FL=dflux(L,d),FR=dflux(R,d),UL=toC(L),UR=toC(R),F; float ih=1.f/(SR-SL);
    F.r =(SR*FL.r -SL*FR.r +SL*SR*(UR.r -UL.r ))*ih;
    F.mx=(SR*FL.mx-SL*FR.mx+SL*SR*(UR.mx-UL.mx))*ih;
    F.my=(SR*FL.my-SL*FR.my+SL*SR*(UR.my-UL.my))*ih;
    F.mz=(SR*FL.mz-SL*FR.mz+SL*SR*(UR.mz-UL.mz))*ih;
    F.E =(SR*FL.E -SL*FR.E +SL*SR*(UR.E -UL.E ))*ih; return F;
}
__device__ __forceinline__ P hanc1d(P q,P s,int d){
    F5 FL=dflux(psub(q,s,-0.5f),d),FR=dflux(psub(q,s,0.5f),d); F5 U=toC(q); float h=0.5f*DTDX;
    float r=fmaxf(U.r-h*(FR.r-FL.r),SMALLR),ir=1.f/r;
    float mx=U.mx-h*(FR.mx-FL.mx),my=U.my-h*(FR.my-FL.my),mz=U.mz-h*(FR.mz-FL.mz),E=U.E-h*(FR.E-FL.E);
    float vx=mx*ir,vy=my*ir,vz=mz*ir;
    return {r,vx,vy,vz,fmaxf((GAMMA-1.f)*(E-0.5f*r*(vx*vx+vy*vy+vz*vz)),SMALLP)};
}
#ifdef PPM
__device__ __forceinline__ float ppm1(float qm,float q0,float qp,float sgn){
    float d=0.25f*(qp-qm), cu=(qm-2.f*q0+qp)*(1.f/6.f); float qr=q0+d-cu, ql=q0-d-cu;
    if((qr-q0)*(q0-ql)<=0.f){ qr=q0; ql=q0; }
    else { float dqe=qr-ql, d6=6.f*(q0-0.5f*(ql+qr));
        if(dqe*d6> dqe*dqe) ql=3.f*q0-2.f*qr; if(dqe*d6<-dqe*dqe) qr=3.f*q0-2.f*ql; }
    return sgn>0.f?qr:ql;
}
__device__ __forceinline__ P ppm_edge(P a,P b,P c,float sgn){
    P e={ppm1(a.r,b.r,c.r,sgn),ppm1(a.u,b.u,c.u,sgn),ppm1(a.v,b.v,c.v,sgn),ppm1(a.w,b.w,c.w,sgn),ppm1(a.p,b.p,c.p,sgn)};
    e.r=fmaxf(e.r,SMALLR); e.p=fmaxf(e.p,SMALLP); return e;
}
__device__ __forceinline__ P padd_pred(P e,P mh,P c){
    P o={e.r+mh.r-c.r,e.u+mh.u-c.u,e.v+mh.v-c.v,e.w+mh.w-c.w,e.p+mh.p-c.p};
    o.r=fmaxf(o.r,SMALLR); o.p=fmaxf(o.p,SMALLP); return o;
}
#endif
struct Ptrs { float* U[5]; };

#define PT (TX*TY*TZ)
#define FX (OX+1)
#define FY (OY+1)
#define FZ (OZ+1)
#define FXYZ (FX*FY*FZ)
#define PV(S,v,lx,ly,lz) ((float)(S)[(v)*PT + (lx)+TX*((ly)+TY*(lz))])
#define FIDX(lx,ly,lz) (((lx)-NG) + FX*(((ly)-NG) + FY*((lz)-NG)))
#define FF(FL,c,d,lx,ly,lz) ((float)(FL)[((c)*3+(d))*FXYZ + FIDX(lx,ly,lz)])
__device__ __forceinline__ P cpt(__half*S,int lx,int ly,int lz){
    float r=fmaxf(PV(S,0,lx,ly,lz),SMALLR),ir=1.f/r;
    float mx=PV(S,1,lx,ly,lz),my=PV(S,2,lx,ly,lz),mz=PV(S,3,lx,ly,lz),E=PV(S,4,lx,ly,lz);
    float vx=mx*ir,vy=my*ir,vz=mz*ir;
    return {r,vx,vy,vz,fmaxf((GAMMA-1.f)*(E-0.5f*r*(vx*vx+vy*vy+vz*vz)),SMALLP)};
}
// flux at the lower-d face of tile cell (lx,ly,lz), computed ONCE
__device__ __forceinline__ F5 ffl(__half*S,int d,int lx,int ly,int lz){
    int ox=d==0,oy=d==1,oz=d==2;
    P m2=cpt(S,lx-2*ox,ly-2*oy,lz-2*oz),m1=cpt(S,lx-ox,ly-oy,lz-oz),c0=cpt(S,lx,ly,lz),p1=cpt(S,lx+ox,ly+oy,lz+oz);
    P sL=slope(m2,m1,c0),sR=slope(m1,c0,p1);
#ifdef PPM
    P L=padd_pred(ppm_edge(m2,m1,c0,+1.f),hanc1d(m1,sL,d),m1), R=padd_pred(ppm_edge(m1,c0,p1,-1.f),hanc1d(c0,sR,d),c0);
#else
    P L=psub(hanc1d(m1,sL,d),sL,0.5f), R=psub(hanc1d(c0,sR,d),sR,-0.5f);
#endif
    L.r=fmaxf(L.r,SMALLR);L.p=fmaxf(L.p,SMALLP);R.r=fmaxf(R.r,SMALLR);R.p=fmaxf(R.p,SMALLP);
    return hll(L,R,d);
}
__global__ void __launch_bounds__(THREADS) hyd(Ptrs q, Ptrs o){
    extern __shared__ __half sm[];
    __half* S=sm; __half* FL=sm + 5*PT;
    int tid=threadIdx.x, tx=tid%OX, ty=(tid/OX)%OY, tz=tid/(OX*OY);
    int x0=blockIdx.x*OX, y0=blockIdx.y*OY, z0=blockIdx.z*OZ;
    for(int c=tid;c<PT;c+=THREADS){ int lx=c%TX,ly=(c/TX)%TY,lz=c/(TX*TY);
        size_t g=gidx(wrap(x0-NG+lx,NX),wrap(y0-NG+ly,NY),wrap(z0-NG+lz,NZ));
        #pragma unroll
        for(int v=0;v<5;v++) S[v*PT+c]=(__half)q.U[v][g]; }
    __syncthreads();
    // PHASE A: each face flux computed ONCE -> shared
    for(int c=tid;c<FXYZ;c+=THREADS){ int fi=c%FX,fj=(c/FX)%FY,fk=c/(FX*FY); int lx=fi+NG,ly=fj+NG,lz=fk+NG;
        #pragma unroll
        for(int d=0;d<3;d++){ F5 f=ffl(S,d,lx,ly,lz);
            FL[(0*3+d)*FXYZ+c]=(__half)f.r;  FL[(1*3+d)*FXYZ+c]=(__half)f.mx; FL[(2*3+d)*FXYZ+c]=(__half)f.my;
            FL[(3*3+d)*FXYZ+c]=(__half)f.mz; FL[(4*3+d)*FXYZ+c]=(__half)f.E; } }
    __syncthreads();
    // PHASE B: conservative update reads the stored fluxes (f32 base from global)
    int li=tx+NG,lj=ty+NG,lk=tz+NG; size_t g=gidx(x0+tx,y0+ty,z0+tz);
#ifndef MEMFLOOR
    #define HY(cc) ( (FF(FL,cc,0,li,lj,lk)-FF(FL,cc,0,li+1,lj,lk)) + (FF(FL,cc,1,li,lj,lk)-FF(FL,cc,1,li,lj+1,lk)) + (FF(FL,cc,2,li,lj,lk)-FF(FL,cc,2,li,lj,lk+1)) )
    o.U[0][g]=q.U[0][g]+DTDX*HY(0); o.U[1][g]=q.U[1][g]+DTDX*HY(1); o.U[2][g]=q.U[2][g]+DTDX*HY(2);
    o.U[3][g]=q.U[3][g]+DTDX*HY(3); o.U[4][g]=q.U[4][g]+DTDX*HY(4);
#else
    for(int v=0;v<5;v++) o.U[v][g]=q.U[v][g];
#endif
}
#ifdef AS_LIB
extern "C" {
int march_nv(){return 5;} int march_nx(){return NX;}
void march_set_dtdx(float v){cudaMemcpyToSymbol(DTDX,&v,sizeof(float));}
double march_run_dev(float* const* qv,float* const* ov,int nsteps){
    size_t n=(size_t)NX*NY*NZ; Ptrs Q,O; for(int v=0;v<5;v++){Q.U[v]=qv[v];O.U[v]=ov[v];}
    dim3 grid(NX/OX,NY/OY,NZ/OZ); size_t shmem=sizeof(__half)*(5*PT+15*FXYZ);
    cudaFuncSetAttribute(hyd,cudaFuncAttributeMaxDynamicSharedMemorySize,shmem);
    cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);cudaEventRecord(t0);
    for(int s=0;s<nsteps;s++){ hyd<<<grid,THREADS,shmem>>>(Q,O); Ptrs t=Q;Q=O;O=t; }
    cudaEventRecord(t1);cudaEventSynchronize(t1); float ms=0;cudaEventElapsedTime(&ms,t0,t1);
    if(Q.U[0]!=qv[0]) for(int v=0;v<5;v++) cudaMemcpy(qv[v],Q.U[v],n*4,cudaMemcpyDeviceToDevice);
    return (double)ms;
}
int march_regs(){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,hyd);return fa.numRegs;}
int march_shmem(){return (int)(sizeof(__half)*(5*PT+15*FXYZ));}
}
#endif
#ifndef AS_LIB
int main(){printf("spike_hys: build -DAS_LIB\n");return 0;}
#endif
