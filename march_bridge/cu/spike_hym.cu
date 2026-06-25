// spike_hym.cu — STAGED 2.5D hydro MARCH = DOCUMENTED NEGATIVE (the WORST of the three structures).
// z-stream (zero z-halo) + a rolling flux ring so each face flux is computed ONCE. -DPPM = parabolic.
//
// RESULT (A6000, 480^3): WORSE than both the fused march AND the staged 3D tile:
//   structure            PLM    PPM
//   fused march (recompute) 6865  3995   <- BEST (4 blocks PLM / 3 blocks PPM, no flux storage)
//   staged 3D tile          4967  3745   (2-3 blocks, halo redundancy)
//   staged march (this)     3729  2732   <- WORST (2 blocks: the 15-comp x 3-plane flux ring = 43KB)
// WHY: storing the flux (15 vals/cell f16) to AVOID the 2x recompute costs more shared than it saves —
// it drops the kernel from the fused march's 4 blocks to 2. The recompute is CHEAP relative to the
// occupancy the flux storage costs. UNIFIED LESSON across spike_hys.cu + this: for a REGISTER-LIGHT
// recompute (hydro/GLM/CT-PLM marches, 48-128 regs, high occupancy) the fused recompute is optimal;
// staging only wins when the FUSED recompute is register-HEAVY enough to pin 1 block (CT's edge-EMF,
// 255 regs -> staging to shared won 61->1206). Stage the recompute that blows registers; never the
// cheap one. See spike_hys.cu (staged 3D tile) and BENCHMARKS.md.
// Prim ring (5 planes) + hydro-flux ring (3 planes, lower faces); update reads the ring (lag 2).
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
#define OX 16
#endif
#ifndef OY
#define OY 12
#endif
#define NG 2
#define TX (OX+2*NG)
#define TY (OY+2*NG)
#define THREADS (OX*OY)
#define PTP (TX*TY)
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
    F.r =(SR*FL.r -SL*FR.r +SL*SR*(UR.r -UL.r ))*ih; F.mx=(SR*FL.mx-SL*FR.mx+SL*SR*(UR.mx-UL.mx))*ih;
    F.my=(SR*FL.my-SL*FR.my+SL*SR*(UR.my-UL.my))*ih; F.mz=(SR*FL.mz-SL*FR.mz+SL*SR*(UR.mz-UL.mz))*ih;
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
    else { float dqe=qr-ql, d6=6.f*(q0-0.5f*(ql+qr)); if(dqe*d6>dqe*dqe) ql=3.f*q0-2.f*qr; if(dqe*d6<-dqe*dqe) qr=3.f*q0-2.f*ql; }
    return sgn>0.f?qr:ql;
}
__device__ __forceinline__ P ppm_edge(P a,P b,P c,float sgn){
    P e={ppm1(a.r,b.r,c.r,sgn),ppm1(a.u,b.u,c.u,sgn),ppm1(a.v,b.v,c.v,sgn),ppm1(a.w,b.w,c.w,sgn),ppm1(a.p,b.p,c.p,sgn)};
    e.r=fmaxf(e.r,SMALLR); e.p=fmaxf(e.p,SMALLP); return e;
}
__device__ __forceinline__ P padd_pred(P e,P mh,P c){
    P o={e.r+mh.r-c.r,e.u+mh.u-c.u,e.v+mh.v-c.v,e.w+mh.w-c.w,e.p+mh.p-c.p}; o.r=fmaxf(o.r,SMALLR); o.p=fmaxf(o.p,SMALLP); return o;
}
#endif
struct Ptrs { float* U[5]; };
#define R5(p) (((p)%5+5)%5)
#define R3(p) (((p)%3+3)%3)
#define PRV(PR,v,s,lx,ly) ((float)(PR)[(v)*5*PTP + (s)*PTP + (lx)+TX*(ly)])
#define FRV(FR,c,d,kp,lx,ly) ((float)(FR)[((c)*3+(d))*3*PTP + R3(kp)*PTP + (lx)+TX*(ly)])
__device__ __forceinline__ P cpm(__half*PR,int s,int lx,int ly){
    float r=fmaxf(PRV(PR,0,s,lx,ly),SMALLR),ir=1.f/r;
    float mx=PRV(PR,1,s,lx,ly),my=PRV(PR,2,s,lx,ly),mz=PRV(PR,3,s,lx,ly),E=PRV(PR,4,s,lx,ly);
    float vx=mx*ir,vy=my*ir,vz=mz*ir;
    return {r,vx,vy,vz,fmaxf((GAMMA-1.f)*(E-0.5f*r*(vx*vx+vy*vy+vz*vz)),SMALLP)};
}
// lower-d face flux of cell (lx,ly) in plane kp (d=0,1 in-plane; d=2 z between kp-1,kp)
__device__ __forceinline__ F5 fflm(__half*PR,int d,int kp,int lx,int ly){
    P m2,m1,c0,p1;
    if(d<2){ int s=R5(kp),ox=d==0,oy=d==1; m2=cpm(PR,s,lx-2*ox,ly-2*oy);m1=cpm(PR,s,lx-ox,ly-oy);c0=cpm(PR,s,lx,ly);p1=cpm(PR,s,lx+ox,ly+oy); }
    else { m2=cpm(PR,R5(kp-2),lx,ly);m1=cpm(PR,R5(kp-1),lx,ly);c0=cpm(PR,R5(kp),lx,ly);p1=cpm(PR,R5(kp+1),lx,ly); }
    P sL=slope(m2,m1,c0),sR=slope(m1,c0,p1);
#ifdef PPM
    P L=padd_pred(ppm_edge(m2,m1,c0,+1.f),hanc1d(m1,sL,d),m1), R=padd_pred(ppm_edge(m1,c0,p1,-1.f),hanc1d(c0,sR,d),c0);
#else
    P L=psub(hanc1d(m1,sL,d),sL,0.5f), R=psub(hanc1d(c0,sR,d),sR,-0.5f);
#endif
    L.r=fmaxf(L.r,SMALLR);L.p=fmaxf(L.p,SMALLP);R.r=fmaxf(R.r,SMALLR);R.p=fmaxf(R.p,SMALLP);
    return hll(L,R,d);
}
__global__ void __launch_bounds__(THREADS) hym(Ptrs q, Ptrs o){
    extern __shared__ __half sm[];
    __half* PR=sm; __half* FR=sm + 5*5*PTP;
    int tid=threadIdx.x, tx=tid%OX, ty=tid/OX, x0=blockIdx.x*OX, y0=blockIdx.y*OY, li=tx+NG, lj=ty+NG;
    auto loadp=[&](int kp){ int s=R5(kp);
        for(int c=tid;c<PTP;c+=THREADS){ int lx=c%TX,ly=c/TX;
            size_t g=gidx(wrap(x0-NG+lx,NX),wrap(y0-NG+ly,NY),wrap(kp,NZ));
            #pragma unroll
            for(int v=0;v<5;v++) PR[v*5*PTP+s*PTP+c]=(__half)q.U[v][g]; } };
    auto fluxset=[&](int m){ int s=R3(m);   // compute lower-face fluxes (x,y,z) of plane m over owned+1
        for(int c=tid;c<(OX+1)*(OY+1);c+=THREADS){ int lx=c%(OX+1)+NG, ly=c/(OX+1)+NG; int idx=lx+TX*ly;
            #pragma unroll
            for(int d=0;d<3;d++){ F5 f=fflm(PR,d,m,lx,ly);
                FR[(0*3+d)*3*PTP+s*PTP+idx]=(__half)f.r;  FR[(1*3+d)*3*PTP+s*PTP+idx]=(__half)f.mx;
                FR[(2*3+d)*3*PTP+s*PTP+idx]=(__half)f.my; FR[(3*3+d)*3*PTP+s*PTP+idx]=(__half)f.mz; FR[(4*3+d)*3*PTP+s*PTP+idx]=(__half)f.E; } } };
    loadp(-3); loadp(-2); loadp(-1); __syncthreads();
    for(int L=0; L<NZ+2; L++){
        if(L<NZ) loadp(L);
        __syncthreads();
        fluxset(L-1);                          // plane m=L-1 (z-flux needs L-3..L; primed for the wrap)
        __syncthreads();
        if(L>=2){ int k=L-2; size_t g=gidx(x0+tx,y0+ty,k);
#ifndef MEMFLOOR
            #define HY(cc) ( (FRV(FR,cc,0,k,li,lj)-FRV(FR,cc,0,k,li+1,lj)) + (FRV(FR,cc,1,k,li,lj)-FRV(FR,cc,1,k,li,lj+1)) + (FRV(FR,cc,2,k,li,lj)-FRV(FR,cc,2,k+1,li,lj)) )
            o.U[0][g]=q.U[0][g]+DTDX*HY(0); o.U[1][g]=q.U[1][g]+DTDX*HY(1); o.U[2][g]=q.U[2][g]+DTDX*HY(2);
            o.U[3][g]=q.U[3][g]+DTDX*HY(3); o.U[4][g]=q.U[4][g]+DTDX*HY(4);
#else
            for(int v=0;v<5;v++) o.U[v][g]=q.U[v][g];
#endif
        }
        __syncthreads();
    }
}
#ifdef AS_LIB
extern "C" {
int march_nv(){return 5;} int march_nx(){return NX;}
void march_set_dtdx(float v){cudaMemcpyToSymbol(DTDX,&v,sizeof(float));}
double march_run_dev(float* const* qv,float* const* ov,int nsteps){
    size_t n=(size_t)NX*NY*NZ; Ptrs Q,O; for(int v=0;v<5;v++){Q.U[v]=qv[v];O.U[v]=ov[v];}
    dim3 grid(NX/OX,NY/OY,1); size_t shmem=sizeof(__half)*(5*5*PTP+15*3*PTP);
    cudaFuncSetAttribute(hym,cudaFuncAttributeMaxDynamicSharedMemorySize,shmem);
    cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);cudaEventRecord(t0);
    for(int s=0;s<nsteps;s++){ hym<<<grid,THREADS,shmem>>>(Q,O); Ptrs t=Q;Q=O;O=t; }
    cudaEventRecord(t1);cudaEventSynchronize(t1); float ms=0;cudaEventElapsedTime(&ms,t0,t1);
    if(Q.U[0]!=qv[0]) for(int v=0;v<5;v++) cudaMemcpy(qv[v],Q.U[v],n*4,cudaMemcpyDeviceToDevice);
    return (double)ms;
}
int march_regs(){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,hym);return fa.numRegs;}
int march_shmem(){return (int)(sizeof(__half)*(5*5*PTP+15*3*PTP));}
}
#endif
#ifndef AS_LIB
int main(){printf("spike_hym: build -DAS_LIB\n");return 0;}
#endif
