// spike_mhd.cu — GLM-MHD 2.5D unigrid light line-march (transverse-free 1D-Hancock + HLL + Dedner GLM).
//
// The MHD twin of spike_25d.cu's -DHANCOCK1D path. 9-var GLM-MHD, physics transliterated
// faithfully from the validated Julia prototype (GLMMHDTurb/glmmhd_turb.jl: cons2prim,
// phys_flux_x, fast_speed, glm_pair, hll_x, riemann_hll) so the C kernel and the Julia
// reference solve identical equations. Same fused 2.5D march: x-y tiled in shared (f16,
// REQUIRED — fp32 9-var tile is 76KB > 48KB; f16 is 38KB), z marched through a rolling
// 5-plane ring, normal-only 1D Hancock (2nd order space+time, no transverse predictor).
//
// Conserved order : (rho, rho*vx, rho*vy, rho*vz, E, Bx, By, Bz, psi)
// Primitive order : (rho, vx, vy, vz, p, Bx, By, Bz, psi)
//
// Build (standalone throughput probe):
//   nvcc -O3 -arch=sm_86 --use_fast_math -DNX=480 -DNY=480 -DNZ=480 -o spike_mhd spike_mhd.cu
// Build (Julia bridge .so, see GLMMHDTurb/march_bridge):
//   nvcc -O3 -arch=sm_86 --use_fast_math -DAS_LIB -DNX=480 -DNY=480 -DNZ=480 --shared -Xcompiler -fPIC -o libmhd480.so spike_mhd.cu
// --use_fast_math is load-bearing (frees registers in the sqrt/recip-heavy fast-speed + HLL path).
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
#ifndef NSP
#define NSP 0                  // passive species via CMA (ride the mass flux). -DPPM = parabolic recon.
#endif
#define NV (9+NSP)
#ifndef OX
#define OX 32
#endif
#ifndef OY
#define OY 8
#endif
#define GHOST 2
#define GX (OX+2*GHOST)
#define GY (OY+2*GHOST)
#define PLANES 5
#ifndef THREADS
#define THREADS (OX*OY)
#endif

__device__ __forceinline__ size_t gidx(int i,int j,int k){ return (size_t)i + (size_t)NX*((size_t)j + (size_t)NY*k); }
__device__ __forceinline__ int wrap(int i,int N){ int m=i%N; return m<0?m+N:m; }
__device__ __forceinline__ int ring(int m){ int r=m%PLANES; return r<0?r+PLANES:r; }

__constant__ float GAMMA=1.4f, DTDX=0.02f, SMALLR=1e-6f, SMALLP=1e-6f, CH=1.0f, GLMFAC=1.0f;

struct Prim { float r,u,v,w,p,bx,by,bz,ps;
#if NSP>0
    float s[NSP];                          // passive species FRACTION X
#endif
};
struct Cons { float r,mx,my,mz,E,bx,by,bz,ps;
#if NSP>0
    float s[NSP];                          // conserved rho*X (or CMA flux)
#endif
};

__device__ __forceinline__ Prim prim_from_cons(float r,float mx,float my,float mz,float E,float bx,float by,float bz,float ps){
    Prim q; q.r=fmaxf(r,SMALLR); float ir=1.f/q.r;
    q.u=mx*ir; q.v=my*ir; q.w=mz*ir;
    float ekin=0.5f*(mx*mx+my*my+mz*mz)*ir, emag=0.5f*(bx*bx+by*by+bz*bz);
    q.p=fmaxf((GAMMA-1.f)*(E-ekin-emag),SMALLP);
    q.bx=bx; q.by=by; q.bz=bz; q.ps=ps; return q;   // species set by GP after (tile holds fraction)
}
__device__ __forceinline__ Cons toCons(Prim q){
    Cons c; c.r=q.r; c.mx=q.r*q.u; c.my=q.r*q.v; c.mz=q.r*q.w;
    float ekin=0.5f*q.r*(q.u*q.u+q.v*q.v+q.w*q.w), emag=0.5f*(q.bx*q.bx+q.by*q.by+q.bz*q.bz);
    c.E=q.p/(GAMMA-1.f)+ekin+emag; c.bx=q.bx; c.by=q.by; c.bz=q.bz; c.ps=q.ps;
#if NSP>0
    #pragma unroll
    for(int i=0;i<NSP;i++) c.s[i]=q.r*q.s[i];   // conserved rho*X
#endif
    return c;
}
// fast magnetosonic speed for normal field bn (prim state, x-normal frame)
__device__ __forceinline__ float fast_speed(Prim q,float bn){
    float c2=GAMMA*q.p/q.r, b2=(q.bx*q.bx+q.by*q.by+q.bz*q.bz)/q.r, d2=0.5f*(b2+c2);
    return sqrtf(d2+sqrtf(fmaxf(d2*d2 - c2*bn*bn/q.r, 0.f)));
}
// ideal-MHD x-flux (x-normal frame); F[Bx]/F[psi] are GLM-set (0 here, overwritten at faces)
__device__ __forceinline__ Cons phys_flux_x(Prim q){
    float b2=q.bx*q.bx+q.by*q.by+q.bz*q.bz, ptot=q.p+0.5f*b2;
    float E=q.p/(GAMMA-1.f)+0.5f*q.r*(q.u*q.u+q.v*q.v+q.w*q.w)+0.5f*b2;
    float vdotb=q.u*q.bx+q.v*q.by+q.w*q.bz;
    Cons f; f.r=q.r*q.u;
    f.mx=q.r*q.u*q.u+ptot-q.bx*q.bx;
    f.my=q.r*q.u*q.v-q.bx*q.by;
    f.mz=q.r*q.u*q.w-q.bx*q.bz;
    f.E =(E+ptot)*q.u-q.bx*vdotb;
    f.bx=0.f; f.by=q.u*q.by-q.v*q.bx; f.bz=q.u*q.bz-q.w*q.bx; f.ps=0.f;
    return f;
}
// rotate prim so direction dir(0=x,1=y,2=z) is x-normal (cyclic vel & B perm)
__device__ __forceinline__ Prim rot_to(Prim q,int dir){
    if(dir==0) return q;
    Prim r=q;
    if(dir==1){ r.u=q.v; r.v=q.w; r.w=q.u; r.bx=q.by; r.by=q.bz; r.bz=q.bx; }
    else       { r.u=q.w; r.v=q.u; r.w=q.v; r.bx=q.bz; r.by=q.bx; r.bz=q.by; }
    return r;
}
// rotate a flux (conserved-comp vector) from x-normal frame back to dir
__device__ __forceinline__ Cons rot_flux_from(Cons f,int dir){
    if(dir==0) return f;
    Cons r=f;
    if(dir==1){ r.mx=f.mz; r.my=f.mx; r.mz=f.my; r.bx=f.bz; r.by=f.bx; r.bz=f.by; }
    else       { r.mx=f.my; r.my=f.mz; r.mz=f.mx; r.bx=f.by; r.by=f.bz; r.bz=f.bx; }
    return r;
}
__device__ __forceinline__ Cons dirFlux(Prim q,int dir){ return rot_flux_from(phys_flux_x(rot_to(q,dir)),dir); }

__device__ __forceinline__ float moncen(float a,float b,float c){
    float dl=b-a, dr=c-b, dc=0.5f*(c-a);
    if(dl*dr<=0.f) return 0.f;
    float s=dl>0.f?1.f:-1.f;
    return s*fminf(fabsf(dc),fminf(2.f*fabsf(dl),2.f*fabsf(dr)));
}
__device__ __forceinline__ Prim slopeP(Prim a,Prim b,Prim c){
    Prim s; s.r=moncen(a.r,b.r,c.r); s.u=moncen(a.u,b.u,c.u); s.v=moncen(a.v,b.v,c.v);
    s.w=moncen(a.w,b.w,c.w); s.p=moncen(a.p,b.p,c.p); s.bx=moncen(a.bx,b.bx,c.bx);
    s.by=moncen(a.by,b.by,c.by); s.bz=moncen(a.bz,b.bz,c.bz); s.ps=moncen(a.ps,b.ps,c.ps);
#if NSP>0
    #pragma unroll
    for(int i=0;i<NSP;i++) s.s[i]=moncen(a.s[i],b.s[i],c.s[i]);
#endif
    return s;
}
__device__ __forceinline__ Prim padd(Prim b,Prim s,float a){
    Prim e; e.r=b.r+a*s.r; e.u=b.u+a*s.u; e.v=b.v+a*s.v; e.w=b.w+a*s.w; e.p=b.p+a*s.p;
    e.bx=b.bx+a*s.bx; e.by=b.by+a*s.by; e.bz=b.bz+a*s.bz; e.ps=b.ps+a*s.ps;
#if NSP>0
    #pragma unroll
    for(int i=0;i<NSP;i++) e.s[i]=b.s[i]+a*s.s[i];
#endif
    return e;
}
// spatial face reconstruction b +/- 0.5 s (sign=+/-1), floored
__device__ __forceinline__ Prim edge(Prim b,Prim s,float sign){
    Prim e=padd(b,s,0.5f*sign); e.r=fmaxf(e.r,SMALLR); e.p=fmaxf(e.p,SMALLP); return e;
}
// normal-only (transverse-free) MUSCL-Hancock half-step of the cell CENTER, direction dir.
// U_half = U0 - 0.5 dtdx (F(c0+0.5s) - F(c0-0.5s)); ideal flux (GLM coupling deferred to faces).
__device__ __forceinline__ Prim hanc1d(Prim c0,Prim s,int dir){
    Cons FL=dirFlux(padd(c0,s,-0.5f),dir), FR=dirFlux(padd(c0,s,0.5f),dir);
    Cons U=toCons(c0); float h=0.5f*DTDX;
    U.r-=h*(FR.r-FL.r); U.mx-=h*(FR.mx-FL.mx); U.my-=h*(FR.my-FL.my); U.mz-=h*(FR.mz-FL.mz);
    U.E-=h*(FR.E-FL.E); U.bx-=h*(FR.bx-FL.bx); U.by-=h*(FR.by-FL.by); U.bz-=h*(FR.bz-FL.bz); U.ps-=h*(FR.ps-FL.ps);
    Prim mh=prim_from_cons(U.r,U.mx,U.my,U.mz,U.E,U.bx,U.by,U.bz,U.ps);
#if NSP>0
    float un=dir==0?c0.u:dir==1?c0.v:c0.w;   // passive advection predictor: X_half = X - h*un*dX
    #pragma unroll
    for(int i=0;i<NSP;i++) mh.s[i]=c0.s[i]-h*un*s.s[i];
#endif
    return mh;
}
#ifdef PPM
// lean parabolic-edge reconstruction (3-pt CW84 parabola + monotonize) + 1D-Hancock predictor.
__device__ __forceinline__ float ppm1(float qm,float q0,float qp,float sgn){
    float d=0.25f*(qp-qm), cu=(qm-2.f*q0+qp)*(1.f/6.f);
    float qr=q0+d-cu, ql=q0-d-cu;
    if((qr-q0)*(q0-ql)<=0.f){ qr=q0; ql=q0; }
    else { float dqe=qr-ql, d6=6.f*(q0-0.5f*(ql+qr));
        if(dqe*d6 >  dqe*dqe) ql=3.f*q0-2.f*qr;
        if(dqe*d6 < -dqe*dqe) qr=3.f*q0-2.f*ql; }
    return sgn>0.f?qr:ql;
}
__device__ __forceinline__ Prim ppm_edge(Prim a,Prim b,Prim c,float sgn){
    Prim e; e.r=ppm1(a.r,b.r,c.r,sgn); e.u=ppm1(a.u,b.u,c.u,sgn); e.v=ppm1(a.v,b.v,c.v,sgn);
    e.w=ppm1(a.w,b.w,c.w,sgn); e.p=ppm1(a.p,b.p,c.p,sgn); e.bx=ppm1(a.bx,b.bx,c.bx,sgn);
    e.by=ppm1(a.by,b.by,c.by,sgn); e.bz=ppm1(a.bz,b.bz,c.bz,sgn); e.ps=ppm1(a.ps,b.ps,c.ps,sgn);
#if NSP>0
    #pragma unroll
    for(int i=0;i<NSP;i++) e.s[i]=ppm1(a.s[i],b.s[i],c.s[i],sgn);
#endif
    e.r=fmaxf(e.r,SMALLR); e.p=fmaxf(e.p,SMALLP); return e;
}
__device__ __forceinline__ Prim padd_pred(Prim e,Prim mh,Prim c){   // PPM face + predictor shift (e+mh-c)
    Prim o; o.r=e.r+mh.r-c.r; o.u=e.u+mh.u-c.u; o.v=e.v+mh.v-c.v; o.w=e.w+mh.w-c.w; o.p=e.p+mh.p-c.p;
    o.bx=e.bx+mh.bx-c.bx; o.by=e.by+mh.by-c.by; o.bz=e.bz+mh.bz-c.bz; o.ps=e.ps+mh.ps-c.ps;
#if NSP>0
    #pragma unroll
    for(int i=0;i<NSP;i++) o.s[i]=e.s[i]+mh.s[i]-c.s[i];
#endif
    o.r=fmaxf(o.r,SMALLR); o.p=fmaxf(o.p,SMALLP); return o;
}
#endif
// Dedner GLM pair: clean normal field; returns cleaned bn* and psi* (uses CH).
__device__ __forceinline__ void glm_pair(float bnL,float bnR,float psiL,float psiR,float& bns,float& psis){
    bns =0.5f*(bnL+bnR)-0.5f*(psiR-psiL)/CH;
    psis=0.5f*(psiL+psiR)-0.5f*CH*(bnR-bnL);
}
// HLL Riemann in direction dir with GLM cleaning (matches riemann_hll/hll_x).
__device__ __forceinline__ Cons hllMHD(Prim Lq,Prim Rq,int dir){
    Prim L=rot_to(Lq,dir), R=rot_to(Rq,dir);
    L.r=fmaxf(L.r,SMALLR); L.p=fmaxf(L.p,SMALLP); R.r=fmaxf(R.r,SMALLR); R.p=fmaxf(R.p,SMALLP);
    float bns,psis; glm_pair(L.bx,R.bx,L.ps,R.ps,bns,psis);
    float fbn=psis, fpsi=CH*CH*bns;
    L.bx=bns; R.bx=bns;
    float cfL=fast_speed(L,bns), cfR=fast_speed(R,bns);
    float SL=fminf(fminf(L.u-cfL,R.u-cfR),0.f), SR=fmaxf(fmaxf(L.u+cfL,R.u+cfR),0.f);
    Cons FL=phys_flux_x(L), FR=phys_flux_x(R), UL=toCons(L), UR=toCons(R), F;
    float ihd=1.f/(SR-SL);
    F.r =(SR*FL.r -SL*FR.r +SL*SR*(UR.r -UL.r ))*ihd;
    F.mx=(SR*FL.mx-SL*FR.mx+SL*SR*(UR.mx-UL.mx))*ihd;
    F.my=(SR*FL.my-SL*FR.my+SL*SR*(UR.my-UL.my))*ihd;
    F.mz=(SR*FL.mz-SL*FR.mz+SL*SR*(UR.mz-UL.mz))*ihd;
    F.E =(SR*FL.E -SL*FR.E +SL*SR*(UR.E -UL.E ))*ihd;
    F.by=(SR*FL.by-SL*FR.by+SL*SR*(UR.by-UL.by))*ihd;
    F.bz=(SR*FL.bz-SL*FR.bz+SL*SR*(UR.bz-UL.bz))*ihd;
    F.bx=fbn; F.ps=fpsi;
    Cons FF=rot_flux_from(F,dir);
#if NSP>0
    #pragma unroll
    for(int i=0;i<NSP;i++) FF.s[i] = FF.r>=0.f ? FF.r*Lq.s[i] : FF.r*Rq.s[i];   // CMA species flux
#endif
    return FF;
}

struct Ptrs { float* v[NV]; };

__global__ void __launch_bounds__(THREADS) march(Ptrs q, Ptrs o){
    __shared__ __half sh[NV][PLANES][GX*GY];
    const int tid=threadIdx.x;
    const int tx=tid%OX, ty=tid/OX;
    const int x0=blockIdx.x*OX, y0=blockIdx.y*OY;
    const int li=tx+GHOST, lj=ty+GHOST;
    auto loadPlane=[&](int k){
        int s=ring(k), gk=wrap(k,NZ);
        for(int c=tid;c<GX*GY;c+=THREADS){
            int lx=c%GX, ly=c/GX;
            int gi=wrap(x0-GHOST+lx,NX), gj=wrap(y0-GHOST+ly,NY);
            size_t g=gidx(gi,gj,gk);
            #pragma unroll
            for(int v=0;v<9;v++) sh[v][s][c]=__float2half(q.v[v][g]);   // conserved
#if NSP>0
            float ir=1.f/fmaxf(q.v[0][g],SMALLR);
            #pragma unroll
            for(int i=0;i<NSP;i++) sh[9+i][s][c]=__float2half(q.v[9+i][g]*ir);   // species FRACTION
#endif
        }
    };
    for(int k=-2;k<=2;k++) loadPlane(k);
    __syncthreads();
#define CR(di,dj,dk,v) __half2float(sh[v][ring(k+(dk))][(li+(di))+GX*(lj+(dj))])
#if NSP>0
#define GP(di,dj,dk) ({ Prim _q=prim_from_cons(CR(di,dj,dk,0),CR(di,dj,dk,1),CR(di,dj,dk,2),CR(di,dj,dk,3),CR(di,dj,dk,4),CR(di,dj,dk,5),CR(di,dj,dk,6),CR(di,dj,dk,7),CR(di,dj,dk,8)); \
    _Pragma("unroll") for(int _i=0;_i<NSP;_i++) _q.s[_i]=CR(di,dj,dk,9+_i); _q; })
#else
#define GP(di,dj,dk) prim_from_cons(CR(di,dj,dk,0),CR(di,dj,dk,1),CR(di,dj,dk,2),CR(di,dj,dk,3),CR(di,dj,dk,4),CR(di,dj,dk,5),CR(di,dj,dk,6),CR(di,dj,dk,7),CR(di,dj,dk,8))
#endif
    for(int k=0;k<NZ;k++){
        Prim c0=GP(0,0,0);
        Cons U=toCons(c0);
#ifndef MEMFLOOR
        Cons div={0,0,0,0,0,0,0,0,0};
        #pragma unroll
        for(int d=0;d<3;d++){
            int ox=(d==0),oy=(d==1),oz=(d==2);
            Prim cm2=GP(-2*ox,-2*oy,-2*oz), cm1=GP(-ox,-oy,-oz),
                 cp1=GP(ox,oy,oz),          cp2=GP(2*ox,2*oy,2*oz);
            Prim sm1=slopeP(cm2,cm1,c0), s0=slopeP(cm1,c0,cp1), sp1=slopeP(c0,cp1,cp2);
            Prim mhm=hanc1d(cm1,sm1,d), mh0=hanc1d(c0,s0,d), mhp=hanc1d(cp1,sp1,d);
#ifdef PPM
            Cons Fm=hllMHD(padd_pred(ppm_edge(cm2,cm1,c0,+1.f),mhm,cm1), padd_pred(ppm_edge(cm1,c0,cp1,-1.f),mh0,c0), d);
            Cons Fp=hllMHD(padd_pred(ppm_edge(cm1,c0,cp1,+1.f),mh0,c0), padd_pred(ppm_edge(c0,cp1,cp2,-1.f),mhp,cp1), d);
#else
            Cons Fm=hllMHD(edge(mhm,sm1,+1.f), edge(mh0,s0,-1.f), d);
            Cons Fp=hllMHD(edge(mh0,s0,+1.f),  edge(mhp,sp1,-1.f), d);
#endif
            div.r+=Fp.r-Fm.r; div.mx+=Fp.mx-Fm.mx; div.my+=Fp.my-Fm.my; div.mz+=Fp.mz-Fm.mz;
            div.E+=Fp.E-Fm.E; div.bx+=Fp.bx-Fm.bx; div.by+=Fp.by-Fm.by; div.bz+=Fp.bz-Fm.bz; div.ps+=Fp.ps-Fm.ps;
#if NSP>0
            #pragma unroll
            for(int i=0;i<NSP;i++) div.s[i]+=Fp.s[i]-Fm.s[i];
#endif
        }
        U.r-=DTDX*div.r; U.mx-=DTDX*div.mx; U.my-=DTDX*div.my; U.mz-=DTDX*div.mz; U.E-=DTDX*div.E;
        U.bx-=DTDX*div.bx; U.by-=DTDX*div.by; U.bz-=DTDX*div.bz; U.ps-=DTDX*div.ps;
        U.ps*=GLMFAC;   // GLM parabolic source (psi decay)
#if NSP>0
        #pragma unroll
        for(int i=0;i<NSP;i++) U.s[i]-=DTDX*div.s[i];
#endif
#endif
        size_t g=gidx(x0+tx,y0+ty,k);
        o.v[0][g]=U.r; o.v[1][g]=U.mx; o.v[2][g]=U.my; o.v[3][g]=U.mz; o.v[4][g]=U.E;
        o.v[5][g]=U.bx; o.v[6][g]=U.by; o.v[7][g]=U.bz; o.v[8][g]=U.ps;
#if NSP>0
        #pragma unroll
        for(int i=0;i<NSP;i++) o.v[9+i][g]=U.s[i];
#endif
        loadPlane(k+3); __syncthreads();
    }
#undef CR
#undef GP
}

__global__ void init(Ptrs q){
    size_t n=(size_t)NX*NY*NZ, i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float ph=0.001f*(float)(i%911);
    Prim s; s.r=1.f+0.3f*ph; s.u=0.1f*ph; s.v=-0.1f*ph; s.w=0.05f*ph; s.p=2.5f+0.5f*ph;
    s.bx=0.2f+0.1f*ph; s.by=0.1f*ph; s.bz=-0.05f*ph; s.ps=0.f;
    Cons c=toCons(s);
    q.v[0][i]=c.r; q.v[1][i]=c.mx; q.v[2][i]=c.my; q.v[3][i]=c.mz; q.v[4][i]=c.E;
    q.v[5][i]=c.bx; q.v[6][i]=c.by; q.v[7][i]=c.bz; q.v[8][i]=c.ps;
#if NSP>0
    #pragma unroll
    for(int j=0;j<NSP;j++) q.v[9+j][i]=c.r*(0.2f+0.3f*ph);   // rho*X
#endif
}

#ifdef AS_LIB
// Julia bridge — same ABI as march_bridge/MarchBridge.jl but NV=9 (GLM-MHD).
extern "C" {
int march_nv(){ return NV; }
int march_nx(){ return NX; }
int march_nsp(){ return NSP; }
void march_set_dtdx(float v){ cudaMemcpyToSymbol(DTDX, &v, sizeof(float)); }
void march_set_glm(float ch,float fac){ cudaMemcpyToSymbol(CH,&ch,sizeof(float)); cudaMemcpyToSymbol(GLMFAC,&fac,sizeof(float)); }
void march_set_gamma(float g){ cudaMemcpyToSymbol(GAMMA, &g, sizeof(float)); }
double march_run_dev(float* const* qv, float* const* ov, int nsteps){
    size_t n=(size_t)NX*NY*NZ; Ptrs q,o;
    for(int v=0;v<NV;v++){ q.v[v]=qv[v]; o.v[v]=ov[v]; }
    dim3 grid(NX/OX, NY/OY);
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for(int it=0;it<nsteps;it++){ march<<<grid,THREADS>>>(q,o); Ptrs t=q;q=o;o=t; }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    if(q.v[0]!=qv[0]) for(int v=0;v<NV;v++) cudaMemcpy(qv[v], q.v[v], n*4, cudaMemcpyDeviceToDevice);
    return (double)ms;
}
}
#endif

#ifndef AS_LIB
int main(){
    size_t n=(size_t)NX*NY*NZ; Ptrs q,o;
    for(int v=0;v<NV;v++){ cudaMalloc(&q.v[v],n*4); cudaMalloc(&o.v[v],n*4); }
    init<<<(n+255)/256,256>>>(q); init<<<(n+255)/256,256>>>(o); cudaDeviceSynchronize();
    dim3 grid(NX/OX, NY/OY);
    size_t shmem=sizeof(__half)*NV*PLANES*GX*GY;
    int nblk=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblk, march, THREADS, 0);
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, march);
    for(int it=0;it<3;it++){ march<<<grid,THREADS>>>(q,o); Ptrs t=q;q=o;o=t; }
    cudaDeviceSynchronize();
    cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess){ printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
    int iters=30;
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for(int it=0;it<iters;it++){ march<<<grid,THREADS>>>(q,o); Ptrs t=q;q=o;o=t; }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    double cells=(double)n*iters, mcs=cells/(ms*1e-3)/1e6, gbs=cells*(NV*4.0*2.0)/(ms*1e-3)/1e9;
    printf("2.5D GLM-MHD march (HANCOCK1D + HLL + Dedner)\n");
    printf("  grid %d^3  tile %dx%d owned  ring %d planes  NV=%d\n", NX,OX,OY,PLANES,NV);
    printf("  regs=%d  shmem=%zuB (%.1fKB)  blocks/SM=%d\n", fa.numRegs, shmem, shmem/1024.0, nblk);
    printf("  %.2f ms / %d iters  =>  %.0f Mcell/s   %.0f GB/s  (%.0f%% of 768 peak)\n",
           ms/iters, iters, mcs, gbs, gbs/768.0*100.0);
    printf("  vs Julia cube GLM-MHD ~1923 pure / 1416 turb Mcell/s\n");
    return 0;
}
#endif
