// spike_25d.cu — throughput probe for a 2.5D unigrid line-march godunov scheme.
//
// Question: on a uniform 480^3 grid (no AMR, no Morton), does a scheme that tiles
// x-y in shared and MARCHES z through a rolling plane-ring (z over-read = 1x, x-y
// over-read = ~1.7x, one barrier per z-advance) beat the production oct-tile kernel?
// Reference points (A6000, fp32, f16 shared tile, PLM, 480^3, measured):
//   production PLM  ~4127 Mcell/s     load+store floor ~5448 Mcell/s
//   DRAM-bandwidth ceiling (2x field, 768 GB/s) ~19200 Mcell/s
//
// Faithful on: SoA flat layout, coalesced x, rolling f16 plane-ring (1x z over-read),
// 2-ghost x-y halo, single barrier per plane-advance, PLM(moncen)+HLL in 3 dirs,
// realistic register/occupancy profile. -DMEMFLOOR skips the flux compute (2.5D mem floor).
//
// RESULT (A6000, 480^3): WITHOUT Hancock (default) 64 regs / 4 blocks-SM / ~7100 Mcell/s
// (+72% over the cube's 4127) — BUT that omits the transverse half-step predictor. With
// -DHANCOCK (faithful MUSCL-Hancock, matches the production cube) registers blow to 227,
// occupancy collapses to 1 block/SM, and it drops to ~2020 Mcell/s — BELOW the staged cube.
// CONCLUSION: the fused 2.5D march wins only for LIGHT kernels; the faithful godunov is
// register-bound (slopes+mh+6 HLL held live per cell), and the cube's staged-through-shared
// structure is register-optimal for it. The march's +72% was a no-Hancock artifact. Same
// verdict reproduced in the Julia prototype (GLMMHDTurb integrator_hydro_march2!: 179 regs,
// ~8x slower than the cube). cf. the earlier integrator_stream! rejection (compute-bound).
//
// BUT (-DHANCOCK1D): the register blow-up is ENTIRELY the TRANSVERSE coupling, not 2nd-order-
// in-time. A normal-only 1D Hancock half-step (uses only the already-computed direction slope,
// no transverse, no shared mh) is 2nd-order in space AND time at **64 regs / 4 blocks-SM /
// ~6680 Mcell/s = +63% over the cube, 87% of the 7649 memfloor**. So a LIGHT 2nd-order fused
// march DOES reach full throughput. OPEN CAVEAT: transverse-free unsplit MUSCL-Hancock has a
// reduced multi-D stability limit (~CFL 1/ndim ≈ 0.33-0.5 in 3D vs the cube's 0.7) and larger
// transverse-error constants — must validate max-stable-CFL + turb statistics before claiming a
// NET win (if CFL>=~0.5 holds it wins; at 0.33 the extra steps ~cancel the throughput gain).
//
// -DDONOR (with HANCOCK1D): donor-cell flux at the x/y block seams -> 1-ghost x-y tile
// (z always 2nd order, no seam). EXACTLY CONSERVATIVE (consistent flux both sides; Julia
// GLMMHDTurb integrator_hydro_lmarch! GH=1 gives rel-dmass=0). Shrinks shared 21.6->17.0 KB
// (-21% over-read) BUT regs 64->72 -> 3 blocks/SM (off the 4-block cliff) -> NET -5% on the
// A6000. The traffic saving is real but the kernel is occupancy-pinned, not DRAM-bound, and
// 4 blocks needs <=64 regs exactly. Likely POSITIVE on a GPU not sitting on that cliff (H200).
//
// -DLLF_FALLBACK (with HANCOCK1D): HLL normally, LLF (Rusanov) at faces with a reconstructed
// interface state below floors. UNIFIED solver shares FL/FR/UL/UR and branches only on the cheap
// combine -> STAYS at 64 regs / 4 blocks (naive bad?llf:hll blows past 64 -> 3 blocks). Cost -2%
// (~6530), exactly conservative, HLL-level accuracy (sigma_s tracks HLL; LLF fires only at rare
// low-rho faces, not over-diffuse pure LLF). Julia twin GLMMHDTurb riem5 Val{:fb}. The light
// scheme is CFL/accuracy-VALIDATED on Mach~4-12 driven hydro turb: stable to CFL=1.0 (no penalty
// vs the cube, no 1/ndim collapse), sigma_s matches; HLL already robust to Mach 12.
//
// -DSCALARS (with HANCOCK1D): +2 passive scalars (7-var, advected as rho*s). regs 64->80,
// shared 21.6->30.2KB, occupancy 4->3 blocks -> -27% (6700->4885 Mcell/s, still +19% over the
// 5-var cube). The cost is ENTIRELY the +40% data: per-VARIABLE throughput is flat (~+0%), the
// scalar advection compute is free. ~-14%/scalar; the 4->3 occupancy drop is the A6000 cliff
// (H200 would hold higher occupancy -> closer to the pure -29% data floor).
//
// -DCMA (with SCALARS): consistent multi-fluid advection -- species ride the single mass flux
// (F_species = F_mass * X_upwind) instead of a Riemann solve per species. CONSERVES sum X_i = 1
// EXACTLY, register-light, count-independent compute. +4% over HLL-per-scalar (4875->5060, -24%
// vs 5-var); regs/occupancy unchanged (data-bound at 7-var, species in the 30KB tile -> 3 blocks).
// FOR SPECIES FRACTIONS (chemistry, 30 dex): store uint16-log10 (0.1%/ULP over [-30,0] dex) in
// the GLOBAL arrays -> the tile fills FROM global so this halves the dominant species traffic ->
// ~-7%/scalar (vs -14% fp32). Reconstruct in log space (d(logX)/dt=-v.grad(logX) identically for
// passive advection -> positivity-preserving, natural over 30 dex; decode exp10 only for the flux).
// Combined CMA + uint16-log10: ~-12..-15% for 2 species (vs -27% naive), exactly conservative.
//
// -DU16SP (MEASURED): species as uint16 log2(X) in GLOBAL, decoded to log2 in the f16 tile,
// CMA flux with exp2f decode, re-encoded on store. 2 species: 5340 Mcell/s = -20% vs 5-var
// (vs fp32-CMA -24%, fp32-HLL -27%) -> +5% over fp32, ~-10%/scalar. The traffic saving (-14%
// bytes) only partly lands because we're pinned at 3 blocks/SM (50% occ) -> latency- not
// bandwidth-limited, and exp2/log2 cost a little. THE REAL POINT IS CORRECTNESS: a linear
// fraction of 1e-20 UNDERFLOWS the f16 tile to 0 (trace species vanish) -- storing log2 is
// REQUIRED for 30-dex fractions, not optional; exp2f only at the flux. uint16-log2 is the only
// CORRECT option AND +5% AND half the checkpoint size. On H200 (holds 4 blocks at 7-var ->
// bandwidth-bound) the -14% traffic saving would translate much closer to fully.
//
// -DU16STATE (MEASURED + ACCURACY-VALIDATED): the entire hydro state in uint16, stored as PRIMITIVE
// (rho/p as log2 over [-32,32]; velocity linear over [-8,8], bounded by Mach NOT rho*v). Tile holds
// primitive directly (GP=make_prim, no per-read c2p); store does c2p->encode. 7410 Mcell/s = +11%
// over fp32 (+81% over cube), 64 regs / 4 blocks. [The earlier +19% used CONSERVED momenta-linear,
// which is +8% faster (no store-side c2p divide) but PHYSICALLY BROKEN: linear momenta destroy the
// turbulence -- spectrum blew up 1714x at k=90, -2% mass. So the accurate version pays ~8% for the
// c2p.] Accuracy (Julia quantization-roundtrip, driven Mach-3.4 turb): spectrum matches fp32 to
// 1.01-1.05 across k=4..90, sigma_s -6%, mass drift -0.4% (irreducible log2-rho per-step requantize;
// mitigate by quantizing every K steps). Halves global storage+checkpoints, enables ~1216^3 on A6000.
//
// -DF16STATE (MEASURED): same primitive state but stored as plain f16 (rho,v,p all __half; direct
// copy to the f16 tile, no codec). +11% over fp32 -- IDENTICAL throughput to uint16-log2-primitive
// (both 2B, bandwidth-bound, store-side c2p divide dominates; the exp2/log2 vs direct decode is
// hidden). BETTER accuracy: sigma_s 1.277 vs uint16-log2's 1.150 (fp32=1.233) -> less diffusive;
// same -0.17% mass drift, same spectrum. So FOR THE HYDRO STATE f16 wins (simpler=native tile format,
// less diffusive, equal speed); use uint16-log2 ONLY for extreme range (species 30dex; f16 spans 9).
//
// build:  nvcc -arch=sm_86 -O3 --use_fast_math -o spike_25d gpu/spike_25d.cu
//         nvcc -arch=sm_86 -O3 --use_fast_math -DMEMFLOOR -o spike_25d_mf gpu/spike_25d.cu

#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifndef NX
#define NX 480
#endif
#ifndef NY
#define NY 480
#endif
#ifndef NZ
#define NZ 480
#endif
#ifdef U16SP                 // uint16 log2-encoded species fractions (implies the scalar machinery)
#ifndef SCALARS
#define SCALARS
#endif
#ifndef CMA
#define CMA                  // species fractions -> consistent multi-fluid advection
#endif
#endif
#ifdef SCALARS
#define NV 7           // 5 hydro + 2 passive scalars (advected, conserved as rho*s)
#else
#define NV 5
#endif

#ifndef OX
#define OX 32          // owned cells per block in x  (== blockDim.x lane group)
#endif
#ifndef OY
#define OY 8           // owned cells per block in y
#endif
#ifdef DONOR
#define GHOST 1        // donor-cell at x/y block seams -> 1-ghost x-y tile
#else
#define GHOST 2        // full 2nd-order -> 2-ghost x-y tile
#endif
#define GX (OX+2*GHOST)
#define GY (OY+2*GHOST)
#define PLANES 5       // rolling ring k-2..k+2 (z always 2nd order, no seam)
#ifndef THREADS
#define THREADS (OX*OY)   // one thread per owned column
#endif

__device__ __forceinline__ size_t gidx(int i,int j,int k){
    return (size_t)i + (size_t)NX*((size_t)j + (size_t)NY*(size_t)k);
}
__device__ __forceinline__ int wrap(int i,int N){ int m=i%N; return m<0?m+N:m; }
__device__ __forceinline__ int ring(int m){ int r=m%PLANES; return r<0?r+PLANES:r; }

struct Ptrs { float* v[NV];
#ifdef U16SP
    unsigned short* su[2];   // species fractions, uint16 log2-encoded global storage (2B vs 4B)
#endif
#ifdef U16STATE
    unsigned short* h[5];    // entire hydro state in uint16 (rho/E log2, momenta linear)
#endif
#ifdef F16STATE
    __half* hf[5];           // entire hydro state in f16 (primitive: rho,vx,vy,vz,p)
#endif
};
#if defined(U16SP) || defined(U16STATE)
// uint16 <-> log2(value) over [-110,0] log2 units (~5e-4 dex/ULP)
__device__ __forceinline__ float dec_log2(unsigned short u){ return -110.f + (float)u*(110.f/65535.f); }
__device__ __forceinline__ unsigned short enc_log2(float l2){ float t=(l2+110.f)*(65535.f/110.f); t=fminf(fmaxf(t,0.f),65535.f); return (unsigned short)(t+0.5f); }
#endif
#ifdef U16STATE
#define VMAX 8.f            // linear uint16 range for the signed VELOCITY (bounded by Mach, not rho*v)
__device__ __forceinline__ float dec_vel(unsigned short u){ return -VMAX + (float)u*(2.f*VMAX/65535.f); }
__device__ __forceinline__ unsigned short enc_vel(float x){ float t=(fminf(fmaxf(x,-VMAX),VMAX)+VMAX)*(65535.f/(2.f*VMAX)); return (unsigned short)(t+0.5f); }
// log2 codec for rho/p over [-32,+32] log2 (can exceed 1, unlike species fractions); ~3e-4 dex/ULP
__device__ __forceinline__ float dec_log2s(unsigned short u){ return -32.f + (float)u*(64.f/65535.f); }
__device__ __forceinline__ unsigned short enc_log2s(float l2){ float t=(l2+32.f)*(65535.f/64.f); t=fminf(fmaxf(t,0.f),65535.f); return (unsigned short)(t+0.5f); }
#endif
struct Prim { float r,u,v,w,p;
#ifdef SCALARS
    float s0,s1;
#endif
};
struct Cons { float r,mx,my,mz,E;
#ifdef SCALARS
    float c0,c1;
#endif
};

__constant__ float GAMMA=1.4f, DTDX=0.05f, SMALLR=1e-6f, SMALLP=1e-6f;

#ifdef SCALARS
__device__ __forceinline__ Prim prim_from_cons(float r,float mx,float my,float mz,float E,float cs0,float cs1){
    float ir = 1.f/fmaxf(r,SMALLR);
    Prim q; q.r=fmaxf(r,SMALLR); q.u=mx*ir; q.v=my*ir; q.w=mz*ir;
    float ke=0.5f*q.r*(q.u*q.u+q.v*q.v+q.w*q.w);
    q.p=fmaxf((GAMMA-1.f)*(E-ke),SMALLP);
#ifdef U16SP
    q.s0=cs0; q.s1=cs1;          // tile holds log2(X) directly (intensive, reconstructed in log)
#else
    q.s0=cs0*ir; q.s1=cs1*ir;
#endif
    return q;
}
#else
__device__ __forceinline__ Prim prim_from_cons(float r,float mx,float my,float mz,float E){
    float ir = 1.f/fmaxf(r,SMALLR);
    Prim q; q.r=fmaxf(r,SMALLR); q.u=mx*ir; q.v=my*ir; q.w=mz*ir;
    float ke=0.5f*q.r*(q.u*q.u+q.v*q.v+q.w*q.w);
    q.p=fmaxf((GAMMA-1.f)*(E-ke),SMALLP);
    return q;
}
#endif
#if defined(U16STATE) || defined(F16STATE)
__device__ __forceinline__ Prim make_prim(float r,float u,float v,float w,float p){ Prim q; q.r=fmaxf(r,SMALLR); q.u=u; q.v=v; q.w=w; q.p=fmaxf(p,SMALLP); return q; }
#endif
__device__ __forceinline__ float moncen(float a,float b,float c){
    // monotonized central slope (slope_type=2)
    float dl=b-a, dr=c-b, dc=0.5f*(c-a);
    if(dl*dr<=0.f) return 0.f;
    float s = dl>0.f?1.f:-1.f;
    return s*fminf(fabsf(dc),fminf(2.f*fabsf(dl),2.f*fabsf(dr)));
}
__device__ __forceinline__ Prim slopeP(Prim a,Prim b,Prim c){
    Prim s; s.r=moncen(a.r,b.r,c.r); s.u=moncen(a.u,b.u,c.u); s.v=moncen(a.v,b.v,c.v);
    s.w=moncen(a.w,b.w,c.w); s.p=moncen(a.p,b.p,c.p);
#ifdef SCALARS
    s.s0=moncen(a.s0,b.s0,c.s0); s.s1=moncen(a.s1,b.s1,c.s1);
#endif
    return s;
}
__device__ __forceinline__ Prim edge(Prim b,Prim s,float sign){
    Prim e; e.r=b.r+sign*0.5f*s.r; e.u=b.u+sign*0.5f*s.u; e.v=b.v+sign*0.5f*s.v;
    e.w=b.w+sign*0.5f*s.w; e.p=b.p+sign*0.5f*s.p;
    e.r=fmaxf(e.r,SMALLR); e.p=fmaxf(e.p,SMALLP);
#ifdef SCALARS
    e.s0=b.s0+sign*0.5f*s.s0; e.s1=b.s1+sign*0.5f*s.s1;
#endif
    return e;
}
__device__ __forceinline__ Cons primFlux(Prim q,int dir){
    float un = dir==0?q.u: dir==1?q.v: q.w;
    float E = q.p/(GAMMA-1.f)+0.5f*q.r*(q.u*q.u+q.v*q.v+q.w*q.w);
    Cons f;
    f.r  = q.r*un;
    f.mx = q.r*q.u*un + (dir==0?q.p:0.f);
    f.my = q.r*q.v*un + (dir==1?q.p:0.f);
    f.mz = q.r*q.w*un + (dir==2?q.p:0.f);
    f.E  = (E+q.p)*un;
#ifdef SCALARS
    f.c0 = q.r*un*q.s0; f.c1 = q.r*un*q.s1;   // passive scalar flux = mass flux * s
#endif
    return f;
}
__device__ __forceinline__ Cons toCons(Prim q){
    Cons c; c.r=q.r; c.mx=q.r*q.u; c.my=q.r*q.v; c.mz=q.r*q.w;
    c.E=q.p/(GAMMA-1.f)+0.5f*q.r*(q.u*q.u+q.v*q.v+q.w*q.w);
#ifdef U16SP
    c.c0=q.r*exp2f(q.s0); c.c1=q.r*exp2f(q.s1);   // conserved rho*X from log2(X)
#elif defined(SCALARS)
    c.c0=q.r*q.s0; c.c1=q.r*q.s1;
#endif
    return c;
}
// transverse Hancock half-step predictor (matches integrator_hydro!'s hancock5): mh = m0 + dt/2*src
__device__ __forceinline__ Prim hanc(Prim m0,Prim xm,Prim xp,Prim ym,Prim yp,Prim zm,Prim zp,float dtdx,float g){
    Prim sx,sy,sz;
    sx.r=moncen(xm.r,m0.r,xp.r);sx.u=moncen(xm.u,m0.u,xp.u);sx.v=moncen(xm.v,m0.v,xp.v);sx.w=moncen(xm.w,m0.w,xp.w);sx.p=moncen(xm.p,m0.p,xp.p);
    sy.r=moncen(ym.r,m0.r,yp.r);sy.u=moncen(ym.u,m0.u,yp.u);sy.v=moncen(ym.v,m0.v,yp.v);sy.w=moncen(ym.w,m0.w,yp.w);sy.p=moncen(ym.p,m0.p,yp.p);
    sz.r=moncen(zm.r,m0.r,zp.r);sz.u=moncen(zm.u,m0.u,zp.u);sz.v=moncen(zm.v,m0.v,zp.v);sz.w=moncen(zm.w,m0.w,zp.w);sz.p=moncen(zm.p,m0.p,zp.p);
    float dv=sx.u+sy.v+sz.w; float ir=1.f/m0.r; Prim mh;
    mh.r=m0.r+dtdx*0.5f*(-m0.u*sx.r-m0.v*sy.r-m0.w*sz.r-dv*m0.r);
    mh.u=m0.u+dtdx*0.5f*(-m0.u*sx.u-m0.v*sy.u-m0.w*sz.u-sx.p*ir);
    mh.v=m0.v+dtdx*0.5f*(-m0.u*sx.v-m0.v*sy.v-m0.w*sz.v-sy.p*ir);
    mh.w=m0.w+dtdx*0.5f*(-m0.u*sx.w-m0.v*sy.w-m0.w*sz.w-sz.p*ir);
    mh.p=m0.p+dtdx*0.5f*(-m0.u*sx.p-m0.v*sy.p-m0.w*sz.p-dv*g*m0.p);
    return mh;
}
// monotonized single-zone parabolic interface value (CW84), one component.
// sgn>0 -> +face, sgn<0 -> -face. 3-point stencil (qm,q0,qp): parabola + monotonize.
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
    e.w=ppm1(a.w,b.w,c.w,sgn); e.p=ppm1(a.p,b.p,c.p,sgn); return e;
}
// NORMAL-only 1D Hancock half-step predictor for direction dir: uses ONLY the
// already-computed direction-dir slope s (no transverse slopes, no cross-dir state).
// Primitive quasilinear A_dir*s; 2nd order in time per direction, transverse-free.
__device__ __forceinline__ Prim hanc1d(Prim m,Prim s,int dir,float dtdx,float g){
    float un = dir==0?m.u : dir==1?m.v : m.w;
    float sn = dir==0?s.u : dir==1?s.v : s.w;
    float ir = 1.f/m.r; float h=0.5f*dtdx;
    Prim mh;
    mh.r = m.r - h*(un*s.r + m.r*sn);
    mh.u = m.u - h*(un*s.u + (dir==0? s.p*ir:0.f));
    mh.v = m.v - h*(un*s.v + (dir==1? s.p*ir:0.f));
    mh.w = m.w - h*(un*s.w + (dir==2? s.p*ir:0.f));
    mh.p = m.p - h*(un*s.p + g*m.p*sn);
#ifdef SCALARS
    mh.s0 = m.s0 - h*un*s.s0; mh.s1 = m.s1 - h*un*s.s1;   // passive advection
#endif
    return mh;
}
__device__ __forceinline__ Cons hll(Prim L,Prim R,int dir){
    float unL=dir==0?L.u:dir==1?L.v:L.w, unR=dir==0?R.u:dir==1?R.v:R.w;
    float aL=sqrtf(GAMMA*L.p/L.r), aR=sqrtf(GAMMA*R.p/R.r);
    float sL=fminf(unL-aL,unR-aR), sR=fmaxf(unL+aL,unR+aR);
    Cons FL=primFlux(L,dir), FR=primFlux(R,dir), UL=toCons(L), UR=toCons(R), F;
    if(sL>=0.f) return FL;
    if(sR<=0.f) return FR;
    float inv=1.f/(sR-sL);
    F.r =(sR*FL.r -sL*FR.r +sL*sR*(UR.r -UL.r ))*inv;
    F.mx=(sR*FL.mx-sL*FR.mx+sL*sR*(UR.mx-UL.mx))*inv;
    F.my=(sR*FL.my-sL*FR.my+sL*sR*(UR.my-UL.my))*inv;
    F.mz=(sR*FL.mz-sL*FR.mz+sL*sR*(UR.mz-UL.mz))*inv;
    F.E =(sR*FL.E -sL*FR.E +sL*sR*(UR.E -UL.E ))*inv;
#ifdef SCALARS
#ifdef CMA
#ifdef U16SP
    F.c0 = F.r>=0.f ? F.r*exp2f(L.s0) : F.r*exp2f(R.s0);   // decode log2-edge -> ride mass flux
    F.c1 = F.r>=0.f ? F.r*exp2f(L.s1) : F.r*exp2f(R.s1);
#else
    F.c0 = F.r>=0.f ? F.r*L.s0 : F.r*R.s0;   // consistent multi-fluid advection: ride the mass flux
    F.c1 = F.r>=0.f ? F.r*L.s1 : F.r*R.s1;
#endif
#else
    F.c0=(sR*FL.c0-sL*FR.c0+sL*sR*(UR.c0-UL.c0))*inv;
    F.c1=(sR*FL.c1-sL*FR.c1+sL*sR*(UR.c1-UL.c1))*inv;
#endif
#endif
    return F;
}

// Unified HLL / LLF-fallback: share FL,FR,UL,UR (the expensive part); branch only on
// the cheap combine. LLF (Rusanov, positivity-robust) when a reconstructed interface
// state is below floors, else HLL. Minimizes the register delta of carrying both.
__device__ __forceinline__ Cons riemann_fb(Prim L,Prim R,int dir){
    float unL=dir==0?L.u:dir==1?L.v:L.w, unR=dir==0?R.u:dir==1?R.v:R.w;
    float aL=sqrtf(GAMMA*L.p/L.r), aR=sqrtf(GAMMA*R.p/R.r);
    Cons FL=primFlux(L,dir), FR=primFlux(R,dir), UL=toCons(L), UR=toCons(R), F;
    if(L.r<1e-3f||L.p<1e-3f||R.r<1e-3f||R.p<1e-3f){           // LLF fallback (robust)
        float s=fmaxf(fabsf(unL)+aL,fabsf(unR)+aR);
        F.r =0.5f*(FL.r +FR.r )-0.5f*s*(UR.r -UL.r );
        F.mx=0.5f*(FL.mx+FR.mx)-0.5f*s*(UR.mx-UL.mx);
        F.my=0.5f*(FL.my+FR.my)-0.5f*s*(UR.my-UL.my);
        F.mz=0.5f*(FL.mz+FR.mz)-0.5f*s*(UR.mz-UL.mz);
        F.E =0.5f*(FL.E +FR.E )-0.5f*s*(UR.E -UL.E );
#ifdef SCALARS
        F.c0=0.5f*(FL.c0+FR.c0)-0.5f*s*(UR.c0-UL.c0);
        F.c1=0.5f*(FL.c1+FR.c1)-0.5f*s*(UR.c1-UL.c1);
#endif
        return F;
    }
    float sL=fminf(unL-aL,unR-aR), sR=fmaxf(unL+aL,unR+aR);   // HLL (accurate)
    if(sL>=0.f) return FL;
    if(sR<=0.f) return FR;
    float inv=1.f/(sR-sL);
    F.r =(sR*FL.r -sL*FR.r +sL*sR*(UR.r -UL.r ))*inv;
    F.mx=(sR*FL.mx-sL*FR.mx+sL*sR*(UR.mx-UL.mx))*inv;
    F.my=(sR*FL.my-sL*FR.my+sL*sR*(UR.my-UL.my))*inv;
    F.mz=(sR*FL.mz-sL*FR.mz+sL*sR*(UR.mz-UL.mz))*inv;
    F.E =(sR*FL.E -sL*FR.E +sL*sR*(UR.E -UL.E ))*inv;
#ifdef SCALARS
#ifdef CMA
#ifdef U16SP
    F.c0 = F.r>=0.f ? F.r*exp2f(L.s0) : F.r*exp2f(R.s0);   // decode log2-edge -> ride mass flux
    F.c1 = F.r>=0.f ? F.r*exp2f(L.s1) : F.r*exp2f(R.s1);
#else
    F.c0 = F.r>=0.f ? F.r*L.s0 : F.r*R.s0;   // consistent multi-fluid advection: ride the mass flux
    F.c1 = F.r>=0.f ? F.r*L.s1 : F.r*R.s1;
#endif
#else
    F.c0=(sR*FL.c0-sL*FR.c0+sL*sR*(UR.c0-UL.c0))*inv;
    F.c1=(sR*FL.c1-sL*FR.c1+sL*sR*(UR.c1-UL.c1))*inv;
#endif
#endif
    return F;
}
#ifdef LLF_FALLBACK
#define RS(L,R,d) riemann_fb(L,R,d)
#else
#define RS(L,R,d) hll(L,R,d)
#endif

__global__ void __launch_bounds__(THREADS) march(Ptrs q, Ptrs o){
    __shared__ __half sh[NV][PLANES][GX*GY];
    const int tid = threadIdx.x;
    const int tx = tid % OX, ty = tid / OX;     // owned column (0..OX,0..OY)
    const int x0 = blockIdx.x*OX, y0 = blockIdx.y*OY;
    const int li = tx+GHOST, lj = ty+GHOST;     // local shared coords of owned cell

    // load plane at absolute k into ring slot ring(k)
    auto loadPlane = [&](int k){
        int s = ring(k), gk = wrap(k,NZ);
        for(int c=tid; c<GX*GY; c+=THREADS){
            int lx=c%GX, ly=c/GX;
            int gi=wrap(x0-GHOST+lx,NX), gj=wrap(y0-GHOST+ly,NY);
            size_t g=gidx(gi,gj,gk);
#ifdef U16STATE
            sh[0][s][c]=__float2half(exp2f(dec_log2s(q.h[0][g])));   // rho   (log2) -> tile holds PRIMITIVE
            sh[1][s][c]=__float2half(dec_vel(q.h[1][g]));           // vx    (linear, tight)
            sh[2][s][c]=__float2half(dec_vel(q.h[2][g]));           // vy
            sh[3][s][c]=__float2half(dec_vel(q.h[3][g]));           // vz
            sh[4][s][c]=__float2half(exp2f(dec_log2s(q.h[4][g])));   // p     (log2)
#elif defined(F16STATE)
            sh[0][s][c]=q.hf[0][g]; sh[1][s][c]=q.hf[1][g]; sh[2][s][c]=q.hf[2][g]; sh[3][s][c]=q.hf[3][g]; sh[4][s][c]=q.hf[4][g];  // direct f16 -> primitive tile
#else
            #pragma unroll
            for(int v=0;v<5;v++) sh[v][s][c]=__float2half(q.v[v][g]);
#endif
#ifdef U16SP
            sh[5][s][c]=__float2half(dec_log2(q.su[0][g]));   // uint16 -> log2(X) in tile
            sh[6][s][c]=__float2half(dec_log2(q.su[1][g]));
#elif defined(SCALARS)
            sh[5][s][c]=__float2half(q.v[5][g]); sh[6][s][c]=__float2half(q.v[6][g]);
#endif
        }
    };
    // prime ring k=-2..2
    for(int k=-2;k<=2;k++) loadPlane(k);
    __syncthreads();

#define CR(di,dj,dk,v) __half2float(sh[v][ring(k+(dk))][(li+(di))+GX*(lj+(dj))])
#ifdef SCALARS
#define GP(di,dj,dk) prim_from_cons(CR(di,dj,dk,0),CR(di,dj,dk,1),CR(di,dj,dk,2),CR(di,dj,dk,3),CR(di,dj,dk,4),CR(di,dj,dk,5),CR(di,dj,dk,6))
#elif defined(U16STATE) || defined(F16STATE)
#define GP(di,dj,dk) make_prim(CR(di,dj,dk,0),CR(di,dj,dk,1),CR(di,dj,dk,2),CR(di,dj,dk,3),CR(di,dj,dk,4))   // tile holds primitive
#else
#define GP(di,dj,dk) prim_from_cons(CR(di,dj,dk,0),CR(di,dj,dk,1),CR(di,dj,dk,2),CR(di,dj,dk,3),CR(di,dj,dk,4))
#endif

    for(int k=0;k<NZ;k++){
        Prim c0 = GP(0,0,0);
        Cons U = toCons(c0);
#ifndef MEMFLOOR
        Cons div={0,0,0,0,0};
        #pragma unroll
        for(int d=0; d<3; d++){
            int ox=(d==0), oy=(d==1), oz=(d==2);
#ifdef DONOR
            // 1-ghost x-y tile: donor-cell flux at x/y block seams (1st order, both sides
            // -> consistent -> conserved), 2nd-order 1D-Hancock interior + all of z.
            int lob=(d==0&&tx==0)||(d==1&&ty==0);          // low face is a block seam
            int hib=(d==0&&tx==OX-1)||(d==1&&ty==OY-1);    // high face is a block seam
            Prim cm1=GP(-ox,-oy,-oz), cp1=GP(ox,oy,oz);
            Prim cm2= lob ? c0 : GP(-2*ox,-2*oy,-2*oz);    // ternary skips the OOB read at the seam
            Prim cp2= hib ? c0 : GP(2*ox,2*oy,2*oz);
            Prim sm1=slopeP(cm2,cm1,c0), s0=slopeP(cm1,c0,cp1), sp1=slopeP(c0,cp1,cp2);
            Cons Fm = lob ? hll(cm1,c0,d)
                          : hll(edge(hanc1d(cm1,sm1,d,DTDX,GAMMA),sm1,+1.f), edge(hanc1d(c0,s0,d,DTDX,GAMMA),s0,-1.f), d);
            Cons Fp = hib ? hll(c0,cp1,d)
                          : hll(edge(hanc1d(c0,s0,d,DTDX,GAMMA),s0,+1.f), edge(hanc1d(cp1,sp1,d,DTDX,GAMMA),sp1,-1.f), d);
#else
            Prim cm2=GP(-2*ox,-2*oy,-2*oz), cm1=GP(-ox,-oy,-oz),
                 cp1=GP(ox,oy,oz),          cp2=GP(2*ox,2*oy,2*oz);
            Prim sm1=slopeP(cm2,cm1,c0), s0=slopeP(cm1,c0,cp1), sp1=slopeP(c0,cp1,cp2);
#ifdef HANCOCK
            // transverse-Hancock edges: mh(cell) +/- 0.5*slope_d  (faithful MUSCL-Hancock, like the cube)
            #define MH(ai,aj,ak) hanc(GP(ai,aj,ak),GP((ai)-1,aj,ak),GP((ai)+1,aj,ak),GP(ai,(aj)-1,ak),GP(ai,(aj)+1,ak),GP(ai,aj,(ak)-1),GP(ai,aj,(ak)+1),DTDX,GAMMA)
            Prim mhm=MH(-ox,-oy,-oz), mh0=MH(0,0,0), mhp=MH(ox,oy,oz);
            #undef MH
            Cons Fm = hll(edge(mhm,sm1,+1.f), edge(mh0,s0,-1.f), d);
            Cons Fp = hll(edge(mh0,s0,+1.f),  edge(mhp,sp1,-1.f), d);
#elif defined(HANCOCK1D)
            // normal-only 1D Hancock: 2nd order in space AND time, transverse-free (light)
            Prim mhm=hanc1d(cm1,sm1,d,DTDX,GAMMA), mh0=hanc1d(c0,s0,d,DTDX,GAMMA), mhp=hanc1d(cp1,sp1,d,DTDX,GAMMA);
            Cons Fm = RS(edge(mhm,sm1,+1.f), edge(mh0,s0,-1.f), d);
            Cons Fp = RS(edge(mh0,s0,+1.f),  edge(mhp,sp1,-1.f), d);
#elif defined(PPM1D)
            // parabolic-edge PPM (monotonized 3-pt parabola) + normal-only 1D Hancock predictor.
            // edges built from the parabola; predictor uses the parabolic slope (eR-eL).
            Prim eRm=ppm_edge(cm2,cm1,c0,+1.f);                 // +face of cm1
            Prim eLc=ppm_edge(cm1,c0,cp1,-1.f), eRc=ppm_edge(cm1,c0,cp1,+1.f);  // -/+ faces of c0
            Prim eLp=ppm_edge(c0,cp1,cp2,-1.f);                 // -face of cp1
            // cheap normal-only 1D Hancock predictor correction added to the parabolic edges
            Prim mhm=hanc1d(cm1,sm1,d,DTDX,GAMMA), mh0=hanc1d(c0,s0,d,DTDX,GAMMA), mhp=hanc1d(cp1,sp1,d,DTDX,GAMMA);
            Prim Lm={eRm.r+mhm.r-cm1.r,eRm.u+mhm.u-cm1.u,eRm.v+mhm.v-cm1.v,eRm.w+mhm.w-cm1.w,eRm.p+mhm.p-cm1.p};
            Prim Lc={eLc.r+mh0.r-c0.r,eLc.u+mh0.u-c0.u,eLc.v+mh0.v-c0.v,eLc.w+mh0.w-c0.w,eLc.p+mh0.p-c0.p};
            Prim Rc={eRc.r+mh0.r-c0.r,eRc.u+mh0.u-c0.u,eRc.v+mh0.v-c0.v,eRc.w+mh0.w-c0.w,eRc.p+mh0.p-c0.p};
            Prim Rp={eLp.r+mhp.r-cp1.r,eLp.u+mhp.u-cp1.u,eLp.v+mhp.v-cp1.v,eLp.w+mhp.w-cp1.w,eLp.p+mhp.p-cp1.p};
            Cons Fm = hll(Lm, Lc, d);
            Cons Fp = hll(Rc, Rp, d);
#else
            Cons Fm = hll(edge(cm1,sm1,+1.f), edge(c0,s0,-1.f), d);   // face k-1/2
            Cons Fp = hll(edge(c0,s0,+1.f),   edge(cp1,sp1,-1.f), d); // face k+1/2
#endif
#endif // DONOR
            div.r+=Fp.r-Fm.r; div.mx+=Fp.mx-Fm.mx; div.my+=Fp.my-Fm.my;
            div.mz+=Fp.mz-Fm.mz; div.E+=Fp.E-Fm.E;
#ifdef SCALARS
            div.c0+=Fp.c0-Fm.c0; div.c1+=Fp.c1-Fm.c1;
#endif
        }
        U.r-=DTDX*div.r; U.mx-=DTDX*div.mx; U.my-=DTDX*div.my; U.mz-=DTDX*div.mz; U.E-=DTDX*div.E;
#ifdef SCALARS
        U.c0-=DTDX*div.c0; U.c1-=DTDX*div.c1;
#endif
#endif
        // write owned cell
        size_t g=gidx(x0+tx,y0+ty,k);
#ifdef U16STATE
        { float ir=1.f/fmaxf(U.r,SMALLR);                          // c2p -> store PRIMITIVE
          float vx=U.mx*ir, vy=U.my*ir, vz=U.mz*ir;
          float ke=0.5f*U.r*(vx*vx+vy*vy+vz*vz);
          float pp=fmaxf((GAMMA-1.f)*(U.E-ke),SMALLP);
          o.h[0][g]=enc_log2s(log2f(fmaxf(U.r,SMALLR)));
          o.h[1][g]=enc_vel(vx); o.h[2][g]=enc_vel(vy); o.h[3][g]=enc_vel(vz);
          o.h[4][g]=enc_log2s(log2f(pp)); }
#elif defined(F16STATE)
        { float ir=1.f/fmaxf(U.r,SMALLR); float vx=U.mx*ir,vy=U.my*ir,vz=U.mz*ir;
          float ke=0.5f*U.r*(vx*vx+vy*vy+vz*vz); float pp=fmaxf((GAMMA-1.f)*(U.E-ke),SMALLP);
          o.hf[0][g]=__float2half(U.r); o.hf[1][g]=__float2half(vx); o.hf[2][g]=__float2half(vy); o.hf[3][g]=__float2half(vz); o.hf[4][g]=__float2half(pp); }
#else
        o.v[0][g]=U.r; o.v[1][g]=U.mx; o.v[2][g]=U.my; o.v[3][g]=U.mz; o.v[4][g]=U.E;
#endif
#ifdef U16SP
        float rinv=1.f/fmaxf(U.r,SMALLR);                       // X_new = (rho*X)_new / rho_new -> log2 -> uint16
        o.su[0][g]=enc_log2(log2f(fmaxf(U.c0*rinv,7.7e-34f)));
        o.su[1][g]=enc_log2(log2f(fmaxf(U.c1*rinv,7.7e-34f)));
#elif defined(SCALARS)
        o.v[5][g]=U.c0; o.v[6][g]=U.c1;
#endif

        // advance ring: load plane k+3 (overwrites k-2 slot), barrier
        loadPlane(k+3);
        __syncthreads();
    }
#undef CR
#undef GP
}

__global__ void init(Ptrs q){
    size_t n=(size_t)NX*NY*NZ, i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float ph = 0.001f*(float)(i%911);
    float rho=1.f+0.3f*ph, mx=0.1f*ph, my=-0.1f*ph, mz=0.05f*ph, E=2.5f+0.5f*ph;
#ifdef U16STATE
    { float ir=1.f/rho, vx=mx*ir, vy=my*ir, vz=mz*ir;
      float ke=0.5f*rho*(vx*vx+vy*vy+vz*vz), pp=fmaxf((GAMMA-1.f)*(E-ke),SMALLP);
      q.h[0][i]=enc_log2s(log2f(rho)); q.h[1][i]=enc_vel(vx); q.h[2][i]=enc_vel(vy); q.h[3][i]=enc_vel(vz); q.h[4][i]=enc_log2s(log2f(pp)); }
#elif defined(F16STATE)
    { float ir=1.f/rho, vx=mx*ir, vy=my*ir, vz=mz*ir; float ke=0.5f*rho*(vx*vx+vy*vy+vz*vz), pp=fmaxf((GAMMA-1.f)*(E-ke),SMALLP);
      q.hf[0][i]=__float2half(rho); q.hf[1][i]=__float2half(vx); q.hf[2][i]=__float2half(vy); q.hf[3][i]=__float2half(vz); q.hf[4][i]=__float2half(pp); }
#else
    q.v[0][i]=rho; q.v[1][i]=mx; q.v[2][i]=my; q.v[3][i]=mz; q.v[4][i]=E;
#endif
#ifdef U16SP
    q.su[0][i]=enc_log2(log2f(0.3f+0.2f*ph));            // ~O(0.3) fraction
    q.su[1][i]=enc_log2(log2f(1e-20f*(1.f+0.5f*ph)));    // trace species, exercises 30-dex range
#elif defined(SCALARS)
    q.v[5][i]=rho*(0.2f+0.5f*ph); q.v[6][i]=rho*(0.7f-0.3f*ph); // rho*s (rho from above)
#endif
}

#ifdef AS_LIB
// --- Julia bridge: drive the nvcc march kernel from CUDA.jl over shared device memory. ---
// Default fp32-conserved build only (the 64-reg HANCOCK1D reference path). qv/ov are arrays
// of NV device pointers (one CuArray plane per conserved variable, SoA: rho,mx,my,mz,E[,s0,s1]).
// Runs nsteps periodic marches; result always left in qv (a final D2D copy if nsteps is odd).
// Returns elapsed kernel time in ms (events; excludes any host/device transfer).
extern "C" {
int march_nv(){ return NV; }
int march_nx(){ return NX; }
void march_set_dtdx(float v){ cudaMemcpyToSymbol(DTDX, &v, sizeof(float)); }
double march_run_dev(float* const* qv, float* const* ov, int nsteps){
    size_t n=(size_t)NX*NY*NZ;
    Ptrs q,o;
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
} // extern "C"
#endif

#ifndef AS_LIB
int main(){
    size_t n=(size_t)NX*NY*NZ;
    Ptrs q,o;
#ifdef U16STATE
    for(int v=0;v<5;v++){ cudaMalloc(&q.h[v],n*2); cudaMalloc(&o.h[v],n*2); }
#elif defined(F16STATE)
    for(int v=0;v<5;v++){ cudaMalloc(&q.hf[v],n*2); cudaMalloc(&o.hf[v],n*2); }
#elif defined(U16SP)
    for(int v=0;v<5;v++){ cudaMalloc(&q.v[v],n*4); cudaMalloc(&o.v[v],n*4); }
    for(int j=0;j<2;j++){ cudaMalloc(&q.su[j],n*2); cudaMalloc(&o.su[j],n*2); }
#else
    for(int v=0;v<NV;v++){ cudaMalloc(&q.v[v],n*4); cudaMalloc(&o.v[v],n*4); }
#endif
    init<<<(n+255)/256,256>>>(q); init<<<(n+255)/256,256>>>(o); cudaDeviceSynchronize();

    dim3 grid(NX/OX, NY/OY);
    size_t shmem = sizeof(__half)*NV*PLANES*GX*GY;
    int nblk=0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblk, march, THREADS, 0);
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, march);

    // warmup
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

    double cells=(double)n*iters;
    double mcs = cells/ (ms*1e-3) /1e6;
    double gbs = cells*(NV*4.0*2.0)/(ms*1e-3)/1e9;   // 2x field essential traffic
#ifdef MEMFLOOR
    const char* mode="MEMFLOOR (load+store, no flux)";
#else
    const char* mode="FULL (PLM moncen + HLL x3)";
#endif
    printf("2.5D march  %s\n", mode);
    printf("  grid 480^3  tile %dx%d owned (x-y over-read %.2fx)  ring %d planes\n",
           OX,OY,(double)(GX*GY)/(OX*OY),PLANES);
    printf("  regs=%d  shmem=%zuB (%.1fKB)  blocks/SM=%d\n", fa.numRegs, shmem, shmem/1024.0, nblk);
    printf("  %.2f ms / %d iters  =>  %.0f Mcell/s   %.0f GB/s  (%.0f%% of 768 peak)\n",
           ms/iters, iters, mcs, gbs, gbs/768.0*100.0);
    printf("  vs production PLM 4127 / floor 5448 Mcell/s\n");
    return 0;
}
#endif
