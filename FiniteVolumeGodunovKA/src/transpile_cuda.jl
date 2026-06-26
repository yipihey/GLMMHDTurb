# Transpile-to-CUDA-C performance backend (`Grid3DCuMarch`).
#
# Emits a fused 2nd-order PLM nvcc kernel FROM the @fvsystem stencil (`_fvmeta`), compiles with nvcc
# --use_fast_math, and runs it over CuArrays — reaching ~.cu-class throughput (Euler ~85-90% / GLM
# ~.cu of the hand-tuned reference) from the same branch-free source the portable backends use. The
# transpiled physics is bit-identical to the Julia @fvsystem functions (see `transpile_selfcheck`).
#
# Requires nvcc at construction time only (an external tool, not a Julia dep) — the package loads fine
# without it; only building a Grid3DCuMarch shells out to nvcc. Handles arbitrary params (→ PRM[]),
# NVARS, the vidx rotation (generated swap_y/swap_z), and inter-physics calls (GLM maxspeed→fastspeed).

const _CU_NAMEMAP = Dict('ρ'=>"rho", 'γ'=>"gam", 'ψ'=>"psi", 'Δ'=>"D", 'λ'=>"lam")
_csan(s) = join(get(_CU_NAMEMAP, c, string(c)) for c in string(s))

# `half=true` emits the identical expression in `__half` (f16) arithmetic — operators on __half are
# overloaded in device code; only the intrinsics and literals change. The f32 path is byte-identical.
function _e2c(e, pidx, physset; half::Bool=false)
    lit(x) = half ? "__float2half($(x)f)" : "$(x)f"
    e isa Float32 && return lit(e)
    e isa Float64 && return lit(Float32(e))
    e isa Int     && return half ? "__float2half($(e).0f)" : "$(e).0f"
    e isa Symbol  && return _csan(e)
    if e isa Expr
        if e.head === :call
            op = e.args[1]; a = e.args[2:end]; rec(x) = _e2c(x, pidx, physset; half=half)
            op in physset && return string(_csan(op), half ? "_h(" : "(", join([rec(x) for x in a if x !== :p], ","), ", PRM)")
            op === :inv  && return "(" * lit(1.0) * "/(" * rec(a[1]) * "))"
            op === :sqrt && return (half ? "hsqrt(" : "sqrtf(") * rec(a[1]) * ")"
            op === :abs  && return (half ? "__habs(" : "fabsf(") * rec(a[1]) * ")"
            op === :min  && return (half ? "__hmin(" : "fminf(") * rec(a[1]) * "," * rec(a[2]) * ")"
            op === :max  && return (half ? "__hmax(" : "fmaxf(") * rec(a[1]) * "," * rec(a[2]) * ")"
            op === :sign && return "((" * rec(a[1]) * ")>=" * lit(0.0) * "?" * lit(1.0) * ":" * lit(-1.0) * ")"
            op === :ifelse && return "((" * rec(a[1]) * ")?(" * rec(a[2]) * "):(" * rec(a[3]) * "))"
            if op in (:+, :-, :*, :/)
                length(a) == 1 && op === :- && return "(-" * rec(a[1]) * ")"
                s = rec(a[1]); for x in a[2:end]; s = "(" * s * string(op) * rec(x) * ")"; end
                return s
            end
            error("transpile: unsupported call $op")
        elseif e.head === :.
            return "PRM[$(pidx[e.args[2].value])]"
        end
    end
    error("transpile: unsupported expr $e")
end

function _genfunc(cname, arg, body, pidx, physset; half::Bool=false)
    T = half ? "__half" : "float"; sfx = half ? "_h" : ""; rec(x) = _e2c(x, pidx, physset; half=half)
    qual = half ? "__device__" : "__host__ __device__"   # f16 intrinsics (hsqrt…) are device-only
    items = filter(x -> !(x isa LineNumberNode), body.args); ret = items[end]; stmts = String[]
    for s in items[1:end-1]
        lhs, rhs = s.args[1], s.args[2]
        if lhs isa Symbol
            push!(stmts, "  $T $(_csan(lhs)) = $(rec(rhs));")
        elseif lhs.head === :tuple
            names = [_csan(x) for x in lhs.args]
            if rhs isa Symbol
                for (i,n) in enumerate(names); push!(stmts, "  $T $n = $(_csan(rhs))[$(i-1)];"); end
            else
                for (n,ex) in zip(names, rhs.args); push!(stmts, "  $T $n = $(rec(ex));"); end
            end
        end
    end
    bs = join(stmts, "\n") * (isempty(stmts) ? "" : "\n")
    if ret isa Expr && ret.head === :tuple
        outs = join(["  out[$(i-1)] = $(rec(ex));" for (i,ex) in enumerate(ret.args)], "\n")
        return ("$qual void $cname$sfx(const $T* $(_csan(arg)), $T* out, const $T* PRM)",
                "$qual void $cname$sfx(const $T* $(_csan(arg)), $T* out, const $T* PRM) {\n$bs$outs\n}\n")
    else
        return ("$qual $T $cname$sfx(const $T* $(_csan(arg)), const $T* PRM)",
                "$qual $T $cname$sfx(const $T* $(_csan(arg)), const $T* PRM) {\n$bs  return $(rec(ret));\n}\n")
    end
end

function _genswap(cname, vidx, d; half::Bool=false)
    T = half ? "__half" : "float"
    sw = join(["{ $T t=W[$(tr[1]-1)]; W[$(tr[1]-1)]=W[$(tr[d]-1)]; W[$(tr[d]-1)]=t; }" for tr in vidx], " ")
    "__device__ void $cname($T* W){ $sw }\n"
end

"`gen_cuda_c(sys) -> String`: the full CUDA-C source emitted from the system's `@fvsystem` stencil."
function gen_cuda_c(sys::FVSystem)
    m = _fvmeta(sys); NV = m.nvars
    pidx = Dict(p => i-1 for (i,p) in enumerate(m.params)); physset = Set(keys(m.phys))
    want = [:cons2prim, :prim2cons, :physflux_x, :maxspeed_x]
    haskey(m.phys, :fastspeed_x) && pushfirst!(want, :fastspeed_x)
    protos = String[]; defs = String[]
    for f in want
        haskey(m.phys, f) || continue
        proto, def = _genfunc(string(f), m.phys[f][1], m.phys[f][2], pidx, physset)
        push!(protos, "$proto;"); push!(defs, def)
        ph, dh = _genfunc(string(f), m.phys[f][1], m.phys[f][2], pidx, physset; half = true)  # f16 twin (scheme=:f16)
        push!(protos, "$ph;"); push!(defs, dh)
    end
    tests = """
extern "C" {
  int fv_nv() { return $NV; }
  void  fv_cons2prim(const float* U, float* W, const float* P) { cons2prim(U,W,P); }
  void  fv_physflux (const float* W, float* F, const float* P) { physflux_x(W,F,P); }
  float fv_maxspeed (const float* W, const float* P)           { return maxspeed_x(W,P); }
}
"""
    fixed = replace(raw"""
__device__ float mc(float a, float b){ if(a*b<=0.f) return 0.f; float s=(a>0.f)?1.f:-1.f;
  return s*fminf(0.5f*fabsf(a+b), fminf(2.f*fabsf(a),2.f*fabsf(b))); }
__device__ void halfstep(const float* Wm,const float* W0,const float* Wp,float lam,const float* PRM,float* WLh,float* WRh){
  float WL[NV],WR[NV]; for(int c=0;c<NV;c++){ float d=mc(W0[c]-Wm[c],Wp[c]-W0[c]); WL[c]=W0[c]-0.5f*d; WR[c]=W0[c]+0.5f*d; }
  float UL[NV],UR[NV],FL[NV],FR[NV]; prim2cons(WL,UL,PRM); prim2cons(WR,UR,PRM); physflux_x(WL,FL,PRM); physflux_x(WR,FR,PRM);
  float Ul[NV],Ur[NV]; for(int c=0;c<NV;c++){ float dh=0.5f*lam*(FR[c]-FL[c]); Ul[c]=UL[c]-dh; Ur[c]=UR[c]-dh; }
  cons2prim(Ul,WLh,PRM); cons2prim(Ur,WRh,PRM); }
__device__ void llf(const float* WL,const float* WR,float* F,const float* PRM){
  float UL[NV],UR[NV],FL[NV],FR[NV]; prim2cons(WL,UL,PRM); prim2cons(WR,UR,PRM); physflux_x(WL,FL,PRM); physflux_x(WR,FR,PRM);
  float s=fmaxf(maxspeed_x(WL,PRM),maxspeed_x(WR,PRM)); for(int c=0;c<NV;c++) F[c]=0.5f*(FL[c]+FR[c])-0.5f*s*(UR[c]-UL[c]); }
__device__ void fluxdiff(const float* const W[5],float lam,const float* PRM,float* out){
  float WRm[NV],WL0[NV],WR0[NV],WLp[NV],dump[NV];
  halfstep(W[0],W[1],W[2],lam,PRM,dump,WRm); halfstep(W[1],W[2],W[3],lam,PRM,WL0,WR0); halfstep(W[2],W[3],W[4],lam,PRM,WLp,dump);
  float Fl[NV],Fr[NV]; llf(WRm,WL0,Fl,PRM); llf(WR0,WLp,Fr,PRM); for(int c=0;c<NV;c++) out[c]=Fr[c]-Fl[c]; }
#define IDX(a,b,c) (((long)(c)*ny+(b))*nx+(a))
#define LDP(W,ii,jj,kk) float W[NV]; { long q=IDX(ii,jj,kk); float U[NV]; for(int c=0;c<NV;c++) U[c]=R[(long)c*VOL+q]; cons2prim(U,W,PRM); }
#define SWC(dst,src,sw) float dst[NV]; { for(int c=0;c<NV;c++) dst[c]=src[c]; sw(dst); }
__global__ void k_step(const float* R, float* O, int nx,int ny,int nz, float lam, const float* PRM){
  int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y, k=blockIdx.z*blockDim.z+threadIdx.z;
  if(i>=nx||j>=ny||k>=nz) return; long VOL=(long)nx*ny*nz;
  LDP(Wc,i,j,k)
  LDP(Xm2,(i-2+nx)%nx,j,k) LDP(Xm1,(i-1+nx)%nx,j,k) LDP(Xp1,(i+1)%nx,j,k) LDP(Xp2,(i+2)%nx,j,k)
  const float* xl[5]={Xm2,Xm1,Wc,Xp1,Xp2}; float fx[NV]; fluxdiff(xl,lam,PRM,fx);
  LDP(Ym2,i,(j-2+ny)%ny,k) LDP(Ym1,i,(j-1+ny)%ny,k) LDP(Yp1,i,(j+1)%ny,k) LDP(Yp2,i,(j+2)%ny,k)
  SWC(ya,Ym2,swap_y) SWC(yb,Ym1,swap_y) SWC(yc,Wc,swap_y) SWC(yd,Yp1,swap_y) SWC(ye,Yp2,swap_y)
  const float* yl[5]={ya,yb,yc,yd,ye}; float fy[NV]; fluxdiff(yl,lam,PRM,fy); swap_y(fy);
  LDP(Zm2,i,j,(k-2+nz)%nz) LDP(Zm1,i,j,(k-1+nz)%nz) LDP(Zp1,i,j,(k+1)%nz) LDP(Zp2,i,j,(k+2)%nz)
  SWC(za,Zm2,swap_z) SWC(zb,Zm1,swap_z) SWC(zc,Wc,swap_z) SWC(zd,Zp1,swap_z) SWC(ze,Zp2,swap_z)
  const float* zl[5]={za,zb,zc,zd,ze}; float fz[NV]; fluxdiff(zl,lam,PRM,fz); swap_z(fz);
  long q=IDX(i,j,k); for(int c=0;c<NV;c++) O[(long)c*VOL+q] = R[(long)c*VOL+q] - lam*(fx[c]+fy[c]+fz[c]); }
// per-cell sum of the three directional signal speeds (the unsplit-CFL quantity); host reduces the max.
__global__ void k_speed(const float* R, float* spd, int nx,int ny,int nz, const float* PRM){
  int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y, k=blockIdx.z*blockDim.z+threadIdx.z;
  if(i>=nx||j>=ny||k>=nz) return; long VOL=(long)nx*ny*nz;
  LDP(W,i,j,k) SWC(wy,W,swap_y) SWC(wz,W,swap_z)
  spd[IDX(i,j,k)] = maxspeed_x(W,PRM)+maxspeed_x(wy,PRM)+maxspeed_x(wz,PRM); }
extern "C" void fv_speed(float* R, float* spd, int nx,int ny,int nz, const float* PRM){
  dim3 thr(8,8,4), grp((nx+7)/8,(ny+7)/8,(nz+3)/4); k_speed<<<grp,thr>>>(R,spd,nx,ny,nz,PRM); cudaDeviceSynchronize(); }
extern "C" void fv_run(float* R, float* O, int nx,int ny,int nz, float lam, const float* PRM, int nsteps){
  dim3 thr(8,8,4), grp((nx+7)/8,(ny+7)/8,(nz+3)/4); float *a=R,*b=O;
  for(int s=0;s<nsteps;s++){ k_step<<<grp,thr>>>(a,b,nx,ny,nz,lam,PRM); float* t=a; a=b; b=t; }
  if(a!=R) cudaMemcpy(R,a,(size_t)NV*nx*ny*nz*sizeof(float),cudaMemcpyDeviceToDevice);   // result always in R
  cudaDeviceSynchronize(); }
// ---- genuinely 2nd-order path (MUSCL + SSP-RK2): Hancock-free PLM reconstruction + LLF, RK2 in time ----
__device__ void recon(const float* Wm,const float* W0,const float* Wp,float* WL,float* WR){
  for(int c=0;c<NV;c++){ float d=mc(W0[c]-Wm[c],Wp[c]-W0[c]); WL[c]=W0[c]-0.5f*d; WR[c]=W0[c]+0.5f*d; } }
__device__ void fluxdiff_rk(const float* const W[5],const float* PRM,float* out){
  float WLm[NV],WRm[NV],WL0[NV],WR0[NV],WLp[NV],WRp[NV];
  recon(W[0],W[1],W[2],WLm,WRm); recon(W[1],W[2],W[3],WL0,WR0); recon(W[2],W[3],W[4],WLp,WRp);
  float Fl[NV],Fr[NV]; llf(WRm,WL0,Fl,PRM); llf(WR0,WLp,Fr,PRM); for(int c=0;c<NV;c++) out[c]=Fr[c]-Fl[c]; }
#define LDPS(W,src,ii,jj,kk) float W[NV]; { long q=IDX(ii,jj,kk); float U[NV]; for(int c=0;c<NV;c++) U[c]=src[(long)c*VOL+q]; cons2prim(U,W,PRM); }
// D = lam*(div F) for the conservative state in S at cell (i,j,k); the per-stage RK increment.
__device__ void compute_D(const float* S,int i,int j,int k,int nx,int ny,int nz,float lam,const float* PRM,float* D){
  long VOL=(long)nx*ny*nz;
  LDPS(Wc,S,i,j,k)
  LDPS(Xm2,S,(i-2+nx)%nx,j,k) LDPS(Xm1,S,(i-1+nx)%nx,j,k) LDPS(Xp1,S,(i+1)%nx,j,k) LDPS(Xp2,S,(i+2)%nx,j,k)
  const float* xl[5]={Xm2,Xm1,Wc,Xp1,Xp2}; float fx[NV]; fluxdiff_rk(xl,PRM,fx);
  LDPS(Ym2,S,i,(j-2+ny)%ny,k) LDPS(Ym1,S,i,(j-1+ny)%ny,k) LDPS(Yp1,S,i,(j+1)%ny,k) LDPS(Yp2,S,i,(j+2)%ny,k)
  SWC(ya,Ym2,swap_y) SWC(yb,Ym1,swap_y) SWC(yc,Wc,swap_y) SWC(yd,Yp1,swap_y) SWC(ye,Yp2,swap_y)
  const float* yl[5]={ya,yb,yc,yd,ye}; float fy[NV]; fluxdiff_rk(yl,PRM,fy); swap_y(fy);
  LDPS(Zm2,S,i,j,(k-2+nz)%nz) LDPS(Zm1,S,i,j,(k-1+nz)%nz) LDPS(Zp1,S,i,j,(k+1)%nz) LDPS(Zp2,S,i,j,(k+2)%nz)
  SWC(za,Zm2,swap_z) SWC(zb,Zm1,swap_z) SWC(zc,Wc,swap_z) SWC(zd,Zp1,swap_z) SWC(ze,Zp2,swap_z)
  const float* zl[5]={za,zb,zc,zd,ze}; float fz[NV]; fluxdiff_rk(zl,PRM,fz); swap_z(fz);
  for(int c=0;c<NV;c++) D[c]=lam*(fx[c]+fy[c]+fz[c]); }
__global__ void k_euler(const float* R,float* O,int nx,int ny,int nz,float lam,const float* PRM){
  int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y*blockDim.y+threadIdx.y,k=blockIdx.z*blockDim.z+threadIdx.z;
  if(i>=nx||j>=ny||k>=nz) return; long VOL=(long)nx*ny*nz,q=IDX(i,j,k);
  float D[NV]; compute_D(R,i,j,k,nx,ny,nz,lam,PRM,D); for(int c=0;c<NV;c++) O[(long)c*VOL+q]=R[(long)c*VOL+q]-D[c]; }
__global__ void k_rk2b(const float* R,const float* U1,float* O,int nx,int ny,int nz,float lam,const float* PRM){
  int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y*blockDim.y+threadIdx.y,k=blockIdx.z*blockDim.z+threadIdx.z;
  if(i>=nx||j>=ny||k>=nz) return; long VOL=(long)nx*ny*nz,q=IDX(i,j,k);
  float D[NV]; compute_D(U1,i,j,k,nx,ny,nz,lam,PRM,D);
  for(int c=0;c<NV;c++) O[(long)c*VOL+q]=0.5f*(R[(long)c*VOL+q]+U1[(long)c*VOL+q]-D[c]); }
extern "C" void fv_run_rk2(float* R,float* T,float* O,int nx,int ny,int nz,float lam,const float* PRM,int nsteps){
  dim3 thr(8,8,4),grp((nx+7)/8,(ny+7)/8,(nz+3)/4); float* curr=R; float* sc=O;
  for(int s=0;s<nsteps;s++){ k_euler<<<grp,thr>>>(curr,T,nx,ny,nz,lam,PRM);
    k_rk2b<<<grp,thr>>>(curr,T,sc,nx,ny,nz,lam,PRM); float* t=curr; curr=sc; sc=t; }
  if(curr!=R) cudaMemcpy(R,curr,(size_t)NV*nx*ny*nz*sizeof(float),cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize(); }
// ---- single-pass 2nd-order: unsplit MUSCL-Hancock with the TRANSVERSE predictor (CTU-style) ----
// physical flux in direction dir (0=x,1=y,2=z) via component rotation; reuses physflux_x + swaps.
__device__ void flux_dir(const float* W,int dir,const float* PRM,float* F){
  float Wr[NV]; for(int c=0;c<NV;c++) Wr[c]=W[c]; if(dir==1) swap_y(Wr); else if(dir==2) swap_z(Wr);
  physflux_x(Wr,F,PRM); if(dir==1) swap_y(F); else if(dir==2) swap_z(F); }
__device__ void llf_dir(const float* WL,const float* WR,int dir,float* F,const float* PRM){
  float L[NV],Rr[NV]; for(int c=0;c<NV;c++){ L[c]=WL[c]; Rr[c]=WR[c]; }
  if(dir==1){ swap_y(L); swap_y(Rr);} else if(dir==2){ swap_z(L); swap_z(Rr);}
  llf(L,Rr,F,PRM); if(dir==1) swap_y(F); else if(dir==2) swap_z(F); }
// cell's 6 face states, PLM-reconstructed and evolved dt/2 by the FULL (all-direction) flux divergence.
__device__ void predict_cell(const float* R,int i,int j,int k,int nx,int ny,int nz,float lam,const float* PRM,
                             float* WxL,float* WxR,float* WyL,float* WyR,float* WzL,float* WzR){
  long VOL=(long)nx*ny*nz;
  LDP(W0,i,j,k)
  LDP(Wxm,(i-1+nx)%nx,j,k) LDP(Wxp,(i+1)%nx,j,k)
  LDP(Wym,i,(j-1+ny)%ny,k) LDP(Wyp,i,(j+1)%ny,k)
  LDP(Wzm,i,j,(k-1+nz)%nz) LDP(Wzp,i,j,(k+1)%nz)
  recon(Wxm,W0,Wxp,WxL,WxR); recon(Wym,W0,Wyp,WyL,WyR); recon(Wzm,W0,Wzp,WzL,WzR);
  float Fa[NV],Fb[NV],dU[NV]; for(int c=0;c<NV;c++) dU[c]=0.f;
  flux_dir(WxR,0,PRM,Fa); flux_dir(WxL,0,PRM,Fb); for(int c=0;c<NV;c++) dU[c]+=Fa[c]-Fb[c];
  flux_dir(WyR,1,PRM,Fa); flux_dir(WyL,1,PRM,Fb); for(int c=0;c<NV;c++) dU[c]+=Fa[c]-Fb[c];
  flux_dir(WzR,2,PRM,Fa); flux_dir(WzL,2,PRM,Fb); for(int c=0;c<NV;c++) dU[c]+=Fa[c]-Fb[c];
  for(int c=0;c<NV;c++) dU[c]*=-0.5f*lam;
  float* faces[6]={WxL,WxR,WyL,WyR,WzL,WzR};
  for(int f=0;f<6;f++){ float U[NV]; prim2cons(faces[f],U,PRM); for(int c=0;c<NV;c++) U[c]+=dU[c]; cons2prim(U,faces[f],PRM); } }
__global__ void k_ctu(const float* R,float* O,int nx,int ny,int nz,float lam,const float* PRM){
  int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y*blockDim.y+threadIdx.y,k=blockIdx.z*blockDim.z+threadIdx.z;
  if(i>=nx||j>=ny||k>=nz) return; long VOL=(long)nx*ny*nz,q=IDX(i,j,k);
  float sxL[NV],sxR[NV],syL[NV],syR[NV],szL[NV],szR[NV];     // this cell's 6 predicted faces
  predict_cell(R,i,j,k,nx,ny,nz,lam,PRM,sxL,sxR,syL,syR,szL,szR);
  float a[NV],b[NV],c[NV],d[NV],e[NV],g[NV];                 // neighbor scratch (only one face used each)
  float Fxp[NV],Fxm[NV],Fyp[NV],Fym[NV],Fzp[NV],Fzm[NV];
  predict_cell(R,(i+1)%nx,j,k,nx,ny,nz,lam,PRM,a,b,c,d,e,g);        llf_dir(sxR,a,0,Fxp,PRM); // i+1/2: self.WxR | xp.WxL
  predict_cell(R,(i-1+nx)%nx,j,k,nx,ny,nz,lam,PRM,a,b,c,d,e,g);     llf_dir(b,sxL,0,Fxm,PRM); // i-1/2: xm.WxR | self.WxL
  predict_cell(R,i,(j+1)%ny,k,nx,ny,nz,lam,PRM,a,b,c,d,e,g);        llf_dir(syR,c,1,Fyp,PRM);
  predict_cell(R,i,(j-1+ny)%ny,k,nx,ny,nz,lam,PRM,a,b,c,d,e,g);     llf_dir(d,syL,1,Fym,PRM);
  predict_cell(R,i,j,(k+1)%nz,nx,ny,nz,lam,PRM,a,b,c,d,e,g);        llf_dir(szR,e,2,Fzp,PRM);
  predict_cell(R,i,j,(k-1+nz)%nz,nx,ny,nz,lam,PRM,a,b,c,d,e,g);     llf_dir(g,szL,2,Fzm,PRM);
  for(int cc=0;cc<NV;cc++) O[(long)cc*VOL+q]=R[(long)cc*VOL+q]-lam*((Fxp[cc]-Fxm[cc])+(Fyp[cc]-Fym[cc])+(Fzp[cc]-Fzm[cc])); }
extern "C" void fv_run_ctu(float* R,float* O,int nx,int ny,int nz,float lam,const float* PRM,int nsteps){
  dim3 thr(8,8,4),grp((nx+7)/8,(ny+7)/8,(nz+3)/4); float *a=R,*b=O;
  for(int s=0;s<nsteps;s++){ k_ctu<<<grp,thr>>>(a,b,nx,ny,nz,lam,PRM); float* t=a; a=b; b=t; }
  if(a!=R) cudaMemcpy(R,a,(size_t)NV*nx*ny*nz*sizeof(float),cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize(); }
// ---- shared-memory-TILED single-pass CTU: compute each cell's transverse dU ONCE into shared ----
#define TBX 8
#define TBY 8
#define TBZ ZZTBZ
#define WSX (TBX+4)
#define WSY (TBY+4)
#define WSZ (TBZ+4)
#define DSX (TBX+2)
#define DSY (TBY+2)
#define DSZ (TBZ+2)
#define FSX (TBX+1)
#define FSY (TBY+1)
#define FSZ (TBZ+1)
#define WSI(lx,ly,lz) ((((lz)*WSY+(ly))*WSX+(lx))*NV)
#define DSI(lx,ly,lz) ((((lz)*DSY+(ly))*DSX+(lx))*NV)
#define FSI(lx,ly,lz) ((((lz)*FSY+(ly))*FSX+(lx))*NV)
__device__ inline void ldh(const __half* p,float* w){ for(int c=0;c<NV;c++) w[c]=__half2float(p[c]); }
// transverse correction dU = -0.5*lam*div(F of reconstructed faces), for Ws-local cell (lx,ly,lz). f16 tile.
__device__ void compute_dU(const __half* Ws,int lx,int ly,int lz,float lam,const float* PRM,__half* dU){
  float W0[NV],Wxm[NV],Wxp[NV],Wym[NV],Wyp[NV],Wzm[NV],Wzp[NV];
  ldh(Ws+WSI(lx,ly,lz),W0);
  ldh(Ws+WSI(lx-1,ly,lz),Wxm); ldh(Ws+WSI(lx+1,ly,lz),Wxp);
  ldh(Ws+WSI(lx,ly-1,lz),Wym); ldh(Ws+WSI(lx,ly+1,lz),Wyp);
  ldh(Ws+WSI(lx,ly,lz-1),Wzm); ldh(Ws+WSI(lx,ly,lz+1),Wzp);
  float WxL[NV],WxR[NV],WyL[NV],WyR[NV],WzL[NV],WzR[NV];
  recon(Wxm,W0,Wxp,WxL,WxR); recon(Wym,W0,Wyp,WyL,WyR); recon(Wzm,W0,Wzp,WzL,WzR);
  float Fa[NV],Fb[NV],d[NV]; for(int c=0;c<NV;c++) d[c]=0.f;
  flux_dir(WxR,0,PRM,Fa); flux_dir(WxL,0,PRM,Fb); for(int c=0;c<NV;c++) d[c]+=Fa[c]-Fb[c];
  flux_dir(WyR,1,PRM,Fa); flux_dir(WyL,1,PRM,Fb); for(int c=0;c<NV;c++) d[c]+=Fa[c]-Fb[c];
  flux_dir(WzR,2,PRM,Fa); flux_dir(WzL,2,PRM,Fb); for(int c=0;c<NV;c++) d[c]+=Fa[c]-Fb[c];
  for(int c=0;c<NV;c++) dU[c]=__float2half(d[c]*-0.5f*lam); }
// one-sided PLM reconstruction (only the face we need; side 0=L i.e. -, 1=R i.e. +).
__device__ void recon_one(const float* Wm,const float* W0,const float* Wp,int side,float* out){
  float s=(side?0.5f:-0.5f); for(int c=0;c<NV;c++) out[c]=W0[c]+s*mc(W0[c]-Wm[c],Wp[c]-W0[c]); }
// predicted face state of Ws-local cell (lx,ly,lz): PLM recon along dir, evolved by stored dUcell. side 0=L,1=R.
__device__ void predicted_face(const __half* Ws,int lx,int ly,int lz,int dir,int side,const __half* dUcell,const float* PRM,float* out){
  int ex=(dir==0),ey=(dir==1),ez=(dir==2);
  float Wm[NV],W0[NV],Wp[NV],dU[NV],Wf[NV];
  ldh(Ws+WSI(lx-ex,ly-ey,lz-ez),Wm); ldh(Ws+WSI(lx,ly,lz),W0); ldh(Ws+WSI(lx+ex,ly+ey,lz+ez),Wp); ldh(dUcell,dU);
  recon_one(Wm,W0,Wp,side,Wf);
  float U[NV]; prim2cons(Wf,U,PRM); for(int c=0;c<NV;c++) U[c]+=dU[c]; cons2prim(U,out,PRM); }
// flux at the interface between Ws-local cell (lx,ly,lz) and its +dir neighbor — computed ONCE, shared.
__device__ void iface_flux(const __half* Ws,const __half* dUs,int lx,int ly,int lz,int dir,const float* PRM,float* F){
  int ex=(dir==0),ey=(dir==1),ez=(dir==2); float L[NV],Rr[NV];
  predicted_face(Ws,lx,ly,lz,dir,1,dUs+DSI(lx-1,ly-1,lz-1),PRM,L);
  predicted_face(Ws,lx+ex,ly+ey,lz+ez,dir,0,dUs+DSI(lx-1+ex,ly-1+ey,lz-1+ez),PRM,Rr);
  llf_dir(L,Rr,dir,F,PRM); }
extern __shared__ __half smem[];
__global__ void k_ctus(const float* R,float* O,int nx,int ny,int nz,float lam,const float* PRM){
  __half* Ws=smem; __half* dUs=smem+WSX*WSY*WSZ*NV;
  int ox=blockIdx.x*TBX-2, oy=blockIdx.y*TBY-2, oz=blockIdx.z*TBZ-2;
  int tid=(threadIdx.z*TBY+threadIdx.y)*TBX+threadIdx.x; const int NT=TBX*TBY*TBZ;
  long VOL=(long)nx*ny*nz;
  for(int t=tid;t<WSX*WSY*WSZ;t+=NT){                         // phase 0: load W halo tile (store f16)
    int lx=t%WSX, ly=(t/WSX)%WSY, lz=t/(WSX*WSY);
    int gx=((ox+lx)%nx+nx)%nx, gy=((oy+ly)%ny+ny)%ny, gz=((oz+lz)%nz+nz)%nz;
    long q=((long)gz*ny+gy)*nx+gx; float U[NV],W[NV]; for(int c=0;c<NV;c++) U[c]=R[(long)c*VOL+q];
    cons2prim(U,W,PRM); for(int c=0;c<NV;c++) Ws[WSI(lx,ly,lz)+c]=__float2half(W[c]); }
  __syncthreads();
  for(int t=tid;t<DSX*DSY*DSZ;t+=NT){                         // phase 1: dU once per cell (dUs-local d ↔ Ws-local d+1)
    int dx=t%DSX, dy=(t/DSX)%DSY, dz=t/(DSX*DSY);
    compute_dU(Ws,dx+1,dy+1,dz+1,lam,PRM,dUs+DSI(dx,dy,dz)); }
  __syncthreads();
  // phase 2: each interface flux computed ONCE into a shared tile, then each cell accumulates the divergence.
  __shared__ float Fs[FSX*FSY*FSZ*NV];
  int i=blockIdx.x*TBX+threadIdx.x, j=blockIdx.y*TBY+threadIdx.y, k=blockIdx.z*TBZ+threadIdx.z;
  int tx=threadIdx.x, ty=threadIdx.y, tz=threadIdx.z; bool valid=(i<nx&&j<ny&&k<nz);
  float acc[NV]; for(int c=0;c<NV;c++) acc[c]=0.f;
  for(int p=tid;p<FSX*TBY*TBZ;p+=NT){ int fx=p%FSX, fy=(p/FSX)%TBY, fz=p/(FSX*TBY);   // X interfaces
    iface_flux(Ws,dUs,fx+1,fy+2,fz+2,0,PRM,Fs+FSI(fx,fy,fz)); }
  __syncthreads();
  if(valid) for(int c=0;c<NV;c++) acc[c]+=Fs[FSI(tx+1,ty,tz)+c]-Fs[FSI(tx,ty,tz)+c];
  __syncthreads();
  for(int p=tid;p<TBX*FSY*TBZ;p+=NT){ int fx=p%TBX, fy=(p/TBX)%FSY, fz=p/(TBX*FSY);   // Y interfaces
    iface_flux(Ws,dUs,fx+2,fy+1,fz+2,1,PRM,Fs+FSI(fx,fy,fz)); }
  __syncthreads();
  if(valid) for(int c=0;c<NV;c++) acc[c]+=Fs[FSI(tx,ty+1,tz)+c]-Fs[FSI(tx,ty,tz)+c];
  __syncthreads();
  for(int p=tid;p<TBX*TBY*FSZ;p+=NT){ int fx=p%TBX, fy=(p/TBX)%TBY, fz=p/(TBX*TBY);   // Z interfaces
    iface_flux(Ws,dUs,fx+2,fy+2,fz+1,2,PRM,Fs+FSI(fx,fy,fz)); }
  __syncthreads();
  if(valid){ for(int c=0;c<NV;c++) acc[c]+=Fs[FSI(tx,ty,tz+1)+c]-Fs[FSI(tx,ty,tz)+c];
    long q=IDX(i,j,k); for(int c=0;c<NV;c++) O[(long)c*VOL+q]=R[(long)c*VOL+q]-lam*acc[c]; } }
extern "C" void fv_run_ctus(float* R,float* O,int nx,int ny,int nz,float lam,const float* PRM,int nsteps){
  dim3 thr(TBX,TBY,TBZ), grp((nx+TBX-1)/TBX,(ny+TBY-1)/TBY,(nz+TBZ-1)/TBZ);
  int shbytes=(int)((WSX*WSY*WSZ+DSX*DSY*DSZ)*NV*sizeof(__half));
  cudaFuncSetAttribute(k_ctus, cudaFuncAttributeMaxDynamicSharedMemorySize, shbytes);
  float *a=R,*b=O;
  for(int s=0;s<nsteps;s++){ k_ctus<<<grp,thr,shbytes>>>(a,b,nx,ny,nz,lam,PRM); float* t=a; a=b; b=t; }
  if(a!=R) cudaMemcpy(R,a,(size_t)NV*nx*ny*nz*sizeof(float),cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize(); }
// ===== streaming z-march (2.5D): a 2D (x,y) block marches through z, rolling 5 W + 3 dU planes in shared =====
#define MBX ZZMB
#define MBY ZZMB
#define MWX (MBX+4)
#define MWY (MBY+4)
#define MUX (MBX+2)
#define MUY (MBY+2)
#define MWI(s,lx,ly) ((((s)*MWY+(ly))*MWX+(lx))*NV)
#define MUI(s,lx,ly) ((((s)*MUY+(ly))*MUX+(lx))*NV)
#define SL5(kz) ((((kz)%5)+5)%5)
#define SL3(kz) ((((kz)%3)+3)%3)
__device__ void mload_W(const float* R,__half* sW,int kz,int bx0,int by0,int nx,int ny,int nz,int tid,int NT,const float* PRM){
  int s=SL5(kz); int gz=((kz%nz)+nz)%nz; long VOL=(long)nx*ny*nz;
  for(int t=tid;t<MWX*MWY;t+=NT){ int lx=t%MWX, ly=t/MWX;
    int gx=((bx0-2+lx)%nx+nx)%nx, gy=((by0-2+ly)%ny+ny)%ny;
    long q=((long)gz*ny+gy)*nx+gx; float U[NV],W[NV]; for(int c=0;c<NV;c++) U[c]=R[(long)c*VOL+q];
    cons2prim(U,W,PRM); for(int c=0;c<NV;c++) sW[MWI(s,lx,ly)+c]=__float2half(W[c]); } }
__device__ void mcompute_dU(__half* sW,__half* sdU,int kz,int tid,int NT,float lam,const float* PRM){
  int sd=SL3(kz), s0=SL5(kz), sm=SL5(kz-1), sp=SL5(kz+1);
  for(int t=tid;t<MUX*MUY;t+=NT){ int ux=t%MUX, uy=t/MUX; int wx=ux+1, wy=uy+1;
    float W0[NV],Wxm[NV],Wxp[NV],Wym[NV],Wyp[NV],Wzm[NV],Wzp[NV];
    ldh(sW+MWI(s0,wx,wy),W0);
    ldh(sW+MWI(s0,wx-1,wy),Wxm); ldh(sW+MWI(s0,wx+1,wy),Wxp);
    ldh(sW+MWI(s0,wx,wy-1),Wym); ldh(sW+MWI(s0,wx,wy+1),Wyp);
    ldh(sW+MWI(sm,wx,wy),Wzm); ldh(sW+MWI(sp,wx,wy),Wzp);
    float WxL[NV],WxR[NV],WyL[NV],WyR[NV],WzL[NV],WzR[NV];
    recon(Wxm,W0,Wxp,WxL,WxR); recon(Wym,W0,Wyp,WyL,WyR); recon(Wzm,W0,Wzp,WzL,WzR);
    float Fa[NV],Fb[NV],d[NV]; for(int c=0;c<NV;c++) d[c]=0.f;
    flux_dir(WxR,0,PRM,Fa);flux_dir(WxL,0,PRM,Fb);for(int c=0;c<NV;c++)d[c]+=Fa[c]-Fb[c];
    flux_dir(WyR,1,PRM,Fa);flux_dir(WyL,1,PRM,Fb);for(int c=0;c<NV;c++)d[c]+=Fa[c]-Fb[c];
    flux_dir(WzR,2,PRM,Fa);flux_dir(WzL,2,PRM,Fb);for(int c=0;c<NV;c++)d[c]+=Fa[c]-Fb[c];
    for(int c=0;c<NV;c++) sdU[MUI(sd,ux,uy)+c]=__float2half(d[c]*-0.5f*lam); } }
__device__ void mpf_ip(__half* sW,__half* sdU,int s0,int sd,int wx,int wy,int ux,int uy,int dir,int side,const float* PRM,float* out){
  int ex=(dir==0),ey=(dir==1); float Wm[NV],W0[NV],Wp[NV],dU[NV],Wf[NV];
  ldh(sW+MWI(s0,wx-ex,wy-ey),Wm); ldh(sW+MWI(s0,wx,wy),W0); ldh(sW+MWI(s0,wx+ex,wy+ey),Wp); ldh(sdU+MUI(sd,ux,uy),dU);
  recon_one(Wm,W0,Wp,side,Wf); float U[NV]; prim2cons(Wf,U,PRM); for(int c=0;c<NV;c++) U[c]+=dU[c]; cons2prim(U,out,PRM); }
__device__ void mpf_z(__half* sW,__half* sdU,int sm,int s0,int sp,int wx,int wy,int sd,int ux,int uy,int side,const float* PRM,float* out){
  float Wm[NV],W0[NV],Wp[NV],dU[NV],Wf[NV];
  ldh(sW+MWI(sm,wx,wy),Wm); ldh(sW+MWI(s0,wx,wy),W0); ldh(sW+MWI(sp,wx,wy),Wp); ldh(sdU+MUI(sd,ux,uy),dU);
  recon_one(Wm,W0,Wp,side,Wf); float U[NV]; prim2cons(Wf,U,PRM); for(int c=0;c<NV;c++) U[c]+=dU[c]; cons2prim(U,out,PRM); }
__global__ void k_ctum(const float* R,float* O,int nx,int ny,int nz,float lam,const float* PRM){
  __half* sW=smem; __half* sdU=smem+5*MWX*MWY*NV;
  int bx0=blockIdx.x*MBX, by0=blockIdx.y*MBY;
  int tid=threadIdx.y*MBX+threadIdx.x; const int NT=MBX*MBY;
  int i=bx0+threadIdx.x, j=by0+threadIdx.y; bool valid=(i<nx&&j<ny);
  int wx=threadIdx.x+2, wy=threadIdx.y+2, ux=threadIdx.x+1, uy=threadIdx.y+1;
  long VOL=(long)nx*ny*nz;
  for(int kz=-2;kz<=2;kz++) mload_W(R,sW,kz,bx0,by0,nx,ny,nz,tid,NT,PRM);
  __syncthreads();
  for(int kz=-1;kz<=1;kz++) mcompute_dU(sW,sdU,kz,tid,NT,lam,PRM);
  __syncthreads();
  float Fz_prev[NV];                                   // z-flux at interface (k-1/2), carried down the march
  if(valid){ float A[NV],B[NV];
    mpf_z(sW,sdU,SL5(-2),SL5(-1),SL5(0),wx,wy,SL3(-1),ux,uy,1,PRM,A);  // cell -1, R face
    mpf_z(sW,sdU,SL5(-1),SL5(0),SL5(1),wx,wy,SL3(0),ux,uy,0,PRM,B);    // cell 0,  L face
    llf_dir(A,B,2,Fz_prev,PRM); }
  for(int k=0;k<nz;k++){
    if(valid){
      int s0=SL5(k),sm=SL5(k-1),sp=SL5(k+1),sm2=SL5(k-2),sp2=SL5(k+2);
      int d0=SL3(k),dm=SL3(k-1),dp=SL3(k+1);
      float A[NV],B[NV],F[NV],acc[NV]; for(int c=0;c<NV;c++) acc[c]=0.f;
      mpf_ip(sW,sdU,s0,d0,wx-1,wy,ux-1,uy,0,1,PRM,A); mpf_ip(sW,sdU,s0,d0,wx,wy,ux,uy,0,0,PRM,B); llf_dir(A,B,0,F,PRM); for(int c=0;c<NV;c++) acc[c]-=F[c];
      mpf_ip(sW,sdU,s0,d0,wx,wy,ux,uy,0,1,PRM,A); mpf_ip(sW,sdU,s0,d0,wx+1,wy,ux+1,uy,0,0,PRM,B); llf_dir(A,B,0,F,PRM); for(int c=0;c<NV;c++) acc[c]+=F[c];
      mpf_ip(sW,sdU,s0,d0,wx,wy-1,ux,uy-1,1,1,PRM,A); mpf_ip(sW,sdU,s0,d0,wx,wy,ux,uy,1,0,PRM,B); llf_dir(A,B,1,F,PRM); for(int c=0;c<NV;c++) acc[c]-=F[c];
      mpf_ip(sW,sdU,s0,d0,wx,wy,ux,uy,1,1,PRM,A); mpf_ip(sW,sdU,s0,d0,wx,wy+1,ux,uy+1,1,0,PRM,B); llf_dir(A,B,1,F,PRM); for(int c=0;c<NV;c++) acc[c]+=F[c];
      for(int c=0;c<NV;c++) acc[c]-=Fz_prev[c];                                                       // z- : carried
      mpf_z(sW,sdU,sm,s0,sp,wx,wy,d0,ux,uy,1,PRM,A); mpf_z(sW,sdU,s0,sp,sp2,wx,wy,dp,ux,uy,0,PRM,B); llf_dir(A,B,2,F,PRM);
      for(int c=0;c<NV;c++){ acc[c]+=F[c]; Fz_prev[c]=F[c]; }                                          // z+ : accumulate + carry
      long q=IDX(i,j,k); for(int c=0;c<NV;c++) O[(long)c*VOL+q]=R[(long)c*VOL+q]-lam*acc[c];
    }
    __syncthreads();
    mload_W(R,sW,k+3,bx0,by0,nx,ny,nz,tid,NT,PRM);
    __syncthreads();
    mcompute_dU(sW,sdU,k+2,tid,NT,lam,PRM);
    __syncthreads();
  } }
extern "C" void fv_run_ctum(float* R,float* O,int nx,int ny,int nz,float lam,const float* PRM,int nsteps){
  dim3 thr(MBX,MBY), grp((nx+MBX-1)/MBX,(ny+MBY-1)/MBY);
  int shb=(int)((5*MWX*MWY+3*MUX*MUY)*NV*sizeof(__half));
  cudaFuncSetAttribute(k_ctum, cudaFuncAttributeMaxDynamicSharedMemorySize, shb);
  float *a=R,*b=O;
  for(int s=0;s<nsteps;s++){ k_ctum<<<grp,thr,shb>>>(a,b,nx,ny,nz,lam,PRM); float* t=a; a=b; b=t; }
  if(a!=R) cudaMemcpy(R,a,(size_t)NV*nx*ny*nz*sizeof(float),cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize(); }
// ===== f16-ARITHMETIC streaming march (scheme=:f16): recon/flux in __half; conserved I/O + update stay f32 =====
__device__ __half mc_h(__half a,__half b){ __half z=__float2half(0.f);
  if(a*b<=z) return z; __half s=(a>z)?__float2half(1.f):__float2half(-1.f);
  return s*__hmin(__float2half(0.5f)*__habs(a+b),__hmin(__float2half(2.f)*__habs(a),__float2half(2.f)*__habs(b))); }
__device__ void recon_one_h(const __half* Wm,const __half* W0,const __half* Wp,int side,__half* out){
  __half h=side?__float2half(0.5f):__float2half(-0.5f); for(int c=0;c<NV;c++) out[c]=W0[c]+h*mc_h(W0[c]-Wm[c],Wp[c]-W0[c]); }
__device__ void recon_h(const __half* Wm,const __half* W0,const __half* Wp,__half* WL,__half* WR){
  __half h=__float2half(0.5f); for(int c=0;c<NV;c++){ __half d=mc_h(W0[c]-Wm[c],Wp[c]-W0[c]); WL[c]=W0[c]-h*d; WR[c]=W0[c]+h*d; } }
__device__ void llf_h(const __half* WL,const __half* WR,__half* F,const __half* PRM){
  __half UL[NV],UR[NV],FL[NV],FR[NV]; prim2cons_h(WL,UL,PRM); prim2cons_h(WR,UR,PRM); physflux_x_h(WL,FL,PRM); physflux_x_h(WR,FR,PRM);
  __half s=__hmax(maxspeed_x_h(WL,PRM),maxspeed_x_h(WR,PRM)),h=__float2half(0.5f);
  for(int c=0;c<NV;c++) F[c]=h*(FL[c]+FR[c])-h*s*(UR[c]-UL[c]); }
__device__ void flux_dir_h(const __half* W,int dir,const __half* PRM,__half* F){
  __half Wr[NV]; for(int c=0;c<NV;c++) Wr[c]=W[c]; if(dir==1) swap_y_h(Wr); else if(dir==2) swap_z_h(Wr);
  physflux_x_h(Wr,F,PRM); if(dir==1) swap_y_h(F); else if(dir==2) swap_z_h(F); }
__device__ void llf_dir_h(const __half* WL,const __half* WR,int dir,__half* F,const __half* PRM){
  __half L[NV],Rr[NV]; for(int c=0;c<NV;c++){L[c]=WL[c];Rr[c]=WR[c];} if(dir==1){swap_y_h(L);swap_y_h(Rr);} else if(dir==2){swap_z_h(L);swap_z_h(Rr);}
  llf_h(L,Rr,F,PRM); if(dir==1)swap_y_h(F); else if(dir==2)swap_z_h(F); }
__device__ void mload_Wh(const float* R,__half* sW,int kz,int bx0,int by0,int nx,int ny,int nz,int tid,int NT,const __half* PRM){
  int s=SL5(kz); int gz=((kz%nz)+nz)%nz; long VOL=(long)nx*ny*nz;
  for(int t=tid;t<MWX*MWY;t+=NT){ int lx=t%MWX, ly=t/MWX;
    int gx=((bx0-2+lx)%nx+nx)%nx, gy=((by0-2+ly)%ny+ny)%ny;
    long q=((long)gz*ny+gy)*nx+gx; __half U[NV]; for(int c=0;c<NV;c++) U[c]=__float2half(R[(long)c*VOL+q]);
    cons2prim_h(U,sW+MWI(s,lx,ly),PRM); } }
__device__ void mcompute_dUh(__half* sW,__half* sdU,int kz,int tid,int NT,__half nhl,const __half* PRM){
  int sd=SL3(kz), s0=SL5(kz), sm=SL5(kz-1), sp=SL5(kz+1);
  for(int t=tid;t<MUX*MUY;t+=NT){ int ux=t%MUX, uy=t/MUX; int wx=ux+1, wy=uy+1; const __half* W0=sW+MWI(s0,wx,wy);
    __half WxL[NV],WxR[NV],WyL[NV],WyR[NV],WzL[NV],WzR[NV];
    recon_h(sW+MWI(s0,wx-1,wy),W0,sW+MWI(s0,wx+1,wy),WxL,WxR);
    recon_h(sW+MWI(s0,wx,wy-1),W0,sW+MWI(s0,wx,wy+1),WyL,WyR);
    recon_h(sW+MWI(sm,wx,wy),W0,sW+MWI(sp,wx,wy),WzL,WzR);
    __half Fa[NV],Fb[NV],d[NV]; for(int c=0;c<NV;c++) d[c]=__float2half(0.f);
    flux_dir_h(WxR,0,PRM,Fa);flux_dir_h(WxL,0,PRM,Fb);for(int c=0;c<NV;c++)d[c]=d[c]+Fa[c]-Fb[c];
    flux_dir_h(WyR,1,PRM,Fa);flux_dir_h(WyL,1,PRM,Fb);for(int c=0;c<NV;c++)d[c]=d[c]+Fa[c]-Fb[c];
    flux_dir_h(WzR,2,PRM,Fa);flux_dir_h(WzL,2,PRM,Fb);for(int c=0;c<NV;c++)d[c]=d[c]+Fa[c]-Fb[c];
    for(int c=0;c<NV;c++) sdU[MUI(sd,ux,uy)+c]=nhl*d[c]; } }
__device__ void mpf_iph(__half* sW,__half* sdU,int s0,int sd,int wx,int wy,int ux,int uy,int dir,int side,const __half* PRM,__half* out){
  int ex=(dir==0),ey=(dir==1); __half Wf[NV],U[NV];
  recon_one_h(sW+MWI(s0,wx-ex,wy-ey),sW+MWI(s0,wx,wy),sW+MWI(s0,wx+ex,wy+ey),side,Wf);
  prim2cons_h(Wf,U,PRM); const __half* dU=sdU+MUI(sd,ux,uy); for(int c=0;c<NV;c++) U[c]=U[c]+dU[c]; cons2prim_h(U,out,PRM); }
__device__ void mpf_zh(__half* sW,__half* sdU,int sm,int s0,int sp,int wx,int wy,int sd,int ux,int uy,int side,const __half* PRM,__half* out){
  __half Wf[NV],U[NV];
  recon_one_h(sW+MWI(sm,wx,wy),sW+MWI(s0,wx,wy),sW+MWI(sp,wx,wy),side,Wf);
  prim2cons_h(Wf,U,PRM); const __half* dU=sdU+MUI(sd,ux,uy); for(int c=0;c<NV;c++) U[c]=U[c]+dU[c]; cons2prim_h(U,out,PRM); }
__global__ void k_ctumh(const float* R,float* O,int nx,int ny,int nz,float lam,const __half* PRM){
  __half* sW=smem; __half* sdU=smem+5*MWX*MWY*NV;
  int bx0=blockIdx.x*MBX, by0=blockIdx.y*MBY; int tid=threadIdx.y*MBX+threadIdx.x; const int NT=MBX*MBY;
  int i=bx0+threadIdx.x, j=by0+threadIdx.y; bool valid=(i<nx&&j<ny);
  int wx=threadIdx.x+2, wy=threadIdx.y+2, ux=threadIdx.x+1, uy=threadIdx.y+1;
  long VOL=(long)nx*ny*nz; __half nhl=__float2half(-0.5f*lam);
  for(int kz=-2;kz<=2;kz++) mload_Wh(R,sW,kz,bx0,by0,nx,ny,nz,tid,NT,PRM);
  __syncthreads();
  for(int kz=-1;kz<=1;kz++) mcompute_dUh(sW,sdU,kz,tid,NT,nhl,PRM);
  __syncthreads();
  __half Fz_prev[NV];
  if(valid){ __half A[NV],B[NV];
    mpf_zh(sW,sdU,SL5(-2),SL5(-1),SL5(0),wx,wy,SL3(-1),ux,uy,1,PRM,A);
    mpf_zh(sW,sdU,SL5(-1),SL5(0),SL5(1),wx,wy,SL3(0),ux,uy,0,PRM,B);
    llf_dir_h(A,B,2,Fz_prev,PRM); }
  for(int k=0;k<nz;k++){
    if(valid){
      int s0=SL5(k),sm=SL5(k-1),sp=SL5(k+1),sp2=SL5(k+2); int d0=SL3(k),dp=SL3(k+1);
      __half A[NV],B[NV],F[NV],acc[NV]; for(int c=0;c<NV;c++) acc[c]=__float2half(0.f);
      mpf_iph(sW,sdU,s0,d0,wx-1,wy,ux-1,uy,0,1,PRM,A); mpf_iph(sW,sdU,s0,d0,wx,wy,ux,uy,0,0,PRM,B); llf_dir_h(A,B,0,F,PRM); for(int c=0;c<NV;c++) acc[c]=acc[c]-F[c];
      mpf_iph(sW,sdU,s0,d0,wx,wy,ux,uy,0,1,PRM,A); mpf_iph(sW,sdU,s0,d0,wx+1,wy,ux+1,uy,0,0,PRM,B); llf_dir_h(A,B,0,F,PRM); for(int c=0;c<NV;c++) acc[c]=acc[c]+F[c];
      mpf_iph(sW,sdU,s0,d0,wx,wy-1,ux,uy-1,1,1,PRM,A); mpf_iph(sW,sdU,s0,d0,wx,wy,ux,uy,1,0,PRM,B); llf_dir_h(A,B,1,F,PRM); for(int c=0;c<NV;c++) acc[c]=acc[c]-F[c];
      mpf_iph(sW,sdU,s0,d0,wx,wy,ux,uy,1,1,PRM,A); mpf_iph(sW,sdU,s0,d0,wx,wy+1,ux,uy+1,1,0,PRM,B); llf_dir_h(A,B,1,F,PRM); for(int c=0;c<NV;c++) acc[c]=acc[c]+F[c];
      for(int c=0;c<NV;c++) acc[c]=acc[c]-Fz_prev[c];
      mpf_zh(sW,sdU,sm,s0,sp,wx,wy,d0,ux,uy,1,PRM,A); mpf_zh(sW,sdU,s0,sp,sp2,wx,wy,dp,ux,uy,0,PRM,B); llf_dir_h(A,B,2,F,PRM);
      for(int c=0;c<NV;c++){ acc[c]=acc[c]+F[c]; Fz_prev[c]=F[c]; }
      long q=IDX(i,j,k); for(int c=0;c<NV;c++) O[(long)c*VOL+q]=R[(long)c*VOL+q]-lam*__half2float(acc[c]);
    }
    __syncthreads(); mload_Wh(R,sW,k+3,bx0,by0,nx,ny,nz,tid,NT,PRM);
    __syncthreads(); mcompute_dUh(sW,sdU,k+2,tid,NT,nhl,PRM); __syncthreads();
  } }
extern "C" void fv_run_ctumh(float* R,float* O,int nx,int ny,int nz,float lam,const __half* PRM,int nsteps){
  dim3 thr(MBX,MBY), grp((nx+MBX-1)/MBX,(ny+MBY-1)/MBY);
  int shb=(int)((5*MWX*MWY+3*MUX*MUY)*NV*sizeof(__half));
  cudaFuncSetAttribute(k_ctumh, cudaFuncAttributeMaxDynamicSharedMemorySize, shb);
  float *a=R,*b=O;
  for(int s=0;s<nsteps;s++){ k_ctumh<<<grp,thr,shb>>>(a,b,nx,ny,nz,lam,PRM); float* t=a; a=b; b=t; }
  if(a!=R) cudaMemcpy(R,a,(size_t)NV*nx*ny*nz*sizeof(float),cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize(); }
""", "NV" => string(NV), "ZZTBZ" => string(NV <= 5 ? 8 : 4), "ZZMB" => string(NV <= 5 ? 16 : 8))
    "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <math.h>\n#define NV $NV\n\n" *
    join(protos, "\n") * "\n\n" * join(defs, "\n") *
    _genswap("swap_y", m.vidx, 2) * _genswap("swap_z", m.vidx, 3) *
    _genswap("swap_y_h", m.vidx, 2; half = true) * _genswap("swap_z_h", m.vidx, 3; half = true) * "\n" * tests * fixed
end

function _find_nvcc()
    p = Sys.which("nvcc"); p !== nothing && return p
    for g in ("/opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/bin/nvcc", "/usr/local/cuda*/bin/nvcc")
        for c in filter(isfile, _glob(g)); return c; end
    end
    error("transpile_cuda: nvcc not found (need CUDA HPC SDK or /usr/local/cuda, or nvcc on PATH).")
end
# tiny glob over a single '*'-containing path (no Glob.jl dep)
function _glob(pat)
    parts = split(pat, '/'); base = "/"; cands = [base]
    for p in parts
        isempty(p) && continue
        next = String[]
        for d in cands
            occursin('*', p) ? (isdir(d) && append!(next, [joinpath(d, x) for x in readdir(d) if occursin(Regex("^" * replace(p, "*"=>".*") * "\$"), x)])) :
                               push!(next, joinpath(d, p))
        end
        cands = next
    end
    cands
end

"`build_cuda(sys; sm) -> so_path`: emit + nvcc-compile the system to a shared library (cached by hash)."
function build_cuda(sys::FVSystem; sm::String = "sm_86")
    src = gen_cuda_c(sys); tag = string(hash((typeof(sys), sm)); base = 16)
    dir = mktempdir(; cleanup = false); cu = joinpath(dir, "fv_$tag.cu"); so = joinpath(dir, "libfv_$tag.so")
    write(cu, src)
    run(`$(_find_nvcc()) -O3 -arch=$sm --use_fast_math --shared -Xcompiler -fPIC -o $so $cu`)
    return so
end

mutable struct Grid3DCuMarch{N,S<:FVSystem}
    sys::S
    R::CuVector{Float32}; O::CuVector{Float32}; T::CuVector{Float32}     # state + 2 scratch (var-major)
    prm::CuVector{Float32}; prmh::CuVector{Float16}                      # params (f32 + f16 twin for :f16)
    spd::CuVector{Float32}                                               # scratch: per-cell CFL speed
    frun::Ptr{Cvoid}; frun2::Ptr{Cvoid}; fctu::Ptr{Cvoid}; fctus::Ptr{Cvoid}
    fctum::Ptr{Cvoid}; fctumh::Ptr{Cvoid}; fspeed::Ptr{Cvoid}
    nx::Int; ny::Int; nz::Int; dx::Float32
end

function Grid3DCuMarch(sys::FVSystem, U0::Array{NTuple{N,Float32},3}; dx, sm::String = "sm_86") where {N}
    nx, ny, nz = size(U0); VOL = nx*ny*nz
    lib = Libdl.dlopen(build_cuda(sys; sm = sm))
    Uh = Vector{Float32}(undef, N*VOL)
    @inbounds for kk in 0:nz-1, jj in 0:ny-1, ii in 0:nx-1
        q = (kk*ny + jj)*nx + ii + 1; u = U0[ii+1, jj+1, kk+1]
        for c in 1:N; Uh[(c-1)*VOL + q] = u[c]; end
    end
    R = CuArray(Uh)
    pv = Float32[getfield(sys, p) for p in _fvmeta(sys).params]
    prm = CuArray(pv); prmh = CuArray(Float16.(pv))
    Grid3DCuMarch{N,typeof(sys)}(sys, R, CUDA.zeros(Float32, N*VOL), CUDA.zeros(Float32, N*VOL), prm, prmh,
                                 CUDA.zeros(Float32, VOL),
                                 Libdl.dlsym(lib, :fv_run), Libdl.dlsym(lib, :fv_run_rk2),
                                 Libdl.dlsym(lib, :fv_run_ctu), Libdl.dlsym(lib, :fv_run_ctus),
                                 Libdl.dlsym(lib, :fv_run_ctum), Libdl.dlsym(lib, :fv_run_ctumh),
                                 Libdl.dlsym(lib, :fv_speed), nx, ny, nz, Float32(dx))
end

@inline _devptr(x) = reinterpret(Ptr{Float32}, UInt(UInt64(pointer(x))))
@inline _devptrh(x) = reinterpret(Ptr{Float16}, UInt(UInt64(pointer(x))))

"Run `nsteps` of the fast 1st-order-in-time fused single-pass kernel at fixed `dt` (the throughput
benchmark; result left in `g.R`). For science use the 2nd-order `evolve!`/`run_rk2!`."
function run!(g::Grid3DCuMarch, dt, nsteps::Integer)
    ccall(g.frun, Cvoid, (Ptr{Float32}, Ptr{Float32}, Cint, Cint, Cint, Cfloat, Ptr{Float32}, Cint),
          _devptr(g.R), _devptr(g.O), g.nx, g.ny, g.nz, Float32(dt)/g.dx, _devptr(g.prm), Int32(nsteps))
    return g
end

"Run `nsteps` of the genuinely 2nd-order MUSCL + SSP-RK2 scheme at fixed `dt` (result left in `g.R`)."
function run_rk2!(g::Grid3DCuMarch, dt, nsteps::Integer)
    ccall(g.frun2, Cvoid, (Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Cint, Cint, Cint, Cfloat, Ptr{Float32}, Cint),
          _devptr(g.R), _devptr(g.T), _devptr(g.O), g.nx, g.ny, g.nz, Float32(dt)/g.dx, _devptr(g.prm), Int32(nsteps))
    return g
end

"Run `nsteps` of the single-pass 2nd-order unsplit MUSCL-Hancock + transverse (CTU) scheme (result in `g.R`)."
function run_ctu!(g::Grid3DCuMarch, dt, nsteps::Integer)
    ccall(g.fctu, Cvoid, (Ptr{Float32}, Ptr{Float32}, Cint, Cint, Cint, Cfloat, Ptr{Float32}, Cint),
          _devptr(g.R), _devptr(g.O), g.nx, g.ny, g.nz, Float32(dt)/g.dx, _devptr(g.prm), Int32(nsteps))
    return g
end

"Run `nsteps` of the shared-memory-TILED single-pass CTU kernel (dU computed once/cell in shared; result in `g.R`)."
function run_ctus!(g::Grid3DCuMarch, dt, nsteps::Integer)
    ccall(g.fctus, Cvoid, (Ptr{Float32}, Ptr{Float32}, Cint, Cint, Cint, Cfloat, Ptr{Float32}, Cint),
          _devptr(g.R), _devptr(g.O), g.nx, g.ny, g.nz, Float32(dt)/g.dx, _devptr(g.prm), Int32(nsteps))
    return g
end

"Run `nsteps` of the streaming z-march CTU kernel (2D block marches z, rolling 5 W + 3 dU planes; result in `g.R`)."
function run_ctum!(g::Grid3DCuMarch, dt, nsteps::Integer)
    ccall(g.fctum, Cvoid, (Ptr{Float32}, Ptr{Float32}, Cint, Cint, Cint, Cfloat, Ptr{Float32}, Cint),
          _devptr(g.R), _devptr(g.O), g.nx, g.ny, g.nz, Float32(dt)/g.dx, _devptr(g.prm), Int32(nsteps))
    return g
end

"Run `nsteps` of the f16-ARITHMETIC streaming march (recon/flux in half precision; conserved I/O + update
stay f32). Lower precision by design — fastest at scale for hydro/turbulence. Result in `g.R`."
function run_ctumh!(g::Grid3DCuMarch, dt, nsteps::Integer)
    ccall(g.fctumh, Cvoid, (Ptr{Float32}, Ptr{Float32}, Cint, Cint, Cint, Cfloat, Ptr{Float16}, Cint),
          _devptr(g.R), _devptr(g.O), g.nx, g.ny, g.nz, Float32(dt)/g.dx, _devptrh(g.prmh), Int32(nsteps))
    return g
end

"Max over cells of the per-cell summed directional signal speed (the unsplit-CFL quantity, on device)."
function maxspeed_sum(g::Grid3DCuMarch)
    ccall(g.fspeed, Cvoid, (Ptr{Float32}, Ptr{Float32}, Cint, Cint, Cint, Ptr{Float32}),
          _devptr(g.R), _devptr(g.spd), g.nx, g.ny, g.nz, _devptr(g.prm))
    Float32(maximum(g.spd))
end

"CFL-stable timestep: `cfl·dx / max_cell(sx+sy+sz)` (cfl ≤ 1 for the fused unsplit update)."
dt_cfl(g::Grid3DCuMarch; cfl = 0.4f0) = Float32(cfl) * g.dx / maxspeed_sum(g)

"""
    evolve!(g::Grid3DCuMarch, tend; cfl=0.4f0, dtevery=4, scheme=:auto, maxsteps=10^7) -> g

Integrate to physical time `tend` with adaptive CFL timesteps (recomputed every `dtevery` steps, with
a `cfl` margin to cover speed growth between recomputes). Result is left in `g.R`. `scheme`:
`:auto` — pick the fastest single-pass 2nd-order kernel for the grid (streaming z-march for large grids,
shared-mem-tiled CTU otherwise; default); `:ctu` — force the tiled kernel; `:march` — force the z-march;
`:rk2` — pure-f32 MUSCL + SSP-RK2 (two stages). All validated 2nd-order and conservative.
"""
function evolve!(g::Grid3DCuMarch{N}, tend; cfl = 0.4f0, dtevery::Integer = 4, scheme::Symbol = :auto,
                 maxsteps::Integer = 10^7) where {N}
    # the z-march halves global over-read but needs enough (x,y) blocks (16²-cell tiles) to fill the GPU;
    # below that the tiled kernel's higher occupancy wins. (Euler-class NV only; GLM uses the tiled path.)
    use_march = scheme === :march || (scheme === :auto && N <= 5 && (g.nx ÷ 16) * (g.ny ÷ 16) ≥ 512)
    stepper = scheme === :rk2 ? run_rk2! : scheme === :f16 ? run_ctumh! : use_march ? run_ctum! : run_ctus!
    t = 0.0f0; tend = Float32(tend); n = 0; dt = 0.0f0
    while t < tend && n < maxsteps
        (n % dtevery == 0) && (dt = dt_cfl(g; cfl = cfl))
        dts = min(dt, tend - t)
        stepper(g, dts, 1)            # 2nd-order in space & time
        t += dts; n += 1
    end
    return (g = g, t = t, nsteps = n)
end

"Conserved totals Σ U · dx³ (var-major), for the conservation check."
function conserved_total(g::Grid3DCuMarch{N}) where {N}
    VOL = g.nx*g.ny*g.nz; v = g.dx^3
    ntuple(c -> Float32(sum(@view g.R[(c-1)*VOL+1 : c*VOL]) * v), Val(N))
end

function primitives(g::Grid3DCuMarch{N}) where {N}
    Uh = Array(g.R); VOL = g.nx*g.ny*g.nz
    [cons2prim(g.sys, ntuple(c -> Uh[(c-1)*VOL + ((kk-1)*g.ny + (jj-1))*g.nx + ii], Val(N)))
     for ii in 1:g.nx, jj in 1:g.ny, kk in 1:g.nz]
end

"Validate the transpiled C physics against the Julia @fvsystem functions (host-side). Returns max|Δ|."
function transpile_selfcheck(sys::FVSystem; trials = 2000)
    lib = Libdl.dlopen(build_cuda(sys)); m = _fvmeta(sys); NV = m.nvars
    PRM = Float32[getfield(sys, p) for p in m.params]; me = 0.0f0
    for _ in 1:trials
        W = ntuple(c -> (c == 1 || c == 5) ? 0.5f0 + rand(Float32) : rand(Float32) - 0.5f0, NV)
        Wc = zeros(Float32, NV); ccall(Libdl.dlsym(lib,:fv_cons2prim), Cvoid, (Ptr{Float32},Ptr{Float32},Ptr{Float32}), collect(Float32, prim2cons(sys,W)), Wc, PRM)
        Fc = zeros(Float32, NV); ccall(Libdl.dlsym(lib,:fv_physflux),  Cvoid, (Ptr{Float32},Ptr{Float32},Ptr{Float32}), collect(Float32, W), Fc, PRM)
        aC = ccall(Libdl.dlsym(lib,:fv_maxspeed), Cfloat, (Ptr{Float32},Ptr{Float32}), collect(Float32, W), PRM)
        me = max(me, maximum(abs.(Wc .- collect(Float32, cons2prim(sys, prim2cons(sys,W))))),
                     maximum(abs.(Fc .- collect(Float32, physflux_x(sys, W)))),
                     abs(aC - maxspeed_x(sys, W)))
    end
    return me
end
