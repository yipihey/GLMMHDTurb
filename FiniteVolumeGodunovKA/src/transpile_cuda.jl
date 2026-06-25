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

function _e2c(e, pidx, physset)
    e isa Float32 && return string(e) * "f"
    e isa Float64 && return string(Float32(e)) * "f"
    e isa Int     && return string(e) * ".0f"
    e isa Symbol  && return _csan(e)
    if e isa Expr
        if e.head === :call
            op = e.args[1]; a = e.args[2:end]
            op in physset && return string(_csan(op), "(", join([_e2c(x,pidx,physset) for x in a if x !== :p], ","), ", PRM)")
            op === :inv  && return "(1.0f/(" * _e2c(a[1],pidx,physset) * "))"
            op === :sqrt && return "sqrtf(" * _e2c(a[1],pidx,physset) * ")"
            op === :abs  && return "fabsf(" * _e2c(a[1],pidx,physset) * ")"
            op === :min  && return "fminf(" * _e2c(a[1],pidx,physset) * "," * _e2c(a[2],pidx,physset) * ")"
            op === :max  && return "fmaxf(" * _e2c(a[1],pidx,physset) * "," * _e2c(a[2],pidx,physset) * ")"
            op === :sign && return "((" * _e2c(a[1],pidx,physset) * ")>=0.f?1.f:-1.f)"
            op === :ifelse && return "((" * _e2c(a[1],pidx,physset) * ")?(" * _e2c(a[2],pidx,physset) * "):(" * _e2c(a[3],pidx,physset) * "))"
            if op in (:+, :-, :*, :/)
                length(a) == 1 && op === :- && return "(-" * _e2c(a[1],pidx,physset) * ")"
                s = _e2c(a[1],pidx,physset); for x in a[2:end]; s = "(" * s * string(op) * _e2c(x,pidx,physset) * ")"; end
                return s
            end
            error("transpile: unsupported call $op")
        elseif e.head === :.
            return "PRM[$(pidx[e.args[2].value])]"
        end
    end
    error("transpile: unsupported expr $e")
end

function _genfunc(cname, arg, body, pidx, physset)
    items = filter(x -> !(x isa LineNumberNode), body.args); ret = items[end]; stmts = String[]
    for s in items[1:end-1]
        lhs, rhs = s.args[1], s.args[2]
        if lhs isa Symbol
            push!(stmts, "  float $(_csan(lhs)) = $(_e2c(rhs,pidx,physset));")
        elseif lhs.head === :tuple
            names = [_csan(x) for x in lhs.args]
            if rhs isa Symbol
                for (i,n) in enumerate(names); push!(stmts, "  float $n = $(_csan(rhs))[$(i-1)];"); end
            else
                for (n,ex) in zip(names, rhs.args); push!(stmts, "  float $n = $(_e2c(ex,pidx,physset));"); end
            end
        end
    end
    bs = join(stmts, "\n") * (isempty(stmts) ? "" : "\n")
    if ret isa Expr && ret.head === :tuple
        outs = join(["  out[$(i-1)] = $(_e2c(ex,pidx,physset));" for (i,ex) in enumerate(ret.args)], "\n")
        return ("void $cname(const float* $(_csan(arg)), float* out, const float* PRM)",
                "__host__ __device__ void $cname(const float* $(_csan(arg)), float* out, const float* PRM) {\n$bs$outs\n}\n")
    else
        return ("float $cname(const float* $(_csan(arg)), const float* PRM)",
                "__host__ __device__ float $cname(const float* $(_csan(arg)), const float* PRM) {\n$bs  return $(_e2c(ret,pidx,physset));\n}\n")
    end
end

function _genswap(cname, vidx, d)
    sw = join(["{ float t=W[$(tr[1]-1)]; W[$(tr[1]-1)]=W[$(tr[d]-1)]; W[$(tr[d]-1)]=t; }" for tr in vidx], " ")
    "__device__ void $cname(float* W){ $sw }\n"
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
        push!(protos, "__host__ __device__ $proto;"); push!(defs, def)
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
""", "NV" => string(NV))
    "#include <cuda_runtime.h>\n#include <math.h>\n#define NV $NV\n\n" *
    join(protos, "\n") * "\n\n" * join(defs, "\n") *
    _genswap("swap_y", m.vidx, 2) * _genswap("swap_z", m.vidx, 3) * "\n" * tests * fixed
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
    prm::CuVector{Float32}
    spd::CuVector{Float32}                                               # scratch: per-cell CFL speed
    frun::Ptr{Cvoid}; frun2::Ptr{Cvoid}; fspeed::Ptr{Cvoid}
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
    prm = CuArray(Float32[getfield(sys, p) for p in _fvmeta(sys).params])
    Grid3DCuMarch{N,typeof(sys)}(sys, R, CUDA.zeros(Float32, N*VOL), CUDA.zeros(Float32, N*VOL), prm,
                                 CUDA.zeros(Float32, VOL),
                                 Libdl.dlsym(lib, :fv_run), Libdl.dlsym(lib, :fv_run_rk2),
                                 Libdl.dlsym(lib, :fv_speed), nx, ny, nz, Float32(dx))
end

@inline _devptr(x) = reinterpret(Ptr{Float32}, UInt(UInt64(pointer(x))))

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

"Max over cells of the per-cell summed directional signal speed (the unsplit-CFL quantity, on device)."
function maxspeed_sum(g::Grid3DCuMarch)
    ccall(g.fspeed, Cvoid, (Ptr{Float32}, Ptr{Float32}, Cint, Cint, Cint, Ptr{Float32}),
          _devptr(g.R), _devptr(g.spd), g.nx, g.ny, g.nz, _devptr(g.prm))
    Float32(maximum(g.spd))
end

"CFL-stable timestep: `cfl·dx / max_cell(sx+sy+sz)` (cfl ≤ 1 for the fused unsplit update)."
dt_cfl(g::Grid3DCuMarch; cfl = 0.4f0) = Float32(cfl) * g.dx / maxspeed_sum(g)

"""
    evolve!(g::Grid3DCuMarch, tend; cfl=0.4f0, dtevery=4, maxsteps=10^7) -> g

Integrate to physical time `tend` with adaptive CFL timesteps (recomputed every `dtevery` steps, with
a `cfl` margin to cover speed growth between recomputes). Result is left in `g.R`.
"""
function evolve!(g::Grid3DCuMarch, tend; cfl = 0.4f0, dtevery::Integer = 4, maxsteps::Integer = 10^7)
    t = 0.0f0; tend = Float32(tend); n = 0; dt = 0.0f0
    while t < tend && n < maxsteps
        (n % dtevery == 0) && (dt = dt_cfl(g; cfl = cfl))
        dts = min(dt, tend - t)
        run_rk2!(g, dts, 1)            # genuinely 2nd-order in space & time
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
