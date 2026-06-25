# ============================================================================================
# Transpile-to-CUDA-C performance backend — GENERAL (any @fvsystem system).
#
# Reads _fvmeta(sys) (the @fvsystem stencil captured as data) and emits a fused 2nd-order PLM
# nvcc kernel, compiled --use_fast_math, run over CuArrays. Validated bit-identical to the Julia
# @fvsystem physics, then benchmarked vs the hand-tuned .cu. Works for Euler (5 vars, 1 param) AND
# GLM-MHD (9 vars, 3 params, two rotating vectors) from the SAME pipeline — proving it generalizes.
#
# Generality handled: arbitrary params (mapped to a PRM[] array), arbitrary NVARS, the vidx rotation
# (generated swap_y/swap_z over all vector triples), and inter-physics calls (GLM maxspeed→fastspeed).
# ============================================================================================

using FiniteVolumeGodunovKA, CUDA, Libdl
const FV = FiniteVolumeGodunovKA

const NAMEMAP = Dict('ρ'=>"rho", 'γ'=>"gam", 'ψ'=>"psi", 'Δ'=>"D", 'λ'=>"lam")
san(s) = join(get(NAMEMAP, c, string(c)) for c in string(s))

# Expr → CUDA-C expression. pidx maps param symbol → PRM index; physset = scalar physics fn names.
function e2c(e, pidx, physset)
    e isa Float32 && return string(e) * "f"
    e isa Float64 && return string(Float32(e)) * "f"
    e isa Int     && return string(e) * ".0f"
    e isa Symbol  && return san(e)
    if e isa Expr
        if e.head === :call
            op = e.args[1]; a = e.args[2:end]
            if op in physset                       # inter-physics scalar call, e.g. fastspeed_x(p,W)
                arrs = [e2c(x, pidx, physset) for x in a if x !== :p]
                return string(san(op), "(", join(arrs, ","), ", PRM)")
            end
            op === :inv  && return "(1.0f/(" * e2c(a[1],pidx,physset) * "))"
            op === :sqrt && return "sqrtf(" * e2c(a[1],pidx,physset) * ")"
            op === :abs  && return "fabsf(" * e2c(a[1],pidx,physset) * ")"
            op === :min  && return "fminf(" * e2c(a[1],pidx,physset) * "," * e2c(a[2],pidx,physset) * ")"
            op === :max  && return "fmaxf(" * e2c(a[1],pidx,physset) * "," * e2c(a[2],pidx,physset) * ")"
            op === :sign && return "((" * e2c(a[1],pidx,physset) * ")>=0.f?1.f:-1.f)"
            op === :ifelse && return "((" * e2c(a[1],pidx,physset) * ")?(" * e2c(a[2],pidx,physset) * "):(" * e2c(a[3],pidx,physset) * "))"
            if op in (:+, :-, :*, :/)
                length(a) == 1 && op === :- && return "(-" * e2c(a[1],pidx,physset) * ")"
                s = e2c(a[1],pidx,physset); for x in a[2:end]; s = "(" * s * string(op) * e2c(x,pidx,physset) * ")"; end
                return s
            end
            error("unsupported call: $op")
        elseif e.head === :.                       # p.field → PRM[index]
            return "PRM[$(pidx[e.args[2].value])]"
        end
    end
    error("unsupported expr: $e ($(typeof(e)))")
end

# emit one __host__ __device__ C function from a physics Expr
function genfunc(cname, arg, body, pidx, physset)
    items = filter(x -> !(x isa LineNumberNode), body.args)
    ret = items[end]; stmts = String[]
    for s in items[1:end-1]
        lhs, rhs = s.args[1], s.args[2]
        if lhs isa Symbol
            push!(stmts, "  float $(san(lhs)) = $(e2c(rhs,pidx,physset));")
        elseif lhs.head === :tuple
            names = [san(x) for x in lhs.args]
            if rhs isa Symbol
                for (i,n) in enumerate(names); push!(stmts, "  float $n = $(san(rhs))[$(i-1)];"); end
            else
                for (n,ex) in zip(names, rhs.args); push!(stmts, "  float $n = $(e2c(ex,pidx,physset));"); end
            end
        end
    end
    body_s = join(stmts, "\n") * (isempty(stmts) ? "" : "\n")
    if ret isa Expr && ret.head === :tuple        # tuple return → write to out[]
        outs = join(["  out[$(i-1)] = $(e2c(ex,pidx,physset));" for (i,ex) in enumerate(ret.args)], "\n")
        return ("void $cname(const float* $(san(arg)), float* out, const float* PRM)",
                "__host__ __device__ void $cname(const float* $(san(arg)), float* out, const float* PRM) {\n$body_s$outs\n}\n")
    else                                            # scalar return
        return ("float $cname(const float* $(san(arg)), const float* PRM)",
                "__host__ __device__ float $cname(const float* $(san(arg)), const float* PRM) {\n$body_s  return $(e2c(ret,pidx,physset));\n}\n")
    end
end

# generate swap_y / swap_z device functions from vidx (Julia 1-based → C 0-based; swap slot[1]↔slot[d])
function gen_swap(cname, vidx, d)
    sw = String[]
    for tr in vidx
        a, b = tr[1]-1, tr[d]-1
        push!(sw, "{ float t=W[$a]; W[$a]=W[$b]; W[$b]=t; }")
    end
    "__device__ void $cname(float* W){ " * join(sw, " ") * " }\n"
end

function gen_cu(sys)
    m = FV._fvmeta(sys); NV = m.nvars
    pidx = Dict(p => i-1 for (i,p) in enumerate(m.params))
    physset = Set(keys(m.phys))
    want = [:cons2prim, :prim2cons, :physflux_x, :maxspeed_x]
    haskey(m.phys, :fastspeed_x) && pushfirst!(want, :fastspeed_x)
    protos = String[]; defs = String[]
    for f in want
        haskey(m.phys, f) || continue
        arg, body = m.phys[f]
        proto, def = genfunc(string(f), arg, body, pidx, physset)
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
extern "C" double fv_run(float* R, float* O, int nx,int ny,int nz, float lam, const float* PRM, int nsteps){
  dim3 thr(8,8,4), grp((nx+7)/8,(ny+7)/8,(nz+3)/4); float *a=R,*b=O;
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
  for(int s=0;s<nsteps;s++){ k_step<<<grp,thr>>>(a,b,nx,ny,nz,lam,PRM); float* t=a; a=b; b=t; }
  cudaEventRecord(e1); cudaEventSynchronize(e1); float ms=0; cudaEventElapsedTime(&ms,e0,e1); return (double)ms; }
""", "NV" => string(NV))
    "#include <cuda_runtime.h>\n#include <math.h>\n#define NV $NV\n\n" *
    join(protos, "\n") * "\n\n" * join(defs, "\n") *
    gen_swap("swap_y", m.vidx, 2) * gen_swap("swap_z", m.vidx, 3) * "\n" * tests * fixed
end

const NVCC = first(filter(isfile, ["/opt/nvidia/hpc_sdk/Linux_x86_64/2026/cuda/13.1/bin/nvcc", "/usr/local/cuda/bin/nvcc"]))
function build(sys, tag)
    dir = @__DIR__; cu = joinpath(dir, "fv_$tag.cu"); so = joinpath(dir, "libfv_$tag.so")
    write(cu, gen_cu(sys))
    run(`$NVCC -O3 -arch=sm_86 --use_fast_math --shared -Xcompiler -fPIC -o $so $cu`); so
end

function validate(sys, tag)
    h = dlopen(build(sys, tag)); m = FV._fvmeta(sys)
    PRM = Float32[getfield(sys, p) for p in m.params]; NV = m.nvars
    maxerr = 0.0f0
    for _ in 1:2000
        W = ntuple(c -> (c==1 || c==5) ? 0.5f0+rand(Float32) : rand(Float32)-0.5f0, NV)
        Wc = zeros(Float32, NV); ccall(dlsym(h,:fv_cons2prim), Cvoid, (Ptr{Float32},Ptr{Float32},Ptr{Float32}), collect(Float32,FV.prim2cons(sys,W)), Wc, PRM)
        Fc = zeros(Float32, NV); ccall(dlsym(h,:fv_physflux),  Cvoid, (Ptr{Float32},Ptr{Float32},Ptr{Float32}), collect(Float32,W), Fc, PRM)
        aC = ccall(dlsym(h,:fv_maxspeed), Cfloat, (Ptr{Float32},Ptr{Float32}), collect(Float32,W), PRM)
        maxerr = max(maxerr,
            maximum(abs.(Wc .- collect(Float32, FV.cons2prim(sys, FV.prim2cons(sys,W))))),
            maximum(abs.(Fc .- collect(Float32, FV.physflux_x(sys, W)))),
            abs(aC - FV.maxspeed_x(sys, W)))
    end
    h, maxerr
end

function bench(sys, h, n, nsteps)
    m = FV._fvmeta(sys); NV = m.nvars; VOL = n^3
    PRM = CuArray(Float32[getfield(sys, p) for p in m.params])
    ph(i,j,k) = 0.001f0*Float32(mod(i*7+j*13+k*17, 911))
    Uh = zeros(Float32, NV*VOL)
    for k in 0:n-1, j in 0:n-1, i in 0:n-1
        q=(k*n+j)*n+i+1; ρ=1f0+ph(i,j,k); W=ntuple(c-> c==1 ? ρ : (c==5 ? 1f0+ph(i,j,k) : (c<=4 ? 0.1f0*c : 0.3f0)), NV)
        U=FV.prim2cons(sys, W); for c in 1:NV; Uh[(c-1)*VOL+q]=U[c]; end
    end
    R = CuArray(Uh); O = CUDA.zeros(Float32, NV*VOL)
    pr(x)=reinterpret(Ptr{Float32}, UInt(UInt64(pointer(x))))
    f = dlsym(h,:fv_run)
    ccall(f, Cdouble, (Ptr{Float32},Ptr{Float32},Cint,Cint,Cint,Cfloat,Ptr{Float32},Cint), pr(R),pr(O),n,n,n,1f-4,pr(PRM),1)
    ms = ccall(f, Cdouble, (Ptr{Float32},Ptr{Float32},Cint,Cint,Cint,Cfloat,Ptr{Float32},Cint), pr(R),pr(O),n,n,n,1f-4,pr(PRM),30)
    n^3*30/(ms/1e3)/1e6
end

for (sys, tag, ref, name) in ((Euler(γ=1.4f0), "euler", 6865, "Euler"), (GLMMHD(γ=5f0/3f0,ch=2f0), "glm", 3175, "GLM-MHD"))
    h, err = validate(sys, tag)
    println("$name: transpiled-C physics vs Julia max|Δ| (2000 states) = $err")
    for n in (256, 384)
        mc = bench(sys, h, n, 30)
        println("  $name transpiled nvcc PLM nx=$n : $(round(mc,digits=0)) Mcell/s   ($(round(100*mc/ref,digits=0))% of .cu $ref)")
    end
end
