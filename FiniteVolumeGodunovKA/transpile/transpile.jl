# ============================================================================================
# Transpile-to-CUDA-C performance backend — v0 proof of concept.
#
# THE POINT OF THE PROJECT: get full hand-written-.cu speed from ONE Julia @fvsystem stencil, by
# emitting the per-cell physics as CUDA-C, compiling with nvcc --use_fast_math, and running it over
# CuArrays (the march_bridge mechanism). The native CUDA.jl backend tops at ~40% of the .cu (codegen
# + split structure); this path targets the rest via nvcc's codegen + a fused single-pass kernel.
#
# v0 scope: Euler. (1) transpile the @fvsystem Euler physics Exprs (cons2prim/prim2cons/physflux_x/
# maxspeed_x) → C; (2) VALIDATE the transpiled C physics against the Julia functions (host-side, the
# part that proves the transpiler); (3) a fused 1st-order unsplit nvcc kernel for the throughput
# number vs the .cu 6865. The fused-march + f16 + PLM are the next tuning; this proves the path.
# ============================================================================================

using FiniteVolumeGodunovKA, CUDA, Libdl
const FV = FiniteVolumeGodunovKA

# ---- the Euler @fvsystem physics, as Exprs (these ARE the stencil; the macro captures the same) ----
const PHYS = (
    cons2prim = (:U, quote
        ρ, mx, my, mz, E = U
        iρ = inv(ρ)
        u, v, w = mx * iρ, my * iρ, mz * iρ
        P = (p.γ - 1) * (E - 0.5f0 * ρ * (u*u + v*v + w*w))
        (ρ, u, v, w, P)
    end, :tuple),
    prim2cons = (:W, quote
        ρ, u, v, w, P = W
        E = P / (p.γ - 1) + 0.5f0 * ρ * (u*u + v*v + w*w)
        (ρ, ρ*u, ρ*v, ρ*w, E)
    end, :tuple),
    physflux_x = (:W, quote
        ρ, u, v, w, P = W
        E = P / (p.γ - 1) + 0.5f0 * ρ * (u*u + v*v + w*w)
        (ρ*u, ρ*u*u + P, ρ*u*v, ρ*u*w, u * (E + P))
    end, :tuple),
    maxspeed_x = (:W, quote
        ρ, u, v, w, P = W
        abs(u) + sqrt(p.γ * P / ρ)
    end, :scalar),
)

# ---- Expr → CUDA-C ----
const NAMEMAP = Dict('ρ'=>"rho", 'γ'=>"gam", 'ψ'=>"psi", 'Δ'=>"D")
san(s::Symbol) = (cs = collect(string(s)); join(get(NAMEMAP, c, string(c)) for c in cs))
san(s::Symbol, ::Val{:strip}) = san(s)

function e2c(e)
    e isa Float32 && return string(e) * "f"
    e isa Float64 && return string(Float32(e)) * "f"
    e isa Int     && return string(e) * ".0f"
    e isa Symbol  && return san(e)
    if e isa Expr
        if e.head === :call
            op = e.args[1]; a = e.args[2:end]
            op === :inv  && return "(1.0f/(" * e2c(a[1]) * "))"
            op === :sqrt && return "sqrtf(" * e2c(a[1]) * ")"
            op === :abs  && return "fabsf(" * e2c(a[1]) * ")"
            op === :min  && return "fminf(" * e2c(a[1]) * "," * e2c(a[2]) * ")"
            op === :max  && return "fmaxf(" * e2c(a[1]) * "," * e2c(a[2]) * ")"
            op === :ifelse && return "((" * e2c(a[1]) * ")?(" * e2c(a[2]) * "):(" * e2c(a[3]) * "))"
            if op in (:+, :-, :*, :/)
                length(a) == 1 && op === :- && return "(-" * e2c(a[1]) * ")"
                s = e2c(a[1]); for x in a[2:end]; s = "(" * s * string(op) * e2c(x) * ")"; end
                return s
            end
            error("unsupported call: $op")
        elseif e.head === :.                     # p.γ  → the param name
            return san(e.args[2].value)
        end
    end
    error("unsupported expr: $e ($(typeof(e)))")
end

# emit a __host__ __device__ C function from a physics Expr
function genfunc(cname, arg, body, kind)
    stmts = String[]; ret = nothing
    items = filter(x -> !(x isa LineNumberNode), body.args)
    for (idx, s) in enumerate(items)
        last = idx == length(items)
        if last
            ret = s; continue
        end
        s isa Expr && s.head === :(=) || error("expected assignment: $s")
        lhs, rhs = s.args[1], s.args[2]
        if lhs isa Symbol
            push!(stmts, "  float $(san(lhs)) = $(e2c(rhs));")
        elseif lhs.head === :tuple
            names = [san(x) for x in lhs.args]
            if rhs isa Symbol                        # (a,b,..) = U  → from input array
                for (i, n) in enumerate(names); push!(stmts, "  float $n = $(san(rhs))[$(i-1)];"); end
            elseif rhs isa Expr && rhs.head === :tuple   # (a,b,..) = (e1,e2,..)
                for (n, ex) in zip(names, rhs.args); push!(stmts, "  float $n = $(e2c(ex));"); end
            else error("unsupported tuple-assign rhs: $rhs"); end
        end
    end
    if kind === :tuple
        ret isa Expr && ret.head === :tuple || error("expected return tuple")
        body_c = join(["  out[$(i-1)] = $(e2c(ex));" for (i, ex) in enumerate(ret.args)], "\n")
        return "__host__ __device__ void $cname(const float* $(san(arg)), float* out, float gam) {\n" *
               join(stmts, "\n") * (isempty(stmts) ? "" : "\n") * body_c * "\n}\n"
    else
        return "__host__ __device__ float $cname(const float* $(san(arg)), float gam) {\n" *
               join(stmts, "\n") * (isempty(stmts) ? "" : "\n") * "  return $(e2c(ret));\n}\n"
    end
end

# ---- build the full .cu (transpiled physics + host test ABI + fused kernel + run ABI) ----
function emit_cu()
    phys = genfunc("cons2prim", PHYS.cons2prim[1], PHYS.cons2prim[2], PHYS.cons2prim[3]) *
           genfunc("prim2cons", PHYS.prim2cons[1], PHYS.prim2cons[2], PHYS.prim2cons[3]) *
           genfunc("physflux_x", PHYS.physflux_x[1], PHYS.physflux_x[2], PHYS.physflux_x[3]) *
           genfunc("maxspeed_x", PHYS.maxspeed_x[1], PHYS.maxspeed_x[2], PHYS.maxspeed_x[3])
    fixed = raw"""

// host test exports — validate the transpiled physics against Julia
extern "C" {
  int  fv_nv() { return 5; }
  void fv_cons2prim(const float* U, float* W, float gam) { cons2prim(U, W, gam); }
  void fv_prim2cons(const float* W, float* U, float gam) { prim2cons(W, U, gam); }
  void fv_physflux (const float* W, float* F, float gam) { physflux_x(W, F, gam); }
  float fv_maxspeed(const float* W, float gam)           { return maxspeed_x(W, gam); }
}

// --- PLM MUSCL-Hancock (system-agnostic fixed C, calls the transpiled physics) ---
__device__ float mc(float a, float b) {                 // monotonized-central limiter
  if (a*b <= 0.f) return 0.f;
  float s = (a > 0.f) ? 1.f : -1.f;
  return s * fminf(0.5f*fabsf(a+b), fminf(2.f*fabsf(a), 2.f*fabsf(b)));
}
__device__ void halfstep(const float* Wm,const float* W0,const float* Wp, float lam, float gam,
                         float* WLh, float* WRh) {        // limited faces + Hancock dt/2 predictor
  float WL[5], WR[5];
  for (int c=0;c<5;c++){ float d=mc(W0[c]-Wm[c], Wp[c]-W0[c]); WL[c]=W0[c]-0.5f*d; WR[c]=W0[c]+0.5f*d; }
  float UL[5],UR[5],FL[5],FR[5]; prim2cons(WL,UL,gam); prim2cons(WR,UR,gam); physflux_x(WL,FL,gam); physflux_x(WR,FR,gam);
  float Ul[5],Ur[5]; for (int c=0;c<5;c++){ float dh=0.5f*lam*(FR[c]-FL[c]); Ul[c]=UL[c]-dh; Ur[c]=UR[c]-dh; }
  cons2prim(Ul,WLh,gam); cons2prim(Ur,WRh,gam);
}
__device__ void llf(const float* WL,const float* WR, float* F, float gam) {  // Rusanov (normal frame)
  float UL[5],UR[5],FL[5],FR[5]; prim2cons(WL,UL,gam); prim2cons(WR,UR,gam); physflux_x(WL,FL,gam); physflux_x(WR,FR,gam);
  float s = fmaxf(maxspeed_x(WL,gam), maxspeed_x(WR,gam));
  for (int c=0;c<5;c++) F[c] = 0.5f*(FL[c]+FR[c]) - 0.5f*s*(UR[c]-UL[c]);
}
// directional flux divergence (Fr-Fl) over a 5-cell stencil; (a,b)=swap into the normal frame (1,1 = x)
__device__ void fluxdiff(const float* const W[5], float lam, float gam, int a, int b, float* out) {
  float s[5][5];
  for (int m=0;m<5;m++){ for (int c=0;c<5;c++) s[m][c]=W[m][c]; if(a!=b){ float t=s[m][a]; s[m][a]=s[m][b]; s[m][b]=t; } }
  float WRm[5],WL0[5],WR0[5],WLp[5],dump[5];
  halfstep(s[0],s[1],s[2],lam,gam,dump,WRm);
  halfstep(s[1],s[2],s[3],lam,gam,WL0,WR0);
  halfstep(s[2],s[3],s[4],lam,gam,WLp,dump);
  float Fl[5],Fr[5]; llf(WRm,WL0,Fl,gam); llf(WR0,WLp,Fr,gam);
  for (int c=0;c<5;c++) out[c]=Fr[c]-Fl[c];
  if(a!=b){ float t=out[a]; out[a]=out[b]; out[b]=t; }
}

#define IDX(ii,jj,kk) ((((kk)*NY+(jj))*NX)+(ii))
// fused PLM (2nd-order) unsplit step: read the 13-cell star once, all 3 flux divergences, write once.
__global__ void k_step(const float* r0,const float* r1,const float* r2,const float* r3,const float* r4,
                       float* o0,float* o1,float* o2,float* o3,float* o4, float lam, float gam) {
  int i=blockIdx.x*blockDim.x+threadIdx.x, j=blockIdx.y*blockDim.y+threadIdx.y, k=blockIdx.z*blockDim.z+threadIdx.z;
  if (i>=NX||j>=NY||k>=NZ) return;
  #define LDP(W, ii,jj,kk) float W[5]; { int q=IDX(ii,jj,kk); float U[5]={r0[q],r1[q],r2[q],r3[q],r4[q]}; cons2prim(U,W,gam); }
  LDP(Wc,i,j,k)
  LDP(Xm2,(i-2+NX)%NX,j,k) LDP(Xm1,(i-1+NX)%NX,j,k) LDP(Xp1,(i+1)%NX,j,k) LDP(Xp2,(i+2)%NX,j,k)
  LDP(Ym2,i,(j-2+NY)%NY,k) LDP(Ym1,i,(j-1+NY)%NY,k) LDP(Yp1,i,(j+1)%NY,k) LDP(Yp2,i,(j+2)%NY,k)
  LDP(Zm2,i,j,(k-2+NZ)%NZ) LDP(Zm1,i,j,(k-1+NZ)%NZ) LDP(Zp1,i,j,(k+1)%NZ) LDP(Zp2,i,j,(k+2)%NZ)
  const float* xl[5]={Xm2,Xm1,Wc,Xp1,Xp2}; const float* yl[5]={Ym2,Ym1,Wc,Yp1,Yp2}; const float* zl[5]={Zm2,Zm1,Wc,Zp1,Zp2};
  float fx[5],fy[5],fz[5];
  fluxdiff(xl,lam,gam,1,1,fx); fluxdiff(yl,lam,gam,1,2,fy); fluxdiff(zl,lam,gam,1,3,fz);
  int q=IDX(i,j,k); const float* rr[5]={r0,r1,r2,r3,r4}; float* oo[5]={o0,o1,o2,o3,o4};
  for (int c=0;c<5;c++) oo[c][q] = rr[c][q] - lam*(fx[c]+fy[c]+fz[c]);
}

extern "C" double fv_run(float** q, float** o, float lam, float gam, int nsteps) {
  dim3 thr(8,8,4), grp((NX+7)/8,(NY+7)/8,(NZ+3)/4);
  float* a[5]; float* b[5];
  for (int c=0;c<5;c++){ a[c]=q[c]; b[c]=o[c]; }
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
  for (int s=0;s<nsteps;s++) {
    k_step<<<grp,thr>>>(a[0],a[1],a[2],a[3],a[4], b[0],b[1],b[2],b[3],b[4], lam, gam);
    for (int c=0;c<5;c++){ float* t=a[c]; a[c]=b[c]; b[c]=t; }          // swap buffers
  }
  cudaEventRecord(e1); cudaEventSynchronize(e1); float ms=0; cudaEventElapsedTime(&ms,e0,e1);
  return (double)ms;
}
"""
    "#include <cuda_runtime.h>\n#include <math.h>\n#ifndef NX\n#define NX 256\n#endif\n#ifndef NY\n#define NY 256\n#endif\n#ifndef NZ\n#define NZ 256\n#endif\n\n" *
    phys * fixed
end

# ---- write, compile, load ----
const NVCC = first(filter(x -> isfile(x), [
    "/opt/nvidia/hpc_sdk/Linux_x86_64/2026/cuda/13.1/bin/nvcc",
    "/usr/local/cuda/bin/nvcc"]))
function build(n)
    dir = @__DIR__; cu = joinpath(dir, "fv_euler.cu"); so = joinpath(dir, "libfv$(n).so")
    write(cu, emit_cu())
    cmd = `$NVCC -O3 -arch=sm_86 --use_fast_math -DNX=$n -DNY=$n -DNZ=$n --shared -Xcompiler -fPIC -o $so $cu`
    run(cmd); return so
end

println("=== emitted CUDA-C (physics) ===")
println(genfunc("cons2prim", PHYS.cons2prim[1], PHYS.cons2prim[2], PHYS.cons2prim[3]))

so = build(256)
h = dlopen(so)
println("nv = ", ccall(dlsym(h, :fv_nv), Cint, ()))

# ---- (1) validate the transpiled physics against the Julia @fvsystem functions ----
s = Euler(γ = 1.4f0); γ = 1.4f0
maxerr = 0.0f0
for _ in 1:2000
    W = (0.5f0 + rand(Float32), rand(Float32)-0.5f0, rand(Float32)-0.5f0, rand(Float32)-0.5f0, 0.5f0 + rand(Float32))
    U = collect(Float32, FV.prim2cons(s, W))
    Wc = zeros(Float32, 5); ccall(dlsym(h, :fv_cons2prim), Cvoid, (Ptr{Float32}, Ptr{Float32}, Cfloat), U, Wc, γ)
    Fc = zeros(Float32, 5); ccall(dlsym(h, :fv_physflux),  Cvoid, (Ptr{Float32}, Ptr{Float32}, Cfloat), collect(Float32, W), Fc, γ)
    aC = ccall(dlsym(h, :fv_maxspeed), Cfloat, (Ptr{Float32}, Cfloat), collect(Float32, W), γ)
    global maxerr = max(maxerr,
        maximum(abs.(Wc .- collect(Float32, FV.cons2prim(s, Tuple(U))))),
        maximum(abs.(Fc .- collect(Float32, FV.physflux_x(s, W)))),
        abs(aC - FV.maxspeed_x(s, W)))
end
println("transpiled-C physics vs Julia: max|Δ| over 2000 random states = ", maxerr)

# ---- (2) throughput of the fused nvcc kernel vs the .cu hydro 6865 ----
function bench(n, nsteps)
    so = build(n); h = dlopen(so)
    ph(i,j,k) = 0.001f0*Float32(mod(i*7+j*13+k*17, 911))
    planes = [CUDA.zeros(Float32, n*n*n) for _ in 1:5]
    Uh = [Array(p) for p in planes]
    for k in 0:n-1, j in 0:n-1, i in 0:n-1
        q = (k*n+j)*n+i+1; ρ=1f0+ph(i,j,k); P=1f0+ph(i,j,k)
        Uh[1][q]=ρ; Uh[2][q]=0.3f0*ρ; Uh[3][q]=0.2f0*ρ; Uh[4][q]=0.1f0*ρ
        Uh[5][q]=P/0.4f0 + 0.5f0*ρ*(0.3f0^2+0.2f0^2+0.1f0^2)
    end
    for c in 1:5; copyto!(planes[c], Uh[c]); end
    out = [CUDA.zeros(Float32, n*n*n) for _ in 1:5]
    qa = [reinterpret(Ptr{Float32}, UInt(UInt64(pointer(p)))) for p in planes]
    oa = [reinterpret(Ptr{Float32}, UInt(UInt64(pointer(p)))) for p in out]
    frun = dlsym(h, :fv_run)
    ccall(frun, Cdouble, (Ptr{Ptr{Float32}}, Ptr{Ptr{Float32}}, Cfloat, Cfloat, Cint), qa, oa, 1f-4, 1.4f0, 1)
    ms = ccall(frun, Cdouble, (Ptr{Ptr{Float32}}, Ptr{Ptr{Float32}}, Cfloat, Cfloat, Cint), qa, oa, 1f-4, 1.4f0, nsteps)
    n^3 * nsteps / (ms/1e3) / 1e6
end
for n in (256, 384, 480)
    m = bench(n, 30)
    println("transpiled nvcc fused-hydro nx=$n : $(round(m,digits=0)) Mcell/s   ($(round(100*m/6865,digits=0))% of .cu 6865)")
end
