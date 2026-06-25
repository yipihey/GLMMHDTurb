# The @fvsystem contract surface.
#
#   @fvsystem Euler begin
#       nvars = 5
#       vidx  = (2, 3, 4)            # momentum components rotate under dim permutation
#       @params γ = 5f0/3f0          # one line per parameter (stored Float32)
#
#       cons2prim(U, p) = ...        # last arg `p` is the system params placeholder
#       prim2cons(W, p) = ...
#       physflux_x(W, p) = ...
#       maxspeed_x(W, p) = ...
#       eig_x(W, p) = ...            # optional (HLL/HLLC)
#   end
#
# Desugars to `struct Euler <: FVSystem; γ::Float32; end` (Trixi-style: idiomatic,
# dispatch-extensible) plus `@inline` methods on the contract functions. Each user
# function `f(args..., p)` becomes `f(_s::Euler, args...)` with `p` bound to `_s`.

# Contract function names are emitted BARE (unescaped) so macro hygiene resolves
# them to THIS module's generic functions — the user's methods extend ours
# regardless of what they imported. Only user fragments (the system name, params,
# arg names, bodies) are esc'd into the caller's scope.
const _CONTRACT_FNS = (:cons2prim, :prim2cons, :physflux_x, :maxspeed_x, :eig_x, :source, :fastspeed_x)
_check_contract(sym::Symbol) = sym in _CONTRACT_FNS ? sym :
    error("@fvsystem: `$sym` is not a contract function $(_CONTRACT_FNS)")

macro fvsystem(name, body)
    body isa Expr && body.head === :block || error("@fvsystem: expected a begin…end block")

    params   = Tuple{Symbol,Any}[]      # (field, default)
    nvars    = nothing
    vidxval  = nothing
    methods  = Expr[]

    for stmt in body.args
        stmt isa LineNumberNode && continue
        if stmt isa Expr && stmt.head === :(=) && stmt.args[1] isa Symbol
            key = stmt.args[1]
            key === :nvars ? (nvars   = stmt.args[2]) :
            key === :vidx  ? (vidxval = stmt.args[2]) :
            error("@fvsystem: unknown setting `$key` (expected nvars or vidx)")
        elseif stmt isa Expr && stmt.head === :macrocall && stmt.args[1] === Symbol("@params")
            for a in stmt.args[2:end]
                a isa LineNumberNode && continue
                a isa Expr && a.head === :(=) ||
                    error("@fvsystem: @params expects `name = default`")
                push!(params, (a.args[1]::Symbol, a.args[2]))
            end
        elseif stmt isa Expr && (stmt.head === :(=) || stmt.head === :function) &&
               stmt.args[1] isa Expr && stmt.args[1].head === :call
            push!(methods, stmt)        # a per-cell physics function
        else
            error("@fvsystem: unsupported statement: $stmt")
        end
    end
    nvars === nothing && error("@fvsystem: `nvars = N` is required")

    ename  = esc(name)
    fields = [:( $(esc(f))::Float32 )     for (f, _) in params]
    kwargs = [Expr(:kw, esc(f), esc(d))   for (f, d) in params]
    fnames = [esc(f)                      for (f, _) in params]

    # Build a method on one of OUR generic functions via GlobalRef, so hygiene does
    # not gensym the name in definition position (a bare symbol there becomes a new
    # local function and silently fails to dispatch). `@inline` is the lowered meta.
    mkmethod(fsym, argsig, body) = Expr(:function,
        Expr(:call, GlobalRef(@__MODULE__, fsym), argsig...),
        Expr(:block, Expr(:meta, :inline), body...))

    defs = Expr[]
    push!(defs, :( struct $(ename) <: FVSystem; $(fields...); end ))
    push!(defs, :( $(ename)(; $(kwargs...)) = $(ename)($(fnames...)) ))
    push!(defs, mkmethod(:nconserved, (Expr(:(::), ename),), (esc(nvars),)))
    vidxval !== nothing &&
        push!(defs, mkmethod(:vidx, (Expr(:(::), ename),), (esc(vidxval),)))

    for m in methods
        sig   = m.args[1]                       # Expr(:call, fname, args..., p)
        fbody = m.args[2]
        fname = _check_contract(sig.args[1])
        call  = sig.args[2:end]
        isempty(call) && error("@fvsystem: `$fname` needs at least the params arg")
        pname = call[end]                       # params placeholder (user's symbol)
        rest  = call[1:end-1]
        s      = gensym(:sys)
        argsig = (Expr(:(::), s, ename), (esc(a) for a in rest)...)
        body   = (Expr(:(=), esc(pname), s), esc(fbody))
        push!(defs, mkmethod(fname, argsig, body))
    end

    Expr(:block, defs...)
end
