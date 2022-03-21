@enum(_Operation, _OP_COEFFICIENT, _OP_VARIABLE, _OP_OPERATION,)

abstract type _AbstractFunction end

struct _Node
    head::_Operation
    hash::UInt
    index::Int
    operation::Symbol
    children::Union{Nothing,Vector{_Node}}

    function _Node(f::_AbstractFunction, coefficient::Float64)
        push!(f.data, coefficient)
        index = length(f.data)
        return new(
            _OP_COEFFICIENT,
            hash(_OP_COEFFICIENT),
            index,
            :NONE,
            nothing,
        )
    end
    function _Node(f::_AbstractFunction, variable::MOI.VariableIndex)
        index = get!(f.variables, variable, length(f.variables) + 1)
        return new(_OP_VARIABLE, hash(_OP_VARIABLE), index, :NONE, nothing)
    end
    function _Node(::_AbstractFunction, operation::Symbol, children::_Node...)
        h = hash(operation, _hash(children...))
        return new(_OP_OPERATION, h, UInt(0), operation, collect(children))
    end
end

_hash(a::_Node, args::_Node...) = hash(a.hash, _hash(args...))

_hash(child::_Node) = child.hash

mutable struct _Function <: _AbstractFunction
    variables::Dict{MOI.VariableIndex,Int}
    coefficients::Vector{Float64}
    data::Vector{Float64}
    expr::_Node
    function _Function(expr::Expr)
        f = new()
        f.variables = Dict{MOI.VariableIndex,Int}()
        f.data = Float64[]
        f.expr = _Node(f, expr)
        f.coefficients = zeros(length(f.variables))
        return f
    end
end

function _Node(f::_Function, expr::Expr)
    if isexpr(expr, :call)
        nodes = [_Node(f, expr.args[i]) for i in 2:length(expr.args)]
        return _Node(f, expr.args[1], nodes...)
    else
        @assert isexpr(expr, :ref)
        @assert expr.args[1] == :x
        return _Node(f, expr.args[2])
    end
end

function _expr_to_symbolics(expr::_Node, p, x)
    if expr.head == _OP_COEFFICIENT
        return p[expr.index]
    elseif expr.head == _OP_VARIABLE
        return x[expr.index]
    else
        @assert expr.head == _OP_OPERATION
        args = [_expr_to_symbolics(c, p, x) for c in expr.children]
        f = getfield(Base, expr.operation)
        return f(args...)
    end
end

function _nonlinear_constraint(expr::Expr)
    lower, upper, body = -Inf, Inf, :()
    if isexpr(expr, :call, 3)
        if expr.args[1] == :(>=)
            lower, upper, body = expr.args[1], expr.args[5], expr.args[3]
        elseif expr.args[1] == :(<=)
            lower, upper, body = -Inf, expr.args[3], expr.args[2]
        else
            @assert expr.args[1] == :(==)
            lower, upper, body = expr.args[3], expr.args[3], expr.args[2]
        end
    else
        @assert isexpr(expr, :comparison, 5)
        lower, upper, body = expr.args[1], expr.args[5], expr.args[3]
    end
    return _Function(body), MOI.NLPBoundsPair(lower, upper)
end

struct _SymbolicsFunction{F,G,H}
    f::F
    ∇f::G
    ∇²f::H
    g::Vector{Float64}
    H::Vector{Float64}
    H_structure::Vector{Tuple{Int,Int}}
end

function _SymbolicsFunction(f::_Function, features::Vector{Symbol})
    d, n = length(f.data), length(f.variables)
    Symbolics.@variables(p[1:d], x[1:n])
    p, x = collect(p), collect(x)
    f_expr = _expr_to_symbolics(f.expr, p, x)
    f = Symbolics.build_function(f_expr, p, x; expression = Val{false})
    if :Jac in features || :Grad in features
        ∇f_expr = Symbolics.gradient(f_expr, x)
        _, g! = Symbolics.build_function(∇f_expr, p, x; expression = Val{false})
        g_cache = zeros(length(x))
    else
        g!, g_cache = nothing, Float64[]
    end
    if :Hess in features
        ∇²f_expr_square = Symbolics.sparsejacobian(∇f_expr, x)
        # ∇²f_expr_square is sparse but it is also symmetrical. MathOptInterface
        # needs only the lower-triangular (technically, it doesn't matter which
        # triangle, but let's choose the lower-triangular for simplicity).
        I, J, V = SparseArrays.findnz(∇²f_expr_square)
        # Rather than using a SparseArray as storage, convert to MOI's
        # ((i, j), v) datastructure. This simplifies
        # `hessian_lagrangian_structure` calls later.
        ∇²f_expr = [V[i] for i in 1:length(I) if I[i] >= J[i]]
        ∇²f_structure = [(I[i], J[i]) for i in 1:length(I) if I[i] >= J[i]]
        _, h! =
            Symbolics.build_function(∇²f_expr, p, x; expression = Val{false})
        h_cache = zeros(length(∇²f_structure))
    else
        h!, h_cache, ∇²f_structure = nothing, Float64[], Tuple{Int,Int}[]
    end
    return _SymbolicsFunction(f, g!, h!, g_cache, h_cache, ∇²f_structure)
end

struct _ConstraintOffset
    index::Int
    ∇f_offset::Int
    ∇²f_offset::Int
end
mutable struct _NonlinearOracle <: MOI.AbstractNLPEvaluator
    use_threads::Bool
    objective::Union{Nothing,_Function}
    constraints::Vector{_Function}
    symbolic_functions::Dict{UInt,_SymbolicsFunction}
    offsets::Vector{Vector{_ConstraintOffset}}
    eval_constraint_timer::Float64
    eval_constraint_jacobian_timer::Float64
    eval_hessian_lagrangian_timer::Float64
end

function MOI.initialize(oracle::_NonlinearOracle, features::Vector{Symbol})
    oracle.eval_constraint_timer = 0.0
    oracle.eval_constraint_jacobian_timer = 0.0
    oracle.eval_hessian_lagrangian_timer = 0.0
    hashes = map(f -> f.expr.hash, oracle.constraints)
    for h in unique(hashes)
        f = oracle.constraints[findfirst(isequal(h), hashes)]
        oracle.symbolic_functions[h] = _SymbolicsFunction(f, features)
    end
    obj_hess_offset = 0
    if oracle.objective !== nothing
        f = oracle.objective
        h = f.expr.hash
        if !haskey(oracle.symbolic_functions, h)
            oracle.symbolic_functions[h] = _SymbolicsFunction(f, features)
            obj_hess_offset = length(oracle.symbolic_functions[h].H)
        end
    end
    if oracle.use_threads
        offsets = Dict{UInt,Vector{_ConstraintOffset}}(
            h => _ConstraintOffset[] for h in keys(oracle.symbolic_functions)
        )
        index, ∇f_offset, ∇²f_offset = 1, 1, obj_hess_offset + 1
        for f in oracle.constraints
            push!(
                offsets[f.expr.hash],
                _ConstraintOffset(index, ∇f_offset, ∇²f_offset),
            )
            index += 1
            symbolic_function = oracle.symbolic_functions[f.expr.hash]
            ∇f_offset += length(symbolic_function.g)
            ∇²f_offset += length(symbolic_function.H)
        end
        for v in values(offsets)
            push!(oracle.offsets, v)
        end
    end
    return
end

MOI.features_available(::_NonlinearOracle) = [:Grad, :Jac, :Hess]

function _eval_function(
    oracle::_NonlinearOracle,
    func::_Function,
    x::AbstractVector{Float64},
)
    for (v, index) in func.variables
        @inbounds func.coefficients[index] = x[v.value]
    end
    @inbounds f = oracle.symbolic_functions[func.expr.hash]
    return f.f(func.data, func.coefficients)::Float64
end

function _eval_gradient(
    oracle::_NonlinearOracle,
    func::_Function,
    x::AbstractVector{Float64},
)
    for (v, index) in func.variables
        @inbounds func.coefficients[index] = x[v.value]
    end
    @inbounds f = oracle.symbolic_functions[func.expr.hash]
    f.∇f(f.g, func.data, func.coefficients)
    return f.g
end

function _eval_hessian(
    oracle::_NonlinearOracle,
    func::_Function,
    x::AbstractVector{Float64},
)
    for (v, index) in func.variables
        @inbounds func.coefficients[index] = x[v.value]
    end
    @inbounds f = oracle.symbolic_functions[func.expr.hash]
    f.∇²f(f.H, func.data, func.coefficients)
    return f.H
end

function MOI.eval_objective(
    oracle::_NonlinearOracle,
    x::AbstractVector{Float64},
)
    @assert oracle.objective !== nothing
    return _eval_function(oracle, oracle.objective, x)
end

function MOI.eval_objective_gradient(
    oracle::_NonlinearOracle,
    g::AbstractVector{Float64},
    x::AbstractVector{Float64},
)
    @assert oracle.objective !== nothing
    ∇f = _eval_gradient(oracle, oracle.objective, x)
    fill!(g, 0.0)
    for (k, v) in oracle.objective.variables
        g[k.value] = ∇f[v]
    end
    return
end

function MOI.eval_constraint(
    oracle::_NonlinearOracle,
    g::AbstractVector{Float64},
    x::AbstractVector{Float64},
)
    start = time()
    if oracle.use_threads
        Threads.@threads for offsets in oracle.offsets
            for c in offsets
                func = oracle.constraints[c.index]
                @inbounds g[c.index] = _eval_function(oracle, func, x)
            end
        end
    else
        for row in 1:length(g)
            g[row] = _eval_function(oracle, oracle.constraints[row], x)
        end
    end
    oracle.eval_constraint_timer += time() - start
    return
end

function MOI.jacobian_structure(oracle::_NonlinearOracle)
    structure = Tuple{Int,Int}[]
    for row in 1:length(oracle.constraints)
        c = oracle.constraints[row]
        for x in sort(collect(keys(c.variables)); by = v -> c.variables[v])
            push!(structure, (row, x.value))
        end
    end
    return structure
end

function MOI.eval_constraint_jacobian(
    oracle::_NonlinearOracle,
    J::AbstractVector{Float64},
    x::AbstractVector{Float64},
)
    start = time()
    if oracle.use_threads
        Threads.@threads for offset in oracle.offsets
            for c in offset
                func = oracle.constraints[c.index]
                g = _eval_gradient(oracle, func, x)
                for i in 1:length(g)
                    @inbounds J[c.∇f_offset+i-1] = g[i]
                end
            end
        end
    else
        k = 1
        for i in 1:length(oracle.constraints)
            g = _eval_gradient(oracle, oracle.constraints[i], x)
            for gi in g
                J[k] = gi
                k += 1
            end
        end
    end
    oracle.eval_constraint_jacobian_timer += time() - start
    return
end

_hessian_lagrangian_structure(::Any, ::Any, ::Nothing) = nothing

function _hessian_lagrangian_structure(structure, oracle, c)
    f = oracle.symbolic_functions[c.expr.hash]
    H_to_x = Dict(v => k.value for (k, v) in c.variables)
    for (i, j) in f.H_structure
        row, col = H_to_x[i], H_to_x[j]
        push!(structure, row >= col ? (row, col) : (col, row))
    end
    return
end

function MOI.hessian_lagrangian_structure(oracle::_NonlinearOracle)
    structure = Tuple{Int,Int}[]
    _hessian_lagrangian_structure(structure, oracle, oracle.objective)
    for c in oracle.constraints
        _hessian_lagrangian_structure(structure, oracle, c)
    end
    return structure
end

function MOI.eval_hessian_lagrangian(
    oracle::_NonlinearOracle,
    H::AbstractVector{Float64},
    x::AbstractVector{Float64},
    σ::Float64,
    μ::AbstractVector{Float64},
)
    start = time()
    hessian_offset = 0
    if oracle.objective !== nothing
        h = _eval_hessian(oracle, oracle.objective, x)
        for i in 1:length(h)
            H[i] = σ * h[i]
        end
        hessian_offset += length(h)
    end
    if oracle.use_threads
        Threads.@threads for offset in oracle.offsets
            for c in offset
                func = oracle.constraints[c.index]
                h = _eval_hessian(oracle, func, x)
                for i in 1:length(h)
                    @inbounds H[c.∇²f_offset+i-1] = μ[c.index] * h[i]
                end
            end
        end
    else
        k = hessian_offset + 1
        for i in 1:length(oracle.constraints)
            h = _eval_hessian(oracle, oracle.constraints[i], x)
            for nzval in h
                H[k] = μ[i] * nzval
                k += 1
            end
        end
    end
    oracle.eval_hessian_lagrangian_timer += time() - start
    return
end

function _to_sparse(IJ, V)
    I, J = [i for (i, _) in IJ], [j for (_, j) in IJ]
    n = max(maximum(I), maximum(J))
    return SparseArrays.sparse(I, J, V, n, n)
end

"""
    nlp_block_data(d::AbstractNLPEvaluator; use_threads::Bool = false)

Convert an AbstractNLPEvaluator into a SymbolicAD instance of MOI.NLPBlockData.
"""
function nlp_block_data(d::MOI.AbstractNLPEvaluator; use_threads::Bool = false)
    MOI.initialize(d, [:ExprGraph])
    objective = d.has_nlobj ? _Function(MOI.objective_expr(d)) : nothing
    n = length(d.constraints)
    functions = Vector{_Function}(undef, n)
    bounds = Vector{MOI.NLPBoundsPair}(undef, n)
    for i in 1:n
        f, bound = _nonlinear_constraint(MOI.constraint_expr(d, i))
        functions[i] = f
        bounds[i] = bound
    end
    oracle = _NonlinearOracle(
        use_threads,
        objective,
        functions,
        Dict{UInt,_SymbolicsFunction}(),
        Vector{_ConstraintOffset}[],
        0.0,
        0.0,
        0.0,
    )
    return MOI.NLPBlockData(bounds, oracle, d.has_nlobj)
end
