# Copyright (c) 2022, Oscar Dowson and contributors
# Copyright (c) 2022, Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

abstract type _AbstractFunction end

"""
    _Operation

An enum representing different node types in a symbolic expression graph.

 * `_OP_COEFFICIENT`: a numerical `Float64`
 * `_OP_VARIABLE`: a decision variable
 * `_OP_OPERATION`: a function call
 * `_OP_INTEGER_EXPONENT`: a special case for handling integer polynomials such
   as `x^2`, since the `2.0` is usually structural rather than a coefficient
   that differs between expressions.
"""
@enum(
    _Operation,
    _OP_COEFFICIENT,
    _OP_VARIABLE,
    _OP_OPERATION,
    _OP_INTEGER_EXPONENT,
)

"""
    _Node

A type to represent a node in the symbolic expression graph.

!!! warning
    `_Node`s can only be used in conjunction with `_Function`.

`_Node` has the following fields:

 * `.head::_Operation`: this describes the node type
 * `.hash::UInt64`: a hash of the expression tree rooted at this node. See the
   Fast hashing section for more details.
 * `.index::Int` : interpretation depends on `.head`. If head is:
   * `_OP_COEFFICIENT`, the index into `f.data`
   * `_OP_VARIABLE`, the index into `f.ordered_variables`
   * `_OP_INTEGER_EXPONENT`, the integer exponent `children[1]^index`
 * `.operation::Symbol` : the symbolic name of the function call, if `.head` is
   `_OP_OPERATION`, otherwise this can be ignored.
 * `.children::Union{Nothing,Vector{_Node}}` : a vector of children of the node,
   if `.head` is `_OP_OPERATION`, otherwise `nothing`.

## Fast hashing

The expression graph is structured as a Merkle tree (a.k.a. a hash tree). That
is, each node contains a field `.hash`, which is the hash of the node's
operation with the hash of its children. Since children are also nodes, hashing
a child is an O(1) lookup of `child.hash`. Moreover, two expressions are
structurally identical if the have the same hash when converted into `_Node`
form.

## Other considerations

Through experience, the representation of the symbolic expression graph is not
the bottleneck in a nonlinear optimization model. Therefore, we choose the
somewhat inefficient storage rpresentation of
`children::Union{Nothing,Vector{_Node}}` for the list of children. This results
in a GC tracked object for each node that has children. If memory issues are
identified in future, changing how we represent the expression graph is one
promising avenue of investigation.
"""
struct _Node
    head::_Operation
    hash::UInt
    index::Int
    operation::Symbol
    children::Union{Nothing,Vector{_Node}}

    function _Node(f::_AbstractFunction, coefficient::Float64)
        # Store the coefficient
        push!(f.data, coefficient)
        # Compute the hash of the node. Because all coefficients are alike, this
        # is just the hash of `_OP_COEFFICIENT`.
        h = hash(_OP_COEFFICIENT)
        # Index is the length of `f.data` when we need to look it up again.
        index = length(f.data)
        return new(_OP_COEFFICIENT, h, index, :NONE, nothing)
    end
    function _Node(f::_AbstractFunction, variable::MOI.VariableIndex)
        # We need to convert the variable to the order in which it will appear
        # in .ordered_variables. Currently, the order is stored in
        # `f.variables`...
        lookup = f.variables::Dict{MOI.VariableIndex,Int}
        # And if the variable hasn't been seen yet, add a new index
        index = get!(lookup, variable, length(lookup) + 1)
        # The hash of this node depends on the variable and it's position in
        # ordered_variables.
        h = hash(_OP_VARIABLE, hash(index))
        return new(_OP_VARIABLE, h, index, :NONE, nothing)
    end
    function _Node(::_AbstractFunction, operation::Symbol, children::_Node...)
        # Okay, a function call. The hash depends on the operation and the
        # children.
        h = hash(operation, _hash(children...))
        return new(_OP_OPERATION, h, 0, operation, collect(children))
    end
    # Special cased performance optimization: x^N where N::Int is a common
    # operation that would otherwise get re-written to x[i]^p[j]. Doing so
    # disables a lot of potential optimizations, so it's worth special-casing
    # this. Other operations aren't as important.
    function _Node(::_AbstractFunction, ::typeof(^), x::_Node, N::Int)
        # The hash depends on ^, as well as the child node `x` and the
        # exponent `N`
        h = hash(_OP_INTEGER_EXPONENT, hash(_hash(x), hash(N)))
        return new(_OP_INTEGER_EXPONENT, h, N, :^, [x])
    end
end

# The hash of a child is an O(1) lookup
_hash(child::_Node) = child.hash

# Recursively has nodes together
_hash(a::_Node, args::_Node...) = hash(a.hash, _hash(args...))

"""
    _Function(model::MOI.Nonlinear.Model, expr::Expr)

This function parses the Julia expression `expr` into a `_Function`.

`_Function` has the following fields:

 * `ordered_variables`: a vector containing the `.value` indices of each
   variable in the order that they are discovered in the expression tree.
 * `coefficients::Vector{Float64}`: a vector to store the numerical input values
   of `x`.
 * `data::Vector{Float64}`: the list of numerical coefficients that appear in
   `expr`
 * `expr::_Node`: the symbolic representation of `expr`.

For more information on how to interpret the fields, see `_Node`.
"""
mutable struct _Function <: _AbstractFunction
    ordered_variables::Vector{Int}
    coefficients::Vector{Float64}
    data::Vector{Float64}
    expr::_Node
    # A field that is only used when constructing the _Function
    variables::Union{Nothing,Dict{MOI.VariableIndex,Int}}
    model::MOI.Nonlinear.Model
    function _Function(model::MOI.Nonlinear.Model, expr::Expr)
        f = new()
        f.variables = Dict{MOI.VariableIndex,Int}()
        f.data = Float64[]
        f.expr = _Node(f, expr)
        f.coefficients = zeros(length(f.variables))
        f.ordered_variables = zeros(Int, length(f.variables))
        for (k, v) in f.variables
            f.ordered_variables[v] = k.value
        end
        f.variables = nothing
        f.model = model
        return f
    end
end

_is_integer(::Any) = false
_is_integer(x::Real) = Base.isinteger(x)

"""
    _Node(f::_Function, expr::Expr)

Parse `expr` into `f` and return a `_Node`.

!!! warning
    This function gets called recursively.
"""
function _Node(f::_Function, expr::Expr)
    if Meta.isexpr(expr, :||) || Meta.isexpr(expr, :&&)
        return _Node(f, expr.head, (_Node(f, arg) for arg in expr.args)...)
    end
    @assert Meta.isexpr(expr, :call)
    # Performance optimization: most calls will be unary or binary
    # operators. Therefore, we can specialize an if-statement to handle the
    # common cases without needing to splat.
    if length(expr.args) == 2
        return _Node(f, expr.args[1], _Node(f, expr.args[2]))
    elseif length(expr.args) == 3
        # Special cased performance optimization for _OP_INTEGER_EXPONENT
        if expr.args[1] == :^ && _is_integer(expr.args[3])
            return _Node(f, ^, _Node(f, expr.args[2]), Int(expr.args[3]))
        end
        return _Node(
            f,
            expr.args[1],
            _Node(f, expr.args[2]),
            _Node(f, expr.args[3]),
        )
    else
        nodes = (_Node(f, expr.args[i]) for i in 2:length(expr.args))
        return _Node(f, expr.args[1], nodes...)
    end
end

struct _SymbolicsFunction{F,G,H}
    f::F
    ∇f::G
    ∇²f::H
    g::Vector{Float64}
    H::Vector{Float64}
    H_structure::Vector{Tuple{Int,Int}}
end

"""
    _expr_to_symbolics(model::MOI.Nonlinear.Model, expr::_Node, p, x)

Convert a `_Node` into a `Symbolics.jl` expression via operator overloading.
"""
function _expr_to_symbolics(model::MOI.Nonlinear.Model, expr::_Node, p, x)
    if expr.head == _OP_COEFFICIENT
        return p[expr.index]
    elseif expr.head == _OP_VARIABLE
        return x[expr.index]
    elseif expr.head == _OP_OPERATION
        args = [_expr_to_symbolics(model, c, p, x) for c in expr.children]
        if hasproperty(Base, expr.operation)
            return getproperty(Base, expr.operation)(args...)
        elseif expr.operation == :&&
            return (&)(args...)
        elseif expr.operation == :||
            return (|)(args...)
        end
        # If the function isn't defined in Base, defer to the operator registry.
        # We don't do this for all functions, because MOI uses NaNMath, which
        # doesn't work with Symbolics. But this will work with SpecialFunctions
        # and user-defined functions (which don't use NaNMath).
        if length(args) == 1
            return MOI.Nonlinear.eval_univariate_function(
                model.operators,
                expr.operation,
                args[1],
            )
        else
            return MOI.Nonlinear.eval_multivariate_function(
                model.operators,
                expr.operation,
                args,
            )
        end
    else
        @assert expr.head == _OP_INTEGER_EXPONENT
        return _expr_to_symbolics(model, expr.children[1], p, x)^expr.index
    end
end

function _SymbolicsFunction(f::_Function, features::Vector{Symbol})
    d, n = length(f.data), length(f.ordered_variables)
    Symbolics.@variables(p[1:d], x[1:n])
    p, x = collect(p), collect(x)
    f_expr = _expr_to_symbolics(f.model, f.expr, p, x)
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

"""
    DefaultBackend() <: AbstractSymbolicBackend

A simple implementation that loops over the different constraints.
"""
struct DefaultBackend <: AbstractSymbolicBackend end

mutable struct _NonlinearOracle{B} <: MOI.AbstractNLPEvaluator
    backend::B
    objective::Union{Nothing,_Function}
    constraints::Vector{_Function}
    symbolic_functions::Dict{UInt,_SymbolicsFunction}
    hessian_sparsity_map::Vector{Int}
    function _NonlinearOracle(backend::B, objective, constraints) where {B}
        return new{B}(
            backend,
            objective,
            constraints,
            Dict{UInt,_SymbolicsFunction}(),
            Int[],
        )
    end
end

function MOI.initialize(
    ::_NonlinearOracle,
    ::AbstractSymbolicBackend,
    ::Vector{Symbol},
)
    return
end

function MOI.initialize(oracle::_NonlinearOracle, features::Vector{Symbol})
    hashes = map(f -> f.expr.hash, oracle.constraints)
    for h in unique(hashes)
        f = oracle.constraints[findfirst(isequal(h), hashes)]
        oracle.symbolic_functions[h] = _SymbolicsFunction(f, features)
    end
    if oracle.objective !== nothing
        f = oracle.objective
        h = f.expr.hash
        if !haskey(oracle.symbolic_functions, h)
            oracle.symbolic_functions[h] = _SymbolicsFunction(f, features)
        end
    end
    MOI.initialize(oracle, oracle.backend, features)
    return
end

MOI.features_available(::_NonlinearOracle) = [:Grad, :Jac, :Hess]

function _update_coefficients(f::_Function, x::AbstractVector{Float64})
    for (i, v) in enumerate(f.ordered_variables)
        @inbounds f.coefficients[i] = x[v]
    end
    return
end

function _eval_function(
    oracle::_NonlinearOracle,
    func::_Function,
    x::AbstractVector{Float64},
)
    _update_coefficients(func, x)
    @inbounds f = oracle.symbolic_functions[func.expr.hash]
    return f.f(func.data, func.coefficients)::Float64
end

function _eval_gradient(
    oracle::_NonlinearOracle,
    func::_Function,
    x::AbstractVector{Float64},
)
    _update_coefficients(func, x)
    @inbounds f = oracle.symbolic_functions[func.expr.hash]
    f.∇f(f.g, func.data, func.coefficients)
    return f.g
end

function _eval_hessian(
    oracle::_NonlinearOracle,
    func::_Function,
    x::AbstractVector{Float64},
)
    _update_coefficients(func, x)
    @inbounds f = oracle.symbolic_functions[func.expr.hash]
    f.∇²f(f.H, func.data, func.coefficients)
    return f.H
end

function MOI.eval_objective(
    oracle::_NonlinearOracle,
    x::AbstractVector{Float64},
)
    @assert oracle.objective !== nothing
    objective_value = _eval_function(oracle, oracle.objective, x)
    return objective_value
end

function MOI.eval_objective_gradient(
    oracle::_NonlinearOracle,
    g::AbstractVector{Float64},
    x::AbstractVector{Float64},
)
    @assert oracle.objective !== nothing
    ∇f = _eval_gradient(oracle, oracle.objective, x)
    fill!(g, 0.0)
    for (i, v) in enumerate(oracle.objective.ordered_variables)
        @inbounds g[v] = ∇f[i]
    end
    return
end

function MOI.eval_constraint(
    oracle::_NonlinearOracle,
    g::AbstractVector{Float64},
    x::AbstractVector{Float64},
)
    for (row, c) in enumerate(oracle.constraints)
        g[row] = _eval_function(oracle, c, x)
    end
    return
end

function MOI.jacobian_structure(oracle::_NonlinearOracle)
    structure = Tuple{Int,Int}[]
    for (row, c) in enumerate(oracle.constraints)
        for col in c.ordered_variables
            push!(structure, (row, col))
        end
    end
    return structure
end

function MOI.eval_constraint_jacobian(
    oracle::_NonlinearOracle,
    J::AbstractVector{Float64},
    x::AbstractVector{Float64},
)
    k = 1
    for c in oracle.constraints
        g = _eval_gradient(oracle, c, x)
        for gi in g
            J[k] = gi
            k += 1
        end
    end
    return
end

function _hessian_lagrangian_structure(
    ::Vector{Tuple{Int,Int}},
    ::_NonlinearOracle,
    ::Any,
    ::Nothing,
)
    return
end

function _hessian_lagrangian_structure(
    structure::Vector{Tuple{Int,Int}},
    oracle::_NonlinearOracle,
    map::Dict{Tuple{Int,Int},Int},
    c::_Function,
)
    f = oracle.symbolic_functions[c.expr.hash]
    for (i, j) in f.H_structure
        row, col = c.ordered_variables[i], c.ordered_variables[j]
        row_col = row >= col ? (row, col) : (col, row)
        index = get(map, row_col, nothing)
        if index === nothing
            push!(structure, row_col)
            map[row_col] = length(structure)
            push!(oracle.hessian_sparsity_map, length(structure))
        else
            push!(oracle.hessian_sparsity_map, index::Int)
        end
    end
    return
end

function MOI.hessian_lagrangian_structure(oracle::_NonlinearOracle)
    structure = Tuple{Int,Int}[]
    map = Dict{Tuple{Int,Int},Int}()
    _hessian_lagrangian_structure(structure, oracle, map, oracle.objective)
    for c in oracle.constraints
        _hessian_lagrangian_structure(structure, oracle, map, c)
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
    fill!(H, 0.0)
    k = 1
    if oracle.objective !== nothing && !iszero(σ)
        h = _eval_hessian(oracle, oracle.objective, x)
        for (i, hi) in enumerate(h)
            @inbounds H[oracle.hessian_sparsity_map[i]] += σ * hi
        end
        k += length(h)
    end
    for (μi, constraint) in zip(μ, oracle.constraints)
        if iszero(μi)
            continue
        end
        h = _eval_hessian(oracle, constraint, x)
        for nzval in h
            @inbounds H[oracle.hessian_sparsity_map[k]] += μi * nzval
            k += 1
        end
    end
    return
end

###
### MOI integration
###

function _to_expr(
    data::MOI.Nonlinear.Model,
    expr::MOI.Nonlinear.Expression,
    variable_order::Dict{Int64,Int},
    subexpressions::Vector{Any},
)
    tree = Any[]
    for node in expr.nodes
        node_expr = if node.type == MOI.Nonlinear.NODE_CALL_MULTIVARIATE
            Expr(:call, data.operators.multivariate_operators[node.index])
        elseif node.type == MOI.Nonlinear.NODE_CALL_UNIVARIATE
            Expr(:call, data.operators.univariate_operators[node.index])
        elseif node.type == MOI.Nonlinear.NODE_SUBEXPRESSION
            subexpressions[node.index]
        elseif node.type == MOI.Nonlinear.NODE_MOI_VARIABLE
            MOI.VariableIndex(variable_order[node.index])
        elseif node.type == MOI.Nonlinear.NODE_PARAMETER
            data.parameters[node.index]
        elseif node.type == MOI.Nonlinear.NODE_LOGIC
            Expr(data.operators.logic_operators[node.index])
        elseif node.type == MOI.Nonlinear.NODE_COMPARISON
            Expr(:call, data.operators.comparison_operators[node.index])
        else
            @assert node.type == MOI.Nonlinear.NODE_VALUE
            expr.values[node.index]
        end
        if 1 <= node.parent <= length(tree)
            push!(tree[node.parent].args, node_expr)
        end
        push!(tree, node_expr)
    end
    return tree[1]
end

function MOI.Nonlinear.Evaluator(
    model::MOI.Nonlinear.Model,
    backend::AbstractSymbolicBackend,
    ordered_variables::Vector{MOI.VariableIndex},
)
    variable_order =
        Dict(x.value => i for (i, x) in enumerate(ordered_variables))
    subexpressions = Vector{Any}(undef, length(model.expressions))
    for (i, sub) in enumerate(model.expressions)
        subexpressions[i] = _to_expr(model, sub, variable_order, subexpressions)
    end
    objective = nothing
    if model.objective !== nothing
        obj = _to_expr(model, model.objective, variable_order, subexpressions)
        objective = _Function(model, obj)
    end
    functions = map(values(model.constraints)) do c
        expr = _to_expr(model, c.expression, variable_order, subexpressions)
        return _Function(model, expr)
    end
    return MOI.Nonlinear.Evaluator(
        model,
        _NonlinearOracle(backend, objective, functions),
    )
end
