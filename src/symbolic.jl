# Copyright (c) 2022, Oscar Dowson
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

derivative(::Real, ::MOI.VariableIndex) = false

function derivative(f::MOI.VariableIndex, x::MOI.VariableIndex)
    return ifelse(f == x, true, false)
end

function derivative(
    f::MOI.ScalarAffineFunction{T},
    x::MOI.VariableIndex,
) where {T}
    ret = zero(T)
    for term in f.terms
        if term.variable == x
            ret += term.coefficient
        end
    end
    return ret
end

function derivative(
    f::MOI.ScalarQuadraticFunction{T},
    x::MOI.VariableIndex,
) where {T}
    constant = zero(T)
    for term in f.affine_terms
        if term.variable == x
            constant += term.coefficient
        end
    end
    aff_terms = MOI.ScalarAffineTerm{T}[]
    for q_term in f.quadratic_terms
        if q_term.variable_1 == q_term.variable_2 == x
            push!(aff_terms, MOI.ScalarAffineTerm(q_term.coefficient, x))
        elseif q_term.variable_1 == x
            push!(
                aff_terms,
                MOI.ScalarAffineTerm(q_term.coefficient, q_term.variable_2),
            )
        elseif q_term.variable_2 == x
            push!(
                aff_terms,
                MOI.ScalarAffineTerm(q_term.coefficient, q_term.variable_1),
            )
        end
    end
    return MOI.ScalarAffineFunction(aff_terms, constant)
end

function _replace_expression(node::Expr, u)
    for i in 1:length(node.args)
        node.args[i] = _replace_expression(node.args[i], u)
    end
    if Meta.isexpr(node, :call)
        op, args = node.args[1], node.args[2:end]
        return MOI.ScalarNonlinearFunction(op, args)
    end
    return u
end

_replace_expression(node::Any, u) = node

function _replace_expression(node::Symbol, u)
    if node == :x
        return u
    end
    return node
end

function derivative(f::MOI.ScalarNonlinearFunction, x::MOI.VariableIndex)
    if length(f.args) == 1
        u = only(f.args)
        if f.head == :+
            return derivative(u, x)
        elseif f.head == :-
            return MOI.ScalarNonlinearFunction(:-, Any[derivative(u, x)])
        elseif f.head == :abs
            scale = MOI.ScalarNonlinearFunction(
                :ifelse,
                Any[MOI.ScalarNonlinearFunction(:>=, Any[u, 0]), 1, -1],
            )
            return MOI.ScalarNonlinearFunction(:*, Any[scale, derivative(u, x)])
        elseif f.head == :sign
            return false
        end
        for (key, df, _) in MOI.Nonlinear.SYMBOLIC_UNIVARIATE_EXPRESSIONS
            if key == f.head
                # The chain rule: d(f(g(x))) / dx = f'(g(x)) * g'(x)
                u = only(f.args)
                df_du = _replace_expression(copy(df), u)
                du_dx = derivative(u, x)
                return MOI.ScalarNonlinearFunction(:*, Any[df_du, du_dx])
            end
        end
    end
    if f.head == :+
        # d/dx(+(args...)) = +(d/dx args)
        args = Any[derivative(arg, x) for arg in f.args]
        return MOI.ScalarNonlinearFunction(:+, args)
    elseif f.head == :-
        # d/dx(-(args...)) = -(d/dx args)
        # Note that - is not unary here because that wouuld be caught above.
        args = Any[derivative(arg, x) for arg in f.args]
        return MOI.ScalarNonlinearFunction(:-, args)
    elseif f.head == :*
        # Product rule: d/dx(*(args...)) = sum(d{i}/dx * args\{i})
        sum_terms = Any[]
        for i in 1:length(f.args)
            g = MOI.ScalarNonlinearFunction(:*, copy(f.args))
            g.args[i] = derivative(f.args[i], x)
            push!(sum_terms, g)
        end
        return MOI.ScalarNonlinearFunction(:+, sum_terms)
    elseif f.head == :^
        @assert length(f.args) == 2
        u, p = f.args
        du_dx = derivative(u, x)
        dp_dx = derivative(p, x)
        if _iszero(dp_dx)
            # p is constant and does not depend on x
            df_du = MOI.ScalarNonlinearFunction(
                :*,
                Any[p, MOI.ScalarNonlinearFunction(:^, Any[u, p-1])],
            )
            du_dx = derivative(u, x)
            return MOI.ScalarNonlinearFunction(:*, Any[df_du, du_dx])
        else
            # u(x)^p(x)
        end
    elseif f.head == :/
        # Quotient rule: d/dx(u / v) = (du/dx)*v - u*(dv/dx)) / v^2
        @assert length(f.args) == 2
        u, v = f.args
        du_dx, dv_dx = derivative(u, x), derivative(v, x)
        return MOI.ScalarNonlinearFunction(
            :/,
            Any[
                MOI.ScalarNonlinearFunction(
                    :-,
                    Any[
                        MOI.ScalarNonlinearFunction(:*, Any[du_dx, v]),
                        MOI.ScalarNonlinearFunction(:*, Any[u, dv_dx]),
                    ],
                ),
                MOI.ScalarNonlinearFunction(:^, Any[v, 2]),
            ],
        )
    elseif f.head == :ifelse
        @assert length(f.args) == 3
        # Pick the derivative of the active branch
        return MOI.ScalarNonlinearFunction(
            :ifelse,
            Any[f.args[1], derivative(f.args[2], x), derivative(f.args[3], x)],
        )
    elseif f.head == :atan
        # TODO
    elseif f.head == :min
        g = derivative(f.args[end], x)
        for i in length(f.args)-1:-1:1
            g = MOI.ScalarNonlinearFunction(
                :ifelse,
                Any[
                    MOI.ScalarNonlinearFunction(:(<=), Any[f.args[i], f]),
                    derivative(f.args[i], x),
                    g,
                ],
            )
        end
        return g
    elseif f.head == :max
        g = derivative(f.args[end], x)
        for i in length(f.args)-1:-1:1
            g = MOI.ScalarNonlinearFunction(
                :ifelse,
                Any[
                    MOI.ScalarNonlinearFunction(:(>=), Any[f.args[i], f]),
                    derivative(f.args[i], x),
                    g,
                ],
            )
        end
        return g
    elseif f.head in (:(>=), :(<=), :(<), :(>), :(==))
        return false
    end
    err = MOI.UnsupportedNonlinearOperator(
        f.head,
        "the operator does not support symbolic differentiation",
    )
    return throw(err)
end

"""
    simplify(f)

Return a simplified version of the function `f`.

!!! warning
    This function is not type stable by design.
"""
simplify(f) = f

function simplify(f::MOI.ScalarAffineFunction{T}) where {T}
    f = MOI.Utilities.canonical(f)
    if isempty(f.terms)
        return f.constant
    end
    return f
end

function simplify(f::MOI.ScalarQuadraticFunction{T}) where {T}
    f = MOI.Utilities.canonical(f)
    if isempty(f.quadratic_terms)
        if isempty(f.affine_terms)
            return f.constant
        end
        return MOI.ScalarAffineFunction(f.affine_terms, f.constant)
    end
    return f
end

# function simplify(f::MOI.ScalarNonlinearFunction)
#     for i in 1:length(f.args)
#         f.args[i] = simplify(f.args[i])
#     end
#     return _eval_if_constant(simplify(Val(f.head), f))
# end
function simplify(f::MOI.ScalarNonlinearFunction)
    stack, result_stack = Any[f], Any[]
    while !isempty(stack)
        arg = pop!(stack)
        if arg isa MOI.ScalarNonlinearFunction
            # We need some sort of hint so that the next time we see this on the
            # stack we evaluate it using the args in `result_stack`. One option
            # would be a custom type. Or we can just wrap in (,) and then check
            # for a Tuple, which isn't (curretly) a valid argument.
            push!(stack, (arg,))
            for child in arg.args
                push!(stack, child)
            end
        elseif arg isa Tuple{<:MOI.ScalarNonlinearFunction}
            f_expr = only(arg)
            args = Any[pop!(result_stack) for _ in 1:length(f_expr.args)]
            result = MOI.ScalarNonlinearFunction(f_expr.head, args)
            # simplify(::Val, ::Any) does not use recursion so this is safe.
            result = simplify(Val(result.head), result)
            result = _eval_if_constant(result)
            push!(result_stack, result)
        else
            push!(result_stack, arg)
        end
    end
    return only(result_stack)
end

function simplify(f::MOI.VectorAffineFunction{T}) where {T}
    f = MOI.Utilities.canonical(f)
    if isempty(f.terms)
        return f.constants
    end
    return f
end

function simplify(f::MOI.VectorQuadraticFunction{T}) where {T}
    f = MOI.Utilities.canonical(f)
    if isempty(f.quadratic_terms)
        if isempty(f.affine_terms)
            return f.constants
        end
        return MOI.VectorAffineFunction(f.affine_terms, f.constants)
    end
    return f
end

function simplify(f::MOI.VectorNonlinearFunction)
    return MOI.VectorNonlinearFunction(simplify.(f.rows))
end

# If a ScalarNonlinearFunction has only constant arguments, we should return
# the vaålue.

_isnum(::Any) = false

_isnum(::Union{Bool,Integer,Float64}) = true

function _eval_if_constant(f::MOI.ScalarNonlinearFunction)
    if all(_isnum, f.args) && hasproperty(Base, f.head)
        return getproperty(Base, f.head)(f.args...)
    end
    return f
end

_eval_if_constant(f) = f

_iszero(x::Any) = _isnum(x) && iszero(x)

_isone(x::Any) = _isnum(x) && isone(x)

"""
    _isexpr(f::Any, head::Symbol[, n::Int])

Return `true` if `f` is a `ScalarNonlinearFunction` with head `head` and, if
specified, `n` arguments.
"""
_isexpr(::Any, ::Symbol, n::Int = 0) = false

_isexpr(f::MOI.ScalarNonlinearFunction, head::Symbol) = f.head == head

function _isexpr(f::MOI.ScalarNonlinearFunction, head::Symbol, n::Int)
    return _isexpr(f, head) && length(f.args) == n
end

"""
    simplify(::Val{head}, f::MOI.ScalarNonlinearFunction)

Return a simplified version of `f` where the head of `f` is `head`.

Implementing this method enables custom simplification rules for different
operators without needing a giant switch statement.

It is important that this function does not recursively call `simplify`. Deal
only with the immediate operator. The children arguments will already be
simplified.
"""
simplify(::Val, f::MOI.ScalarNonlinearFunction) = f

function simplify(::Val{:*}, f::MOI.ScalarNonlinearFunction)
    new_args = Any[]
    first_constant = 0
    for arg in f.args
        if _isexpr(arg, :*)
            # If the child is a :*, lift its arguments to the parent
            append!(new_args, arg.args)
        elseif _iszero(arg)
            # If any argument is zero, the entire expression must be false
            return false
        elseif _isone(arg)
            # Skip any arguments that are one
        elseif arg isa Real
            # Collect all constant arguments into a single value
            if first_constant == 0
                push!(new_args, arg)
                first_constant = length(new_args)
            else
                new_args[first_constant] *= arg
            end
        else
            push!(new_args, arg)
        end
    end
    if isempty(new_args)
        return true
    elseif length(new_args) == 1
        return only(new_args)
    end
    return MOI.ScalarNonlinearFunction(:*, new_args)
end

function simplify(::Val{:+}, f::MOI.ScalarNonlinearFunction)
    if length(f.args) == 1
        # +(x) -> x
        return only(f.args)
    elseif length(f.args) == 2 && _isexpr(f.args[2], :-, 1)
        # +(x, -y) -> -(x, y)
        return MOI.ScalarNonlinearFunction(
            :-,
            Any[f.args[1], f.args[2].args[1]],
        )
    end
    new_args = Any[]
    first_constant = 0
    for arg in f.args
        if _isexpr(arg, :+)
            # If a child is a :+, lift its arguments to the parent
            append!(new_args, arg.args)
        elseif _iszero(arg)
            # Skip any zero arguments
        elseif arg isa Real
            # Collect all constant arguments into a single value
            if first_constant == 0
                push!(new_args, arg)
                first_constant = length(new_args)
            else
                new_args[first_constant] += arg
            end
        else
            push!(new_args, arg)
        end
    end
    if isempty(new_args)
        # +() -> false
        return false
    elseif length(new_args) == 1
        # +(x) -> x
        return only(new_args)
    end
    return MOI.ScalarNonlinearFunction(:+, new_args)
end

function simplify(::Val{:-}, f::MOI.ScalarNonlinearFunction)
    if length(f.args) == 1
        if _isexpr(f.args[1], :-, 1)
            # -(-(x)) => x
            return f.args[1].args[1]
        end
    elseif length(f.args) == 2
        if _iszero(f.args[1])
            # 0 - x => -x
            return MOI.ScalarNonlinearFunction(:-, Any[f.args[2]])
        elseif _iszero(f.args[2])
            # x - 0 => x
            return f.args[1]
        elseif f.args[1] == f.args[2]
            # x - x => 0
            return false
        elseif _isexpr(f.args[2], :-, 1)
            # x - -(y) => x + y
            return MOI.ScalarNonlinearFunction(
                :+,
                Any[f.args[1], f.args[2].args[1]],
            )
        end
    end
    return f
end

function simplify(::Val{:^}, f::MOI.ScalarNonlinearFunction)
    if _iszero(f.args[2])
        # x^0 => 1
        return true
    elseif _isone(f.args[2])
        # x^1 => x
        return f.args[1]
    elseif _iszero(f.args[1])
        # 0^x => 0
        return false
    elseif _isone(f.args[1])
        # 1^x => 1
        return true
    end
    return f
end

function variables(f::MOI.AbstractScalarFunction)
    ret = MOI.VariableIndex[]
    variables!(ret, f)
    return ret
end

variables(::Real) = MOI.VariableIndex[]
variables!(ret, ::Real) = nothing

function variables!(ret, f::MOI.VariableIndex)
    if !(f in ret)
        push!(ret, f)
    end
    return
end

function variables!(ret, f::MOI.ScalarAffineTerm)
    if !(f.variable in ret)
        push!(ret, f.variable)
    end
    return
end

function variables!(ret, f::MOI.ScalarAffineFunction)
    for term in f.terms
        variables!(ret, term)
    end
    return
end

function variables!(ret, f::MOI.ScalarQuadraticTerm)
    if !(f.variable_1 in ret)
        push!(ret, f.variable_1)
    end
    if !(f.variable_2 in ret)
        push!(ret, f.variable_2)
    end
    return
end

function variables!(ret, f::MOI.ScalarQuadraticFunction)
    for term in f.affine_terms
        variables!(ret, term)
    end
    for q_term in f.quadratic_terms
        variables!(ret, q_term)
    end
    return
end

function variables!(ret, f::MOI.ScalarNonlinearFunction)
    for arg in f.args
        variables!(ret, arg)
    end
    return
end

gradient(::Real) = Dict{MOI.VariableIndex,Any}()
function gradient(f::MOI.AbstractScalarFunction)
    return Dict{MOI.VariableIndex,Any}(
        x => simplify(derivative(f, x)) for x in variables(f)
    )
end

struct Node
end

struct Tape
    nodes::Vector{Node}
    output::Vector{Float64}
end

function evaluate(f::Vector{Any}, x::Dict{MOI.VariableIndex,Float64})
    return
end
