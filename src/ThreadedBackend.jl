# Copyright (c) 2022, Oscar Dowson and contributors
# Copyright (c) 2022, Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

struct _ConstraintOffset
    index::Int
    ∇f_offset::Int
    ∇²f_offset::Int
end

struct ThreadedBackend <: AbstractSymbolicBackend
    offsets::Vector{Vector{_ConstraintOffset}}
    ThreadedBackend() = new(Vector{_ConstraintOffset}[])
end

function MOI.initialize(
    oracle::_NonlinearOracle,
    backend::ThreadedBackend,
    ::Vector{Symbol},
)
    obj_hess_offset = 0
    if oracle.objective !== nothing
        h = oracle.objective.expr.hash
        obj_hess_offset = length(oracle.symbolic_functions[h].H)
    end
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
        push!(backend.offsets, v)
    end
    return
end

function MOI.eval_constraint(
    oracle::_NonlinearOracle{ThreadedBackend},
    g::AbstractVector{Float64},
    x::AbstractVector{Float64},
)
    Threads.@threads for offsets in oracle.backend.offsets
        for c in offsets
            g[c.index] = _eval_function(oracle, oracle.constraints[c.index], x)
        end
    end
    return
end

function MOI.eval_constraint_jacobian(
    oracle::_NonlinearOracle{ThreadedBackend},
    J::AbstractVector{Float64},
    x::AbstractVector{Float64},
)
    Threads.@threads for offset in oracle.backend.offsets
        for c in offset
            g = _eval_gradient(oracle, oracle.constraints[c.index], x)
            for i in eachindex(g)
                J[c.∇f_offset+i-1] = g[i]
            end
        end
    end
    return
end

function _hessian_lagrangian_structure(
    structure::Vector{Tuple{Int,Int}},
    oracle::_NonlinearOracle{ThreadedBackend},
    ::Dict{Tuple{Int,Int},Int},
    c::_Function,
)
    f = oracle.symbolic_functions[c.expr.hash]
    for (i, j) in f.H_structure
        row, col = c.ordered_variables[i], c.ordered_variables[j]
        push!(structure, row >= col ? (row, col) : (col, row))
    end
    return
end

function MOI.eval_hessian_lagrangian(
    oracle::_NonlinearOracle{ThreadedBackend},
    H::AbstractVector{Float64},
    x::AbstractVector{Float64},
    σ::Float64,
    μ::AbstractVector{Float64},
)
    fill!(H, 0.0)
    if oracle.objective !== nothing && !iszero(σ)
        # It is important that we do not use any local variables that also
        # appear in the `Threads.@threads` block, otherwise Julia will create a
        # Core.Box. I also ran into some confusing debug issues, where we would
        # get out-of-bounds accesses.
        h_obj = _eval_hessian(oracle, oracle.objective, x)
        for i in eachindex(h_obj)
            H[i] = σ * h_obj[i]
        end
    end
    Threads.@threads for offset in oracle.backend.offsets
        for c in offset
            if iszero(μ[c.index])
                continue
            end
            h = _eval_hessian(oracle, oracle.constraints[c.index], x)
            for j in eachindex(h)
                H[c.∇²f_offset+j-1] = μ[c.index] * h[j]
            end
        end
    end
    return
end
