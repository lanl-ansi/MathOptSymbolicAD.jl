struct Optimizer{O} <: MOI.AbstractOptimizer
    backend::AbstractNonlinearOracleBackend
    inner::O
    function Optimizer(
        inner;
        backend::AbstractNonlinearOracleBackend = DefaultBackend(),
    )
        model = MOI.instantiate(inner)
        return new{typeof(model)}(backend, model)
    end
end

function MOI.set(model::Optimizer, ::MOI.NLPBlock, value)
    @info("Replacing NLPBlock with the SymbolicAD one")
    MOI.set(
        model.inner,
        MOI.NLPBlock(),
        _nlp_block_data(value.evaluator; backend = model.backend),
    )
    return
end

MOI.is_empty(model::Optimizer) = MOI.is_empty(model.inner)

MOI.empty!(model::Optimizer) = MOI.empty!(model.inner)

function MOI.supports_constraint(
    model::Optimizer,
    f::Type{<:MOI.AbstractFunction},
    s::Type{<:MOI.AbstractSet},
)
    return MOI.supports_constraint(model.inner, f, s)
end

function MOI.supports(model::Optimizer, attr::MOI.AnyAttribute, args...)
    return MOI.supports(model.inner, attr, args...)
end

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike)
    index_map = MOI.copy_to(model.inner, src)
    if MOI.NLPBlock() in MOI.get(src, MOI.ListOfModelAttributesSet())
        block = MOI.get(src, MOI.NLPBlock())
        MOI.set(model, MOI.NLPBlock(), block)
    end
    return index_map
end

function MOI.set(model::Optimizer, attr::MOI.AnyAttribute, args...)
    return MOI.set(model.inner, attr, args...)
end

function MOI.get(model::Optimizer, attr::MOI.AnyAttribute, args...)
    return MOI.get(model.inner, attr, args...)
end

MOI.optimize!(model::Optimizer) = MOI.optimize!(model.inner)
