"""
    run_unit_benchmark(
        model::JuMP.Model;
        direct_model::Bool,
        atol::Float64 = 1e-10,
    )

Given a JuMP Model `model`, run a series of unit tests. This can be helpful to
test differences between JuMP and SymbolicAD (using an atol of `atol`), and also
to check relative performance.

If `direct_model`, build the SymbolicAD backend directly from JuMP's NLP
datastructures. If `direct_model == false`, build from `MOI.objective_expr` and
`MOI.constraint_expr`.
"""
function run_unit_benchmark(
    model::JuMP.Model;
    direct_model::Bool,
    atol::Float64 = 1e-10,
    rtol::Float64 = 0.0,
)
    @info "Constructing oracles"

    @time d = JuMP.NLPEvaluator(model)
    @time nlp_block = if direct_model
        _nlp_block_data(model; backend = DefaultBackend())
    else
        _nlp_block_data(d; backend = DefaultBackend())
    end
    oracle = nlp_block.evaluator

    @info "MOI.initialize"

    @time MOI.initialize(d, [:Grad, :Jac, :Hess])
    @time MOI.initialize(oracle, [:Grad, :Jac, :Hess])

    # !!! note
    #     When trying to micro-benchmark this, be careful, because JuMP caches
    #     the last x and avoids redundant computation. Therefore calling
    #     MOI.eval_constraint multiple times with the same x will be optimistic
    #     compared with calling it with different x

    x = map(JuMP.all_variables(model)) do xi
        lb = JuMP.has_lower_bound(xi) ? JuMP.lower_bound(xi) : -1e6
        ub = JuMP.has_upper_bound(xi) ? JuMP.upper_bound(xi) : 1e6
        return lb + (ub - lb) * rand()
    end

    if d.has_nlobj
        # MOI.eval_objective

        @info "MOI.eval_objective"

        @time MOI.eval_objective(d, x)
        @time MOI.eval_objective(d, x)
        @time MOI.eval_objective(oracle, x)
        @time MOI.eval_objective(oracle, x)

        d_eval_objective = MOI.eval_objective(d, x)
        oracle_eval_objective = MOI.eval_objective(oracle, x)
        Test.@test isapprox(
            d_eval_objective,
            oracle_eval_objective;
            atol = atol,
            rtol = rtol,
        )
        println("Error = ", d_eval_objective - oracle_eval_objective)

        # MOI.eval_objective_gradient

        @info "MOI.eval_objective_gradient"

        f = zeros(length(x))
        f1 = zeros(length(x))

        @time MOI.eval_objective_gradient(d, f1, x)
        @time MOI.eval_objective_gradient(d, f1, x)
        @time MOI.eval_objective_gradient(oracle, f, x)
        @time MOI.eval_objective_gradient(oracle, f, x)

        Test.@test isapprox(f, f1; atol = atol, rtol = rtol)
        println("Error = ", maximum(abs.(f .- f1)))
    end

    # MOI.eval_constraint

    @info "MOI.eval_constraint"

    g = zeros(length(oracle.constraints))
    g1 = zeros(length(oracle.constraints))

    @time MOI.eval_constraint(d, g1, x)
    @time MOI.eval_constraint(d, g1, x)
    @time MOI.eval_constraint(oracle, g, x)
    @time MOI.eval_constraint(oracle, g, x)

    Test.@test isapprox(g, g1; atol = atol, rtol = rtol)
    println("Error = ", maximum(abs.(g .- g1)))

    # MOI.jacobian_structure

    @info "MOI.jacobian_structure"

    @time J1 = MOI.jacobian_structure(d)
    @time J = MOI.jacobian_structure(oracle)

    p, p1 = sortperm(J), sortperm(J1)
    Test.@test J[p] == J1[p1]

    # MOI.eval_constraint_jacobian

    @info "MOI.eval_constraint_jacobian"

    J_nz, J_nz_1 = zeros(length(J)), zeros(length(J1))

    @time MOI.eval_constraint_jacobian(d, J_nz_1, x)
    @time MOI.eval_constraint_jacobian(d, J_nz_1, x)
    @time MOI.eval_constraint_jacobian(oracle, J_nz, x)
    @time MOI.eval_constraint_jacobian(oracle, J_nz, x)

    Test.@test isapprox(J_nz[p], J_nz_1[p1]; atol = atol, rtol = rtol)
    println("Error = ", LinearAlgebra.norm(J_nz[p] .- J_nz_1[p1]))

    # MOI.hessian_lagrangian_structure

    @info "MOI.hessian_lagrangian_structure"

    @time H1 = MOI.hessian_lagrangian_structure(d)
    @time H = MOI.hessian_lagrangian_structure(oracle)

    # MOI.eval_hessian_lagrangian

    @info "MOI.eval_hessian_lagrangian"

    μ, H_nz, H_nz_1 = rand(length(g)), zeros(length(H)), zeros(length(H1))
    σ = rand()
    @time MOI.eval_hessian_lagrangian(d, H_nz_1, x, σ, μ)
    @time MOI.eval_hessian_lagrangian(d, H_nz_1, x, σ, μ)
    @time MOI.eval_hessian_lagrangian(oracle, H_nz, x, σ, μ)
    @time MOI.eval_hessian_lagrangian(oracle, H_nz, x, σ, μ)

    # For the Hessian, we won't compute identical sparsity patterns due to
    # duplicates, etc. What we need to do is check the full sparse matrix.

    function _to_sparse(IJ, V)
        I, J = [i for (i, _) in IJ], [j for (_, j) in IJ]
        n = max(maximum(I), maximum(J))
        return SparseArrays.sparse(I, J, V, n, n)
    end

    d_sparse = _to_sparse(H, H_nz)
    oracle_sparse = _to_sparse(H1, H_nz_1)
    Test.@test isapprox(d_sparse, oracle_sparse; atol = atol, rtol = rtol)
    println("Error = ", LinearAlgebra.norm(d_sparse - oracle_sparse))
    return
end

"""
    run_solution_benchmark(model::JuMP.Model, optimizer; atol = 1e-6)

Given a JuMP Model `model`, solve the problem using JuMP's built-in AD, as well
as the `DefaultBackend` and `ThreadedBackend` of SymbolicAD and check that they
find the same solutions.

`optimizer` must be something that can be passed to `MOI.instantiate`.
"""
function run_solution_benchmark(
    model::JuMP.Model,
    optimizer;
    atol::Float64 = 1e-6,
    rtol::Float64 = 0.0,
)
    @info "Solving with JuMP"

    jump_model = MOI.instantiate(optimizer)
    MOI.copy_to(jump_model, model)
    MOI.optimize!(jump_model)
    jump_solution = MOI.get(
        jump_model,
        MOI.VariablePrimal(),
        MOI.get(jump_model, MOI.ListOfVariableIndices()),
    )
    jump_nlp_block = MOI.get(model, MOI.NLPBlock())

    @info "Solving with serial SymbolicAD"

    serial_model = MOI.instantiate(optimizer)
    MOI.copy_to(serial_model, model)
    serial_nlp_block =
        _nlp_block_data(JuMP.NLPEvaluator(model); backend = DefaultBackend())
    MOI.set(serial_model, MOI.NLPBlock(), serial_nlp_block)
    MOI.optimize!(serial_model)
    serial_solution = MOI.get(
        serial_model,
        MOI.VariablePrimal(),
        MOI.get(serial_model, MOI.ListOfVariableIndices()),
    )

    @info "Solving with threaded SymbolicAD"

    threaded_model = MOI.instantiate(optimizer)
    MOI.copy_to(threaded_model, model)
    threaded_nlp_block =
        _nlp_block_data(JuMP.NLPEvaluator(model); backend = ThreadedBackend())
    MOI.set(threaded_model, MOI.NLPBlock(), threaded_nlp_block)
    MOI.optimize!(threaded_model)
    threaded_solution = MOI.get(
        threaded_model,
        MOI.VariablePrimal(),
        MOI.get(threaded_model, MOI.ListOfVariableIndices()),
    )

    @info "Validating solutions"
    println("Errors = ", extrema(jump_solution .- serial_solution))
    Test.@test ≈(jump_solution, serial_solution; atol = atol, rtol = rtol)
    Test.@test ≈(jump_solution, threaded_solution; atol = atol, rtol = rtol)

    @info "Timing statistics"

    println("JuMP timing statistics")
    println(" - ", jump_nlp_block.evaluator.eval_constraint_timer)
    println(" - ", jump_nlp_block.evaluator.eval_constraint_jacobian_timer)
    println(" - ", jump_nlp_block.evaluator.eval_hessian_lagrangian_timer)

    println("Serial SymbolicAD timing statistics")
    println(" - ", serial_nlp_block.evaluator.eval_constraint_timer)
    println(" - ", serial_nlp_block.evaluator.eval_constraint_jacobian_timer)
    println(" - ", serial_nlp_block.evaluator.eval_hessian_lagrangian_timer)

    println("Threaded SymbolicAD timing statistics")
    println(" - ", threaded_nlp_block.evaluator.eval_constraint_timer)
    println(" - ", threaded_nlp_block.evaluator.eval_constraint_jacobian_timer)
    println(" - ", threaded_nlp_block.evaluator.eval_hessian_lagrangian_timer)
    return
end
