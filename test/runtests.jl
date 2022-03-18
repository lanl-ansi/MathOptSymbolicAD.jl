module RunTests

using JuMP

import Ipopt
import PowerModels
import SymbolicAD
import Test

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            Test.@testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function run_unit_benchmark(model::JuMP.Model)
    @info "Constructing oracles"

    @time d = JuMP.NLPEvaluator(model)
    @time nlp_block = SymbolicAD.nlp_block_data(d)
    oracle = nlp_block.evaluator

    @info "MOI.initialize"

    @time MOI.initialize(d, [:Grad, :Jac, :Hess])
    @time MOI.initialize(oracle, [:Grad, :Jac, :Hess])

    # !!! note
    #     When trying to micro-benchmark this, be careful, because JuMP caches
    #     the last x and avoids redundant computation. Therefore calling
    #     MOI.eval_constraint multiple times with the same x will be optimistic
    #     compared with calling it with different x

    x = rand(JuMP.num_variables(model))

    if d.has_nlobj
        # MOI.eval_objective

        @info "MOI.eval_objective"

        @time MOI.eval_objective(d, x)
        @time MOI.eval_objective(d, x)
        @time MOI.eval_objective(oracle, x)
        @time MOI.eval_objective(oracle, x)

        Test.@test isapprox(
            MOI.eval_objective(d, x),
            MOI.eval_objective(oracle, x);
            atol = 1e-10,
        )

        # MOI.eval_objective_gradient

        @info "MOI.eval_objective_gradient"

        f = zeros(length(x))
        f1 = zeros(length(x))

        @time MOI.eval_objective_gradient(d, f1, x)
        @time MOI.eval_objective_gradient(d, f1, x)
        @time MOI.eval_objective_gradient(oracle, f, x)
        @time MOI.eval_objective_gradient(oracle, f, x)

        Test.@test isapprox(f, f1; atol = 1e-10)
    end

    # MOI.eval_constraint

    @info "MOI.eval_constraint"

    g = zeros(length(oracle.constraints))
    g1 = zeros(length(oracle.constraints))

    @time MOI.eval_constraint(d, g1, x)
    @time MOI.eval_constraint(d, g1, x)
    @time MOI.eval_constraint(oracle, g, x)
    @time MOI.eval_constraint(oracle, g, x)

    Test.@test isapprox(g, g1; atol = 1e-10)

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

    Test.@test isapprox(J_nz[p], J_nz_1[p1]; atol = 1e-10)

    # MOI.hessian_lagrangian_structure

    @info "MOI.hessian_lagrangian_structure"

    @time H1 = MOI.hessian_lagrangian_structure(d)
    @time H = MOI.hessian_lagrangian_structure(oracle)

    # MOI.eval_hessian_lagrangian

    @info "MOI.eval_hessian_lagrangian"

    μ, H_nz, H_nz_1 = rand(length(g)), zeros(length(H)), zeros(length(H1))

    @time MOI.eval_hessian_lagrangian(d, H_nz_1, x, 1.0, μ)
    @time MOI.eval_hessian_lagrangian(d, H_nz_1, x, 1.0, μ)
    @time MOI.eval_hessian_lagrangian(oracle, H_nz, x, 1.0, μ)
    @time MOI.eval_hessian_lagrangian(oracle, H_nz, x, 1.0, μ)

    # For the Hessian, we won't compute identical sparsity patterns due to
    # duplicates, etc. What we need to do is check the full sparse matrix.

    Test.@test isapprox(
        SymbolicAD._to_sparse(H, H_nz),
        SymbolicAD._to_sparse(H1, H_nz_1);
        atol = 1e-6,
    )
    return
end

function run_solution_benchmark(model::JuMP.Model)
    @info "Solving with JuMP"

    jump_model = Ipopt.Optimizer()
    MOI.copy_to(jump_model, model)
    MOI.optimize!(jump_model)
    jump_solution = MOI.get(
        jump_model,
        MOI.VariablePrimal(),
        MOI.get(jump_model, MOI.ListOfVariableIndices()),
    )
    jump_nlp_block = MOI.get(model, MOI.NLPBlock())

    @info "Solving with serial SymbolicAD"

    serial_model = Ipopt.Optimizer()
    MOI.copy_to(serial_model, model)
    serial_nlp_block =
        SymbolicAD.nlp_block_data(JuMP.NLPEvaluator(model); use_threads = false)
    MOI.set(serial_model, MOI.NLPBlock(), serial_nlp_block)
    MOI.optimize!(serial_model)
    serial_solution = MOI.get(
        serial_model,
        MOI.VariablePrimal(),
        MOI.get(serial_model, MOI.ListOfVariableIndices()),
    )

    @info "Solving with threaded SymbolicAD"

    threaded_model = Ipopt.Optimizer()
    MOI.copy_to(threaded_model, model)
    threaded_nlp_block =
        SymbolicAD.nlp_block_data(JuMP.NLPEvaluator(model); use_threads = true)
    MOI.set(threaded_model, MOI.NLPBlock(), threaded_nlp_block)
    MOI.optimize!(threaded_model)
    threaded_solution = MOI.get(
        threaded_model,
        MOI.VariablePrimal(),
        MOI.get(threaded_model, MOI.ListOfVariableIndices()),
    )

    @info "Validating solutions"

    Test.@test isapprox(jump_solution, serial_solution; atol = 1e-6)
    Test.@test isapprox(jump_solution, threaded_solution; atol = 1e-6)

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

"""
    run_clnlbeam_benchmark(; N::Int)

This is a good benchmark for the perils of Symbolic AD, because it has a
nonlinear objective function, and the nonlinear objective function has thousands
of terms.
"""
function run_clnlbeam_benchmark(; N::Int)
    h = 1 / N
    model = Model()
    @variable(model, -1 <= t[1:(N+1)] <= 1)
    @variable(model, -0.05 <= x[1:(N+1)] <= 0.05)
    @variable(model, u[1:(N+1)])
    @NLobjective(
        model,
        Min,
        sum(
            h / 2 * (u[i+1]^2 + u[i]^2) + 350 * h / 2 * (cos(t[i+1]) + cos(t[i])) for
            i = 1:N
        ),
    )
    for i = 1:N
        @NLconstraint(model, x[i+1] - x[i] == h / 2 * (sin(t[i+1]) + sin(t[i])))
        @constraint(model, t[i+1] - t[i] == h / 2 * (u[i+1] - u[i]))
    end
    run_unit_benchmark(model)
    run_solution_benchmark(model)
    return
end

function power_model(case::String)
    pm = PowerModels.instantiate_model(
        joinpath(@__DIR__, "data", case),
        PowerModels.ACPPowerModel,
        PowerModels.build_opf,
    )
    return pm.model
end

function test_case5_pjm_unit()
    model = power_model("pglib_opf_case5_pjm.m")
    run_unit_benchmark(model)
    return
end

function test_case5_pjm_solution()
    model = power_model("pglib_opf_case5_pjm.m")
    run_solution_benchmark(model)
    return
end

function test_clnlbeam()
    run_clnlbeam_benchmark(; N = 10)
    run_clnlbeam_benchmark(; N = 100)
    return
end

function test_optimizer_clnlbeam(; N::Int = 10)
    h = 1 / N
    model = Model(() -> SymbolicAD.Optimizer(Ipopt.Optimizer))
    @variable(model, -1 <= t[1:(N+1)] <= 1)
    @variable(model, -0.05 <= x[1:(N+1)] <= 0.05)
    @variable(model, u[1:(N+1)])
    @NLobjective(
        model,
        Min,
        sum(
            h / 2 * (u[i+1]^2 + u[i]^2) + 350 * h / 2 * (cos(t[i+1]) + cos(t[i])) for
            i = 1:N
        ),
    )
    for i = 1:N
        @NLconstraint(model, x[i+1] - x[i] == h / 2 * (sin(t[i+1]) + sin(t[i])))
        @constraint(model, t[i+1] - t[i] == h / 2 * (u[i+1] - u[i]))
    end
    optimize!(model)
    Test.@test isapprox(objective_value(model), 350; atol = 1e-6)
    t_sol = value.(t)
    u_sol = value.(u)
    Test.@test isapprox(
        sum(
            h / 2 * (u_sol[i+1]^2 + u_sol[i]^2) +
            350 * h / 2 * (cos(t_sol[i+1]) + cos(t_sol[i])) for i = 1:N
        ),
        350.0;
        atol = 1e-6,
    )
    return
end

end  # module

RunTests.runtests()
