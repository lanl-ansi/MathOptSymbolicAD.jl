# Copyright (c) 2022, Oscar Dowson and contributors
# Copyright (c) 2022, Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module RunTests

using JuMP

import Ipopt
import LinearAlgebra
import MathOptSymbolicAD
import PowerModels
import Random
import SparseArrays
import SpecialFunctions
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

"""
    run_unit_benchmark(model::JuMP.Model; atol::Float64 = 1e-10)

Given a JuMP Model `model`, run a series of unit tests. This can be helpful to
test differences between JuMP and MathOptSymbolicAD (using an atol of `atol`),
and also to check relative performance.
"""
function run_unit_benchmark(
    model::JuMP.Model;
    atol::Float64 = 1e-10,
    rtol::Float64 = 0.0,
)
    seed = 1234
    sym = _run_unit_benchmark(model, MathOptSymbolicAD.DefaultBackend(), seed)
    moi = _run_unit_benchmark(model, MOI.Nonlinear.SparseReverseMode(), seed)
    # f
    Test.@test isapprox(sym.f, moi.f; atol = atol, rtol = rtol)
    println("Error = ", sym.f - moi.f)
    # ∇f
    Test.@test isapprox(sym.∇f, moi.∇f; atol = atol, rtol = rtol)
    println("Error = ", maximum(abs.(sym.∇f .- moi.∇f)))
    # g
    Test.@test isapprox(sym.g, moi.g; atol = atol, rtol = rtol)
    println("Error = ", maximum(abs.(sym.g .- moi.g)))
    # J
    sym_p, moi_p = sortperm(sym.J), sortperm(moi.J)
    Test.@test sym.J[sym_p] == moi.J[moi_p]
    # J_nz
    sym_J, moi_J = sym.J_nz[sym_p], moi.J_nz[moi_p]
    Test.@test isapprox(sym_J, moi_J; atol = atol, rtol = rtol)
    println("Error = ", LinearAlgebra.norm(sym_J .- moi_J))
    # H, H_nz
    # For the Hessian, we won't compute identical sparsity patterns due to
    # duplicates, etc. What we need to do is check the full sparse matrix.
    function _to_sparse(IJ, V)
        I, J = [i for (i, _) in IJ], [j for (_, j) in IJ]
        n = max(maximum(I), maximum(J))
        return SparseArrays.sparse(I, J, V, n, n)
    end
    sym_H = _to_sparse(sym.H, sym.H_nz)
    moi_H = _to_sparse(moi.H, moi.H_nz)
    Test.@test isapprox(sym_H, moi_H; atol = atol, rtol = rtol)
    println("Error = ", LinearAlgebra.norm(sym_H - moi_H))
    return
end

function _run_unit_benchmark(model, backend, seed)
    @info "Constructing oracles"
    @time d = JuMP.NLPEvaluator(model; _differentiation_backend = backend)
    @info "MOI.initialize"
    @time MOI.initialize(d, [:Grad, :Jac, :Hess])
    Random.seed!(seed)
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
    f, ∇f = 0.0, zeros(length(x))
    if d.model.objective !== nothing
        # MOI.eval_objective
        @info "MOI.eval_objective"
        @time MOI.eval_objective(d, x)
        @time MOI.eval_objective(d, x)
        f = MOI.eval_objective(d, x)
        # MOI.eval_objective_gradient
        @info "MOI.eval_objective_gradient"
        @time MOI.eval_objective_gradient(d, ∇f, x)
        @time MOI.eval_objective_gradient(d, ∇f, x)
    end
    # MOI.eval_constraint
    @info "MOI.eval_constraint"
    g = zeros(length(d.model.constraints))
    @time MOI.eval_constraint(d, g, x)
    @time MOI.eval_constraint(d, g, x)
    # MOI.jacobian_structure
    @info "MOI.jacobian_structure"
    @time J = MOI.jacobian_structure(d)
    # MOI.eval_constraint_jacobian
    @info "MOI.eval_constraint_jacobian"
    J_nz = zeros(length(J))
    @time MOI.eval_constraint_jacobian(d, J_nz, x)
    @time MOI.eval_constraint_jacobian(d, J_nz, x)
    # MOI.hessian_lagrangian_structure
    @info "MOI.hessian_lagrangian_structure"
    @time H = MOI.hessian_lagrangian_structure(d)
    # MOI.eval_hessian_lagrangian
    @info "MOI.eval_hessian_lagrangian"
    μ, H_nz = rand(length(g)), zeros(length(H))
    σ = rand()
    @time MOI.eval_hessian_lagrangian(d, H_nz, x, σ, μ)
    @time MOI.eval_hessian_lagrangian(d, H_nz, x, σ, μ)
    return (f = f, ∇f = ∇f, g = g, J = J, J_nz = J_nz, H = H, H_nz = H_nz)
end

"""
    run_solution_benchmark(model::JuMP.Model, optimizer; atol = 1e-6)

Given a JuMP Model `model`, solve the problem using JuMP's built-in AD, as well
as the `DefaultBackend` and `ThreadedBackend` of MathOptSymbolicAD and check
that they find the same solutions.

`optimizer` must be something that can be passed to `MOI.instantiate`.
"""
function run_solution_benchmark(
    model::JuMP.Model,
    optimizer;
    atol::Float64 = 1e-6,
    rtol::Float64 = 0.0,
)
    @info "Solving with SparseReverseMode"
    moi_solution = _run_solution_benchmark(
        model,
        optimizer,
        MOI.Nonlinear.SparseReverseMode(),
    )
    @info "Solving with serial MathOptSymbolicAD"
    serial_solution = _run_solution_benchmark(
        model,
        optimizer,
        MathOptSymbolicAD.DefaultBackend(),
    )
    @info "Solving with threaded MathOptSymbolicAD"
    threaded_solution = _run_solution_benchmark(
        model,
        optimizer,
        MathOptSymbolicAD.ThreadedBackend(),
    )
    @info "Validating solutions"
    println("Errors = ", extrema(moi_solution .- serial_solution))
    Test.@test ≈(moi_solution, serial_solution; atol = atol, rtol = rtol)
    Test.@test ≈(moi_solution, threaded_solution; atol = atol, rtol = rtol)
    return
end

function _run_solution_benchmark(model, optimizer, backend)
    set_optimizer(model, optimizer)
    set_attribute(model, MOI.AutomaticDifferentiationBackend(), backend)
    optimize!(model)
    println("Timing statistics")
    nlp_block = MOI.get(unsafe_backend(model), MOI.NLPBlock())
    println(" - ", nlp_block.evaluator.eval_constraint_timer)
    println(" - ", nlp_block.evaluator.eval_constraint_jacobian_timer)
    println(" - ", nlp_block.evaluator.eval_hessian_lagrangian_timer)
    return value.(all_variables(model))
end

"""
    _run_clnlbeam_benchmark(; N::Int)

This is a good benchmark for the perils of Symbolic AD, because it has a
nonlinear objective function, and the nonlinear objective function has thousands
of terms.
"""
function _run_clnlbeam_benchmark(; N::Int)
    h = 1 / N
    model = Model()
    @variable(model, -1 <= t[1:(N+1)] <= 1)
    @variable(model, -0.05 <= x[1:(N+1)] <= 0.05)
    @variable(model, u[1:(N+1)])
    @NLobjective(
        model,
        Min,
        sum(
            h / 2 * (u[i+1]^2 + u[i]^2) +
            350 * h / 2 * (cos(t[i+1]) + cos(t[i])) for i in 1:N
        ),
    )
    for i in 1:N
        @NLconstraint(model, x[i+1] - x[i] == h / 2 * (sin(t[i+1]) + sin(t[i])))
        @constraint(model, t[i+1] - t[i] == h / 2 * (u[i+1] - u[i]))
    end
    # Different rtol because the points we pick can be quite bad
    run_unit_benchmark(model; rtol = 1e-6)
    run_solution_benchmark(model, Ipopt.Optimizer)
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

function test_case5_pjm_solution()
    model = power_model("pglib_opf_case5_pjm.m")
    run_solution_benchmark(model, Ipopt.Optimizer)
    return
end

function test_clnlbeam()
    _run_clnlbeam_benchmark(; N = 10)
    _run_clnlbeam_benchmark(; N = 100)
    return
end

function test_optimizer_clnlbeam(; N::Int = 10)
    h = 1 / N
    model = Model(Ipopt.Optimizer)
    @variable(model, -1 <= t[1:(N+1)] <= 1)
    @variable(model, -0.05 <= x[1:(N+1)] <= 0.05)
    @variable(model, u[1:(N+1)])
    @NLobjective(
        model,
        Min,
        sum(
            h / 2 * (u[i+1]^2 + u[i]^2) +
            350 * h / 2 * (cos(t[i+1]) + cos(t[i])) for i in 1:N
        ),
    )
    for i in 1:N
        @NLconstraint(model, x[i+1] - x[i] == h / 2 * (sin(t[i+1]) + sin(t[i])))
        @constraint(model, t[i+1] - t[i] == h / 2 * (u[i+1] - u[i]))
    end
    optimize!(
        model;
        _differentiation_backend = MathOptSymbolicAD.DefaultBackend(),
    )
    Test.@test isapprox(objective_value(model), 350; atol = 1e-6)
    t_sol = value.(t)
    u_sol = value.(u)
    Test.@test isapprox(
        sum(
            h / 2 * (u_sol[i+1]^2 + u_sol[i]^2) +
            350 * h / 2 * (cos(t_sol[i+1]) + cos(t_sol[i])) for i in 1:N
        ),
        350.0;
        atol = 1e-6,
    )
    return
end

function test_optimizer_clnlbeam_expr(; N::Int = 10)
    h = 1 / N
    model = Model(Ipopt.Optimizer)
    @variable(model, -1 <= t[1:(N+1)] <= 1)
    @variable(model, -0.05 <= x[1:(N+1)] <= 0.05)
    @variable(model, u[1:(N+1)])
    @objective(
        model,
        Min,
        sum(
            h / 2 * (u[i+1]^2 + u[i]^2) +
            350 * h / 2 * (cos(t[i+1]) + cos(t[i])) for i in 1:N
        ),
    )
    for i in 1:N
        @constraint(model, x[i+1] - x[i] == h / 2 * (sin(t[i+1]) + sin(t[i])))
        @constraint(model, t[i+1] - t[i] == h / 2 * (u[i+1] - u[i]))
    end
    set_attribute(
        model,
        MOI.AutomaticDifferentiationBackend(),
        MathOptSymbolicAD.DefaultBackend(),
    )
    optimize!(model)
    Test.@test isapprox(objective_value(model), 350; atol = 1e-6)
    t_sol = value.(t)
    u_sol = value.(u)
    Test.@test isapprox(
        sum(
            h / 2 * (u_sol[i+1]^2 + u_sol[i]^2) +
            350 * h / 2 * (cos(t_sol[i+1]) + cos(t_sol[i])) for i in 1:N
        ),
        350.0;
        atol = 1e-6,
    )
    return
end

function test_optimizer_case5_pjm()
    model = power_model("pglib_opf_case5_pjm.m")
    set_optimizer(model, Ipopt.Optimizer)
    set_attribute(
        model,
        MOI.AutomaticDifferentiationBackend(),
        MathOptSymbolicAD.DefaultBackend(),
    )
    optimize!(model)
    symbolic_obj = objective_value(model)
    optimize!(model)
    Test.@test isapprox(objective_value(model), symbolic_obj, atol = 1e-6)
    return
end

function test_user_defined_functions()
    model = Model(Ipopt.Optimizer)
    @variable(model, 0.5 <= x <= 1.0)
    register(model, :mysin, 1, a -> sin(a); autodiff = true)
    register(model, :pow, 2, (a, b) -> a^b; autodiff = true)
    @NLobjective(model, Max, mysin(x) + log(x) + dawson(x) - pow(x, 2))
    optimize!(
        model;
        _differentiation_backend = MathOptSymbolicAD.DefaultBackend(),
    )
    Test.@test termination_status(model) == LOCALLY_SOLVED
    return
end

function test_user_defined_functions_expr()
    model = Model(Ipopt.Optimizer)
    @variable(model, 0.5 <= x <= 1.0)
    @operator(model, mysin, 1, a -> sin(a))
    @operator(model, pow, 2, (a, b) -> a^b)
    @objective(
        model,
        Max,
        mysin(x) + log(x) + SpecialFunctions.dawson(x) - pow(x, 2),
    )
    set_attribute(
        model,
        MOI.AutomaticDifferentiationBackend(),
        MathOptSymbolicAD.DefaultBackend(),
    )
    optimize!(model)
    Test.@test termination_status(model) == LOCALLY_SOLVED
    return
end

function test_nested_subexpressions()
    model = Model(Ipopt.Optimizer)
    @variable(model, 0.5 <= x <= 1.0)
    @NLexpression(model, my_expr1, x - 1)
    @NLexpression(model, my_expr2, my_expr1^2)
    @NLobjective(model, Min, my_expr2)
    optimize!(
        model;
        _differentiation_backend = MathOptSymbolicAD.DefaultBackend(),
    )
    Test.@test termination_status(model) == LOCALLY_SOLVED
    Test.@test ≈(value(x), 1.0; atol = 1e-3)
    return
end

function test_nested_subexpressions_expr()
    model = Model(Ipopt.Optimizer)
    @variable(model, 0.5 <= x <= 1.0)
    @expression(model, my_expr1, x - 1)
    @expression(model, my_expr2, my_expr1^2)
    @objective(model, Min, my_expr2)
    set_attribute(
        model,
        MOI.AutomaticDifferentiationBackend(),
        MathOptSymbolicAD.DefaultBackend(),
    )
    optimize!(model)
    Test.@test termination_status(model) == LOCALLY_SOLVED
    Test.@test ≈(value(x), 1.0; atol = 1e-3)
    return
end

function test_constant_subexpressions()
    model = Model(Ipopt.Optimizer)
    @variable(model, 0.5 <= x <= 1.0)
    @NLexpression(model, my_expr1, 1.0)
    @NLexpression(model, my_expr2, x)
    @NLobjective(model, Min, (my_expr2 - my_expr1)^2)
    optimize!(
        model;
        _differentiation_backend = MathOptSymbolicAD.DefaultBackend(),
    )
    Test.@test termination_status(model) == LOCALLY_SOLVED
    Test.@test ≈(value(x), 1.0; atol = 1e-3)
    return
end

function test_constant_subexpressions_expr()
    model = Model(Ipopt.Optimizer)
    @variable(model, 0.5 <= x <= 1.0)
    @expression(model, my_expr1, 1.0)
    @expression(model, my_expr2, x)
    @objective(model, Min, (my_expr2 - my_expr1)^2)
    set_attribute(
        model,
        MOI.AutomaticDifferentiationBackend(),
        MathOptSymbolicAD.DefaultBackend(),
    )
    optimize!(model)
    Test.@test termination_status(model) == LOCALLY_SOLVED
    Test.@test ≈(value(x), 1.0; atol = 1e-3)
    return
end

function test_logic_comparison_expr()
    if VERSION < v"1.7"
        return  # Symbolics doesn't support Base.ifelse in Julia v1.6
    end
    model = Model(Ipopt.Optimizer)
    @variable(model, -1 <= x <= 1)
    @objective(model, Max, ifelse(-0.5 <= x && x <= 0.5, 1 - x^2, 0))
    set_attribute(
        model,
        MOI.AutomaticDifferentiationBackend(),
        MathOptSymbolicAD.DefaultBackend(),
    )
    optimize!(model)
    Test.@test termination_status(model) == LOCALLY_SOLVED
    return
end

function test_logic_comparison_or_expr()
    if VERSION < v"1.7"
        return  # Symbolics doesn't support Base.ifelse in Julia v1.6
    end
    model = Model(Ipopt.Optimizer)
    @variable(model, -1 <= x <= 1)
    @objective(model, Max, ifelse(x < -0.5 || 0.5 < x, 0, 1 - x^2))
    set_attribute(
        model,
        MOI.AutomaticDifferentiationBackend(),
        MathOptSymbolicAD.DefaultBackend(),
    )
    optimize!(model)
    Test.@test termination_status(model) == LOCALLY_SOLVED
    return
end

function test_default_x_to_x()
    model = Model(Ipopt.Optimizer)
    @variable(model, 0.1 <= x <= 1)
    @objective(model, Min, x^x)
    set_attribute(
        model,
        MOI.AutomaticDifferentiationBackend(),
        MathOptSymbolicAD.ThreadedBackend(),
    )
    optimize!(model)
    Test.@test termination_status(model) == LOCALLY_SOLVED
    Test.@test ≈(value(x), 0.37; atol = 1e-2)
    return
end

function test_default_parameter()
    model = Model(Ipopt.Optimizer)
    @variable(model, 0.1 <= x <= 1)
    @variable(model, p in Parameter(2))
    @objective(model, Min, x^p)
    set_attribute(
        model,
        MOI.AutomaticDifferentiationBackend(),
        MathOptSymbolicAD.DefaultBackend(),
    )
    optimize!(model)
    Test.@test termination_status(model) == LOCALLY_SOLVED
    Test.@test ≈(value(x), 0.1; atol = 1e-2)
    Test.@test ≈(objective_value(model), 0.1^2; atol = 1e-2)
    return
end

function test_threaded_objective()
    model = Model(Ipopt.Optimizer)
    @variable(model, -1 <= x <= 1)
    @objective(model, Min, (x - 1)^4)
    set_attribute(
        model,
        MOI.AutomaticDifferentiationBackend(),
        MathOptSymbolicAD.ThreadedBackend(),
    )
    @constraint(model, sin(x) <= 2)  # non-binding constraint
    optimize!(model)
    Test.@test termination_status(model) == LOCALLY_SOLVED
    Test.@test ≈(value(x), 1.0; atol = 1e-2)
    return
end

end  # module

RunTests.runtests()

include("SymbolicAD.jl")
