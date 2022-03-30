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
    SymbolicAD.run_unit_benchmark(model; direct_model = true, rtol = 1e-6)
    SymbolicAD.run_solution_benchmark(model, Ipopt.Optimizer)
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
    SymbolicAD.run_unit_benchmark(model; direct_model = false)
    SymbolicAD.run_unit_benchmark(model; direct_model = true)
    return
end

function test_case5_pjm_solution()
    model = power_model("pglib_opf_case5_pjm.m")
    SymbolicAD.run_solution_benchmark(model, Ipopt.Optimizer)
    return
end

function test_clnlbeam()
    _run_clnlbeam_benchmark(; N = 10)
    _run_clnlbeam_benchmark(; N = 100)
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
            h / 2 * (u[i+1]^2 + u[i]^2) +
            350 * h / 2 * (cos(t[i+1]) + cos(t[i])) for i in 1:N
        ),
    )
    for i in 1:N
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
            350 * h / 2 * (cos(t_sol[i+1]) + cos(t_sol[i])) for i in 1:N
        ),
        350.0;
        atol = 1e-6,
    )
    return
end

function test_optimizer_case5_pjm()
    model = power_model("pglib_opf_case5_pjm.m")
    set_optimizer(model, () -> SymbolicAD.Optimizer(Ipopt.Optimizer))
    optimize!(model)
    symbolic_obj = objective_value(model)
    set_optimizer(model, Ipopt.Optimizer)
    optimize!(model)
    Test.@test isapprox(objective_value(model), symbolic_obj, atol = 1e-6)
    return
end

function test_optimizer_case5_pjm_optimize_hook()
    model = power_model("pglib_opf_case5_pjm.m")
    set_optimizer(model, Ipopt.Optimizer)
    set_optimize_hook(model, SymbolicAD.optimize_hook)
    optimize!(model)
    symbolic_obj = objective_value(model)
    set_optimize_hook(model, nothing)
    optimize!(model)
    Test.@test isapprox(objective_value(model), symbolic_obj, atol = 1e-6)
    return
end

"""
    test_optimize_hook()

This model is chosen to have a number of unique features that cover the range of
node types in JuMP's nonlinear expression data structure.
"""
function test_optimize_hook()
    model = Model(Ipopt.Optimizer)
    @variable(model, 0 <= x <= 2π, start = π)
    @NLexpression(model, y, sin(x))
    @NLparameter(model, p == 1)
    @NLconstraint(model, sqrt(y + 2) <= p + 1)
    @NLobjective(model, Min, p * y)
    set_optimize_hook(model, SymbolicAD.optimize_hook)
    optimize!(model)
    Test.@test isapprox(objective_value(model), -1; atol = 1e-6)
    Test.@test isapprox(value(x), 1.5 * π; atol = 1e-6)
    return
end

end  # module

RunTests.runtests()
