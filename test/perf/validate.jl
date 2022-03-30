import Ipopt
import PowerModels
import SymbolicAD

function power_model(case::String)
    pm = PowerModels.instantiate_model(
        joinpath(dirname(@__DIR__), "data", case),
        PowerModels.ACPPowerModel,
        PowerModels.build_opf,
    )
    return pm.model
end

model = power_model("pglib_opf_case2853_sdet.m")
SymbolicAD.run_unit_benchmark(model; direct_model = true, atol = 1e-6)
SymbolicAD.run_solution_benchmark(model, Ipopt.Optimizer; atol = 1.0)

# model = power_model("pglib_opf_case5_pjm.m")
# SymbolicAD.run_unit_benchmark(model; direct_model = true, atol = 1e-11)
