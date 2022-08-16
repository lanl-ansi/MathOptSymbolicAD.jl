import Ipopt
import JuMP
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

import ProfileView

model = power_model("pglib_opf_case118_ieee.m")
JuMP.set_optimizer(model, Ipopt.Optimizer)
JuMP.set_optimizer_attribute(model, "print_timing_statistics", "yes")
JuMP.optimize!(model; _differentiation_backend = SymbolicAD.DefaultBackend())

ProfileView.@profview(
    JuMP.optimize!(
        model;
        _differentiation_backend = SymbolicAD.DefaultBackend(),
    )
)
