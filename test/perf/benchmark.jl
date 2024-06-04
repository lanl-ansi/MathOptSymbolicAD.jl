# Copyright (c) 2022, Oscar Dowson and contributors
# Copyright (c) 2022, Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

import Ipopt
import JuMP
import PowerModels
import MathOptSymbolicAD

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
JuMP.optimize!(
    model;
    _differentiation_backend = MathOptSymbolicAD.DefaultBackend(),
)

ProfileView.@profview(
    JuMP.optimize!(
        model;
        _differentiation_backend = MathOptSymbolicAD.DefaultBackend(),
    )
)
