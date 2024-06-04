# Copyright (c) 2022, Oscar Dowson and contributors
# Copyright (c) 2022, Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

import Ipopt
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

model = power_model("pglib_opf_case2853_sdet.m")
MathOptSymbolicAD.run_unit_benchmark(model; direct_model = true, atol = 1e-6)
MathOptSymbolicAD.run_solution_benchmark(model, Ipopt.Optimizer; atol = 1.0)

# model = power_model("pglib_opf_case5_pjm.m")
# MathOptSymbolicAD.run_unit_benchmark(model; direct_model = true, atol = 1e-11)
