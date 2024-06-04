# Copyright (c) 2022, Oscar Dowson and contributors
# Copyright (c) 2022, Triad National Security, LLC
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module MathOptSymbolicAD

import MathOptInterface as MOI
import SparseArrays
import Symbolics

abstract type AbstractSymbolicBackend <:
              MOI.Nonlinear.AbstractAutomaticDifferentiation end

include("nonlinear_oracle.jl")
include("ThreadedBackend.jl")

end
