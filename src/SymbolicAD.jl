# Copyright (c) 2022: Oscar Dowson, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module SymbolicAD

import Base.Meta: isexpr
import MathOptInterface
import SparseArrays
import Symbolics

const MOI = MathOptInterface

abstract type AbstractSymbolicBackend <:
              MOI.Nonlinear.AbstractAutomaticDifferentiation end

include("nonlinear_oracle.jl")
include("ThreadedBackend.jl")

end
