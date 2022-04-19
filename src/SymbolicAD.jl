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
