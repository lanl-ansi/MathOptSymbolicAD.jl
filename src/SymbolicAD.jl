module SymbolicAD

import Base.Meta: isexpr
import MathOptInterface
import SparseArrays
import Symbolics

const MOI = MathOptInterface

include("nonlinear_oracle.jl")
include("MOI_wrapper.jl")

end
