# MathOptSymbolicAD.jl

[![Build Status](https://github.com/lanl-ansi/MathOptSymbolicAD.jl/workflows/CI/badge.svg)](https://github.com/lanl-ansi/MathOptSymbolicAD.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/lanl-ansi/MathOptSymbolicAD.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/lanl-ansi/MathOptSymbolicAD.jl)

[MathOptSymbolicAD.jl](https://github.com/lanl-ansi/MathOptSymbolicAD.jl) is a Julia
package that implements a symbolic automatic differentiation backend for JuMP.

## License

MathOptSymbolicAD.jl is provided under a BSD-3 license as part of the Grid Optimization
Competition Solvers project, C19076.

See [LICENSE.md](https://github.com/lanl-ansi/MathOptSymbolicAD.jl/blob/master/LICENSE.md)
for details.

## Installation

Install `MathOptSymbolicAD.jl` using the Julia package manager:
```julia
import Pkg
Pkg.add("MathOptSymbolicAD")
```

## Getting help

For help, questions, comments, and suggestions, please
[open a GitHub issue](https://github.com/lanl-ansi/MathOptSymbolicAD.jl/issues/new).

## Use with JuMP

To use MathOptSymbolicAD.jl with JuMP, set the
`MOI.AutomaticDifferentiationBackend()` attribute to one of the following:

 * `MathOptSymbolicAD.DefaultBackend()`: a original backend that uses
   `Symbolics.jl`
 * `MathOptSymbolicAD.ThreadedBackend()`: a variation of `DefaultBackend` that
   additionally uses multithreading to compute the Jacobian and Hessian
   callbacks

```julia
using JuMP
import Ipopt
import MathOptSymbolicAD
model = Model(Ipopt.Optimizer)
set_attribute(
    model,
    MOI.AutomaticDifferentiationBackend(),
    MathOptSymbolicAD.DefaultBackend(),
    # or ...
    # MathOptSymbolicAD.ThreadedBackend(),
)
@variable(model, x[1:2])
@objective(model, Min, (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
optimize!(model)
```

## Background

`MathOptSymbolicAD` is inspired by Hassan Hijazi's work on
[coin-or/gravity](https://github.com/coin-or/Gravity), a high-performance
algebraic modeling language in C++.

Hassan made the following observations:

 * For large scale models, symbolic differentiation is slower than other
   automatic differentiation techniques.
 * However, most large-scale nonlinear programs have a lot of structure.
 * Gravity asks the user to provide structure in the form of
   _template constraints_, where the user gives the symbolic form of the
   constraint as well as a set of data to convert from a symbolic form to the
   numerical form.
 * Instead of differentiating each constraint in its numerical form, we can
   compute one symbolic derivative of the constraint in symbolic form, and then
   plug in the data in to get the numerical derivative of each function.
 * As a final step, if users don't provide the structure, we can still infer it
   --perhaps with less accuracy--by comparing the expression tree of each
   constraint.

The symbolic differentiation approach of Gravity works well when the problem is
large with few unique constraints. For example, a model like:
```julia
model = Model()
@variable(model, 0 <= x[1:10_000] <= 1)
@constraint(model, [i=1:10_000], sin(x[i]) <= 1)
@objective(model, Max, sum(x))
```
is ideal, because although the Jacobian matrix has 10,000 rows, we can compute
the derivative of `sin(x[i])` as `cos(x[i])`, and then fill in the Jacobian by
evaluating the derivative function instead of having to differentiation 10,000
expressions.

The symbolic differentiation approach of Gravity works poorly if there are a
large number of unique constraints in the model (which would require a lot of
expressions to be symbolically differentiated), or if the nonlinear functions
contain a large number of nonlinear terms (which would make the symbolic
derivative expensive to compute).

For more details, see Oscar's [JuMP-dev 2022 talk](https://www.youtube.com/watch?v=d_X3gj3Iz-k),
although note that the syntax has changed since the original recording.
