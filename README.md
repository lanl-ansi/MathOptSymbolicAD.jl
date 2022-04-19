# SymbolicAD

**This is package is an experimental work-in-progress. Use at your own risk.**

This package implements an experimental symbolic automatic differentiation
backend for JuMP.

## Installation

Install SymbolicAD as follows:
```julia
import Pkg
Pkg.add("https://github.com/odow/SymbolicAD.jl)
```

## Use with JuMP

```julia
using JuMP
import Ipopt
import SymbolicAD
model = Model(Ipopt.Optimizer)
# ...
optimize!(model; differentiation_backend = SymbolicAD.DefaultBackend())
```

## Background

`SymbolicAD` is inspired by Hassan Hijazi's work on
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
@NLconstraint(model, [i=1:10_000], sin(x[i]) <= 1)
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
