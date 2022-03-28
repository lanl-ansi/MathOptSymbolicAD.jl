# SymbolicAD

**This is package is an experimental work-in-progress. Use at your own risk.**

This package contains some experiments with symbolic differentiation of JuMP
models. It is inspired by some work on
[coin-or/gravity](https://github.com/coin-or/Gravity), which made the following
observations:

 * Symbolic differentiation is slow
 * Most NLP's have a lot of structure
 * We can ask the user to provide structure in "template constraints" where
   they give the symbolic form of the constraint and then provide a set of
   data to convert from a symbolic form to the numerical form.
 * If we did symbolic differentiation on the symbolic form of the
   constraint, we'd have a symbolic derivative we could plug the data in to
   get the numerical derivative of each function
 * We don't have to ask the user to provide structure, we can infer it by
   looking at the expression tree of each constraint.

Hopefully, this should be very fast if the problem is large with few unique
constraints. So something like
```julia
@NLconstraint(model, [i=1:10_000], sin(x[i]) <= 1)
```
is great because we compute the derivative as `cos(x[i])` and then we can
fill in the 10_000 derivative functions without having to do other calculus.

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
model = Model(() -> SymbolicAD.Optimizer(Ipopt.Optimizer))
```
