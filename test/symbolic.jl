# Copyright (c) 2022, Oscar Dowson and contributors
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE.md file.

module TestMathOptSymbolicAD

using JuMP
using Test
import MathOptSymbolicAD

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            Test.@testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_derivative()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @variable(model, z)
    @testset "$f" for (f, fp) in Any[
        # derivative(::Real, ::MOI.VariableIndex)
        1.0=>0.0,
        1.23=>0.0,
        # derivative(f::MOI.VariableIndex, x::MOI.VariableIndex)
        x=>1.0,
        y=>0.0,
        # derivative(f::MOI.ScalarAffineFunction{T}, x::MOI.VariableIndex)
        1.0*x=>1.0,
        1.0*x+2.0=>1.0,
        2.0*x+2.0=>2.0,
        2.0*x+y+2.0=>2.0,
        2.0*x+y+z+2.0=>2.0,
        # derivative(f::MOI.ScalarQuadraticFunction{T}, x::MOI.VariableIndex)
        QuadExpr(1.0 * x)=>1.0,
        QuadExpr(1.0 * x + 0.0 * y)=>1.0,
        x*y=>1.0*y,
        y*x=>1.0*y,
        x^2=>2.0*x,
        x^2+3x+4=>2.0*x+3.0,
        (x-1.0)^2=>2.0*(x-1),
        (3*x+1.0)^2=>6.0*(3x+1),
        # Univariate
        #   f.head == :+
        @force_nonlinear(+x)=>1,
        @force_nonlinear(+sin(x))=>cos(x),
        #   f.head == :-
        @force_nonlinear(-sin(x))=>-cos(x),
        #   f.head == :abs
        @force_nonlinear(
            abs(sin(x))
        )=>op_ifelse(op_greater_than_or_equal_to(sin(x), 0), 1, -1)*cos(x),
        #   f.head == :sign
        sign(x)=>false,
        # SYMBOLIC_UNIVARIATE_EXPRESSIONS
        sin(x)=>cos(x),
        cos(x)=>-sin(x),
        log(x)=>1/x,
        log(2x)=>1/(2x)*2.0,
        # f.head == :+
        sin(x)+cos(x)=>cos(x)-sin(x),
        # f.head == :-
        sin(x)-cos(x)=>cos(x)+sin(x),
        # f.head == :*
        @force_nonlinear(*(x, y, z))=>@force_nonlinear(*(y, z)),
        @force_nonlinear(*(y, x, z))=>@force_nonlinear(*(y, z)),
        @force_nonlinear(*(y, z, x))=>@force_nonlinear(*(y, z)),
        # :^
        sin(x)^2=>@force_nonlinear(*(2.0, sin(x), cos(x))),
        sin(x)^1=>cos(x),
        # :/
        @force_nonlinear(/(x, 2))=>0.5,
        @force_nonlinear(
            x^2 / (x + 1)
        )=>@force_nonlinear((*(2, x, x + 1) - x^2) / (x + 1)^2),
        # :ifelse
        op_ifelse(z, x^2, x)=>op_ifelse(z, 2x, 1),
        # :atan
        # :min
        min(x, x^2)=>op_ifelse(op_less_than_or_equal_to(x, min(x, x^2)), 1, 2x),
        # :max
        max(
            x,
            x^2,
        )=>op_ifelse(op_greater_than_or_equal_to(x, max(x, x^2)), 1, 2x),
        # comparisons
        op_greater_than_or_equal_to(x, y)=>false,
        op_equal_to(x, y)=>false,
    ]
        g = MathOptSymbolicAD.derivative(moi_function(f), index(x))
        h = MathOptSymbolicAD.simplify(g)
        if !(h ≈ moi_function(fp))
            @show h
            @show f
        end
        @test h ≈ moi_function(fp)
    end
    return
end

function test_gradient()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @variable(model, z)
    @testset "$f" for (f, fp) in Any[
        # ::Real
        1.0=>Dict(),
        # ::AffExpr
        x=>Dict(x => 1),
        x+y=>Dict(x => 1, y => 1),
        2x+y=>Dict(x => 2, y => 1),
        2x+3y+1=>Dict(x => 2, y => 3),
        # ::QuadExpr
        2x^2+3y+z=>Dict(x => 4x, y => 3, z => 1),
        # ::NonlinearExpr
        sin(x)=>Dict(x => cos(x)),
        sin(x + y)=>Dict(x => cos(x + y), y => cos(x + y)),
        sin(x + 2y)=>Dict(x => cos(x + 2y), y => cos(x + 2y) * 2),
    ]
        g = MathOptSymbolicAD.gradient(moi_function(f))
        h = Dict{MOI.VariableIndex,Any}(
            index(k) => moi_function(v) for (k, v) in fp
        )
        @test length(g) == length(h)
        for k in keys(g)
            @test g[k] ≈ h[k]
        end
    end
    return
end

function test_simplify()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @variable(model, z)
    @testset "$f" for (f, fp) in Any[
        # simplify(f)
        x=>x,
        # simplify(f::MOI.ScalarAffineFunction{T})
        AffExpr(2.0)=>2.0,
        # simplify(f::MOI.ScalarQuadraticFunction{T})
        QuadExpr(x + 1)=>x+1,
        # simplify(f::MOI.ScalarNonlinearFunction)
        @force_nonlinear(sin(*(3, x^0)))=>sin(3),
        sin(log(x))=>sin(log(x)),
        op_ifelse(z, x, 0)=>op_ifelse(z, x, 0),
        # simplify(::Val{:*}, f::MOI.ScalarNonlinearFunction)
        @force_nonlinear(*(x, *(y, z)))=>@force_nonlinear(*(x, y, z)),
        @force_nonlinear(
            *(x, *(y, z, *(x, 2)))
        )=>@force_nonlinear(*(x, y, z, x, 2)),
        @force_nonlinear(*(x, 3, 2))=>@force_nonlinear(*(x, 6)),
        @force_nonlinear(*(3, x, 2))=>@force_nonlinear(*(6, x)),
        @force_nonlinear(*(x, 1))=>x,
        @force_nonlinear(*(-(x, x), 1))=>0,
        # simplify(::Val{:+}, f::MOI.ScalarNonlinearFunction)
        @force_nonlinear(+(x, +(y, z)))=>@force_nonlinear(+(x, y, z)),
        +(sin(x), -cos(x))=>sin(x)-cos(x),
        @force_nonlinear(+(x, 1, 2))=>@force_nonlinear(+(x, 3)),
        @force_nonlinear(+(1, x, 2))=>@force_nonlinear(+(3, x)),
        @force_nonlinear(+(x, 0))=>x,
        @force_nonlinear(+(-(x, x), 0))=>0,
        # simplify(::Val{:-}, f::MOI.ScalarNonlinearFunction)
        @force_nonlinear(-(-(x)))=>x,
        @force_nonlinear(-(x, 0))=>x,
        @force_nonlinear(-(0, x))=>@force_nonlinear(-x),
        @force_nonlinear(-(x, x))=>0,
        @force_nonlinear(-(x, -y))=>@force_nonlinear(x + y),
        @force_nonlinear(-(x, y))=>@force_nonlinear(x - y),
        # simplify(::Val{:^}, f::MOI.ScalarNonlinearFunction)
        @force_nonlinear(^(x, 0))=>1,
        @force_nonlinear(^(x, 1))=>x,
        @force_nonlinear(^(0, x))=>0,
        @force_nonlinear(^(1, x))=>1,
        x^y=>x^y,
    ]
        g = MathOptSymbolicAD.simplify(moi_function(f))
        if !(g ≈ moi_function(fp))
            @show f
            @show g
        end
        @test g ≈ moi_function(fp)
    end
    return
end

function test_variable()
    model = Model()
    @variable(model, x)
    @variable(model, y)
    @variable(model, z)
    @testset "$f" for (f, fp) in Any[
        # ::Real
        1.0=>[],
        # ::VariableRef,
        x=>[x],
        # ::AffExpr
        AffExpr(2.0)=>[],
        x+1=>[x],
        2x+1=>[x],
        2x+y+1=>[x, y],
        y+1+z=>[y, z],
        # ::QuadExpr
        zero(QuadExpr)=>[],
        QuadExpr(x + 1)=>[x],
        QuadExpr(x + 1 + y)=>[x, y],
        x^2=>[x],
        x^2+x=>[x],
        x^2+y=>[y, x],
        x*y=>[x, y],
        y*x=>[y, x],
        # ::NonlinearExpr
        sin(x)=>[x],
        sin(x + y)=>[x, y],
        sin(x)*cos(y)=>[x, y],
    ]
        @test MathOptSymbolicAD.variables(moi_function(f)) == index.(fp)
    end
    return
end

end  # module

TestMathOptSymbolicAD.runtests()
