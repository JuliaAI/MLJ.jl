module TestParameters

# using Revise
using MLJ
using Test

import MLJ: Scale, scale, transform, inverse_transform

mutable struct DummyModel <: Deterministic
    K::Int
    metric::Float64
    kernel::Char
end

dummy_model = DummyModel(4, 9.5, 'k')
@test params(dummy_model) ==
    (K = 4, metric = 9.5, kernel = 'k')


mutable struct SuperModel <: Deterministic
    lambda::Float64
    model1::DummyModel
    model2::DummyModel
end

dummy1 = DummyModel(1, 9.5, 'k')
dummy2 = DummyModel(2, 9.5, 'k')
super_model = SuperModel(0.5, dummy1, dummy2) 
params(super_model)

tree = params(super_model)

p1 = range(dummy_model, :K, lower=1, upper=10, scale=:log10) 
p2 = range(dummy_model, :kernel, values=['c', 'd']) 
p3 = range(super_model, :lambda, lower=0.1, upper=1, scale=:log2) 
p4 = range(dummy_model, :K, lower=1, upper=3, scale=x->2x) 
@test_throws ErrorException range(dummy_model, :K, lower=1, values=['c', 'd'])
@test_throws ErrorException range(dummy_model, :kernel, upper=10)

@test MLJ.scale(p1) == :log10
@test MLJ.scale(p2) == :none
@test MLJ.scale(p3) == :log2
@test MLJ.scale(p4) == :custom
@test scale(sin) === sin
@test transform(Scale, scale(:log), ℯ) == 1
@test inverse_transform(Scale, scale(:log), 1) == float(ℯ)

@test MLJ.iterator(p1, 5)  == [1, 2, 3, 6, 10]
@test MLJ.iterator(p2) == collect(p2.values)
u = 2^(log2(0.1)/2)
@test MLJ.iterator(p3, 3) ≈ [0.1, u, 1]
@test MLJ.iterator(p4, 3) == [2, 4, 6]

# test ranges constructed from neste parameters specified with dots:
q1 = range(super_model, :(model1.K) , lower=1, upper=10, scale=:log10) 
@test iterator(q1, 5) == iterator(p1, 5)
q2 = range

# test unwinding of iterators
iterators = ([1, 2], ["a","b"], ["x", "y", "z"])
@test MLJ.unwind(iterators...) ==
[1  "a"  "x";
 2  "a"  "x";
 1  "b"  "x";
 2  "b"  "x";
 1  "a"  "y";
 2  "a"  "y";
 1  "b"  "y";
 2  "b"  "y";
 1  "a"  "z";
 2  "a"  "z";
 1  "b"  "z";
 2  "b"  "z"]


end
true
