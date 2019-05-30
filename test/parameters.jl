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

changes = (lambda = 1.0, model2 = (K = 3,))
set_params!(super_model, changes)

@test params(super_model) == (lambda = 1.0,
      model1 = (K = 1, metric = 9.5, kernel = 'k'),
      model2 = (K = 3, metric = 9.5, kernel = 'k'))

@test MLJ.flat_keys(params(super_model)) == ["lambda", "model1.K", "model1.metric", "model1.kernel",
                                "model2.K", "model2.metric", "model2.kernel"]

@test MLJ.flat_length(changes) == 2 
@test MLJ.flat_length(params(super_model)) == 7

@test MLJ.flat_values(params(dummy_model)) == (4, 9.5, 'k')
@test MLJ.flat_values(params(super_model)) ==
    (1.0, 1, 9.5, 'k', 3, 9.5, 'k')
@test MLJ.flat_values(changes) == (1.0, 3)

@test copy(changes) == changes

tree = params(super_model)

# copy with changes:
@test copy(params(dummy_model), (42, 7.2, 'r')) ==
    (K = 42, metric = 7.2, kernel = 'r')
@test copy(tree, (2.0, 2, 19, 'z', 6, 20, 'x')) ==
    (lambda = 2.0, model1 = (K = 2, metric = 19, kernel = 'z'),
          model2 = (K = 6, metric = 20, kernel = 'x'))


p1 = range(dummy_model, :K, lower=1, upper=10, scale=:log10) 
p2 = range(dummy_model, :kernel, values=['c', 'd']) 
p3 = range(super_model, :lambda, lower=0.1, upper=1, scale=:log2) 
p4 = range(dummy_model, :K, lower=1, upper=3, scale=x->2x) 
@test_throws ErrorException range(dummy_model, :K, lower=1, values=['c', 'd'])
@test_throws ErrorException range(dummy_model, :kernel, upper=10)

@test occursin(r"\e\[34mNumericRange\{K\} @ .*\e\[39m", repr(TestParameters.p1))
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

# test iterator of a nested_iterators (parameter space iterators):
nested_iterators = (lambda = MLJ.iterator(p3, 2),
                        model1 = (K = MLJ.iterator(p1, 2),
                                         kernel = MLJ.iterator(p2)))
models = MLJ.iterator(super_model, nested_iterators)
@test map(MLJ.params, models) ==
    [(lambda = 0.1, model1 = (K = 1, metric = 9.5, kernel = 'c'),
            model2 = (K = 3, metric = 9.5, kernel = 'k')), 
     (lambda = 1.0, model1 = (K = 1, metric = 9.5, kernel = 'c'),
            model2 = (K = 3, metric = 9.5, kernel = 'k')), 
     (lambda = 0.1, model1 = (K = 10, metric = 9.5, kernel = 'c'),
            model2 = (K = 3, metric = 9.5, kernel = 'k')),
     (lambda = 1.0, model1 = (K = 10, metric = 9.5, kernel = 'c'),
            model2 = (K = 3, metric = 9.5, kernel = 'k')),
     (lambda = 0.1, model1 = (K = 1, metric = 9.5, kernel = 'd'),
            model2 = (K = 3, metric = 9.5, kernel = 'k')), 
     (lambda = 1.0, model1 = (K = 1, metric = 9.5, kernel = 'd'),
            model2 = (K = 3, metric = 9.5, kernel = 'k')), 
     (lambda = 0.1, model1 = (K = 10, metric = 9.5, kernel = 'd'),
            model2 = (K = 3, metric = 9.5, kernel = 'k')),
     (lambda = 1.0, model1 = (K = 10, metric = 9.5, kernel = 'd'),
            model2 = (K = 3, metric = 9.5, kernel = 'k'))]

end
true
