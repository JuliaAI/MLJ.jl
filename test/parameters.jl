module TestParameters

# using Revise
using MLJ
using Test

mutable struct DummyModel <: Deterministic{Int}
    K::Int
    metric::Float64
    kernel::Char
end

dummy_model = DummyModel(4, 9.5, 'k')
@test params(dummy_model) ==
    Params(:K => 4, :metric => 9.5, :kernel => 'k')


mutable struct SuperModel <: Deterministic{Any}
    lambda::Float64
    model1::DummyModel
    model2::DummyModel
end

dummy1 = DummyModel(1, 9.5, 'k')
dummy2 = DummyModel(2, 9.5, 'k')
super_model = SuperModel(0.5, dummy1, dummy2) 
params(super_model)

changes = Params(:lambda => 1.0, :model2 => Params(:K => 3))
set_params!(super_model, changes)

@test params(super_model) == Params(:lambda => 1.0,
      :model1 => Params(:K => 1, :metric => 9.5, :kernel => 'k'),
      :model2 => Params(:K => 3, :metric => 9.5, :kernel => 'k'))


@test length(changes) == 2 
@test length(params(super_model)) == 7

@test MLJ.flat_values(params(dummy_model)) == (4, 9.5, 'k')
@test MLJ.flat_values(params(super_model)) ==
    (1.0, 1, 9.5, 'k', 3, 9.5, 'k')
@test MLJ.flat_values(changes) == (1.0, 3)

@test copy(changes) == changes

tree = params(super_model)

# copy with changes:
@test copy(params(dummy_model), (42, 7.2, 'r')) ==
    Params(:K => 42, :metric => 7.2, :kernel => 'r')
@test copy(tree, (2.0, 2, 19, 'z', 6, 20, 'x')) ==
    Params(:lambda => 2.0, :model1 => Params(:K => 2, :metric => 19, :kernel => 'z'),
          :model2 => Params(:K => 6, :metric => 20, :kernel => 'x'))


p1 = range(dummy_model, :K, lower=1, upper=10, scale=:log10) 
p2 = range(dummy_model, :kernel, values=['c', 'd']) 
p3 = range(super_model, :lambda, lower=0.1, upper=1, scale=:log2) 
p4 = strange(dummy_model, :K, lower=1, upper=3, scale=x->2x) |> last

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
nested_iterators = Params(:lambda => MLJ.iterator(p3, 2),
                        :model1 => Params(:K => MLJ.iterator(p1, 2),
                                         :kernel => MLJ.iterator(p2)))
models = MLJ.iterator(super_model, nested_iterators)
@test map(MLJ.params, models) ==
    [Params(:lambda => 0.1, :model1 => Params(:K => 1, :metric => 9.5, :kernel => 'c'),
            :model2 => Params(:K => 3, :metric => 9.5, :kernel => 'k')), 
     Params(:lambda => 1.0, :model1 => Params(:K => 1, :metric => 9.5, :kernel => 'c'),
            :model2 => Params(:K => 3, :metric => 9.5, :kernel => 'k')), 
     Params(:lambda => 0.1, :model1 => Params(:K => 10, :metric => 9.5, :kernel => 'c'),
            :model2 => Params(:K => 3, :metric => 9.5, :kernel => 'k')),
     Params(:lambda => 1.0, :model1 => Params(:K => 10, :metric => 9.5, :kernel => 'c'),
            :model2 => Params(:K => 3, :metric => 9.5, :kernel => 'k')),
     Params(:lambda => 0.1, :model1 => Params(:K => 1, :metric => 9.5, :kernel => 'd'),
            :model2 => Params(:K => 3, :metric => 9.5, :kernel => 'k')), 
     Params(:lambda => 1.0, :model1 => Params(:K => 1, :metric => 9.5, :kernel => 'd'),
            :model2 => Params(:K => 3, :metric => 9.5, :kernel => 'k')), 
     Params(:lambda => 0.1, :model1 => Params(:K => 10, :metric => 9.5, :kernel => 'd'),
            :model2 => Params(:K => 3, :metric => 9.5, :kernel => 'k')),
     Params(:lambda => 1.0, :model1 => Params(:K => 10, :metric => 9.5, :kernel => 'd'),
            :model2 => Params(:K => 3, :metric => 9.5, :kernel => 'k'))]

end
true
