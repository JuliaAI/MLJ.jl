module TestParameters

# using Revise
using MLJ
using Test

mutable struct DummyModel <: Supervised{Int}
    K::Int
    metric::Float64
    kernel::Char
end

dummy_model = DummyModel(4, 9.5, 'k')
@test get_params(dummy_model) ==
    Params(:K => 4, :metric => 9.5, :kernel => 'k')


mutable struct SuperModel <: Supervised{Any}
    lambda::Float64
    model1::DummyModel
    model2::DummyModel
end

dummy1 = DummyModel(1, 9.5, 'k')
dummy2 = DummyModel(2, 9.5, 'k')
super_model = SuperModel(0.5, dummy1, dummy2) 
get_params(super_model)

changes = Params(:lambda => 1.0, :model2 => Params(:K => 3))
set_params!(super_model, changes)

@test get_params(super_model) == Params(:lambda => 1.0,
      :model1 => Params(:K => 1, :metric => 9.5, :kernel => 'k'),
      :model2 => Params(:K => 3, :metric => 9.5, :kernel => 'k'))


@test length(changes) == 2 
@test length(get_params(super_model)) == 7

@test MLJ.flat_values(get_params(dummy_model)) == (4, 9.5, 'k')
@test MLJ.flat_values(get_params(super_model)) ==
    (1.0, 1, 9.5, 'k', 3, 9.5, 'k')
@test MLJ.flat_values(changes) == (1.0, 3)

@test copy(changes) == changes

tree = get_params(super_model)

@test copy(get_params(dummy_model), (42, 7.2, 'r')) ==
    Params(:K => 42, :metric => 7.2, :kernel => 'r')
@test copy(tree, (2.0, 2, 19, 'z', 6, 20, 'x')) ==
    Params(:lambda => 2.0, :model1 => Params(:K => 2, :metric => 19, :kernel => 'z'),
          :model2 => Params(:K => 6, :metric => 20, :kernel => 'x'))


p1 = ParamRange(dummy_model, :K, lower=1, upper=10)
p2 = ParamRange(dummy_model, :kernel, values=['c', 'd', 'k', 'r'])
@test typeof(collect(MLJ.iterator(p1, 5))[1] ) == Int
@test collect(MLJ.iterator(p2)) == [p2.values...]

end
