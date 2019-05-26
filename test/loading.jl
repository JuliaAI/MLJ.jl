module TestLoading

# using Revise
using Test
using MLJ

pkgs = keys(MLJ.metadata())
@test "MLJ" in pkgs
@test "DecisionTree" in pkgs

@test "DecisionTreeClassifier" in models()["DecisionTree"]
@test "ConstantClassifier" in models()["MLJ"]

@load DecisionTreeClassifier
@test @isdefined DecisionTreeClassifier
@load DecisionTreeRegressor pkg=DecisionTree
@test @isdefined DecisionTreeRegressor

end # module
true
