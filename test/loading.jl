module TestLoading

# using Revise
using Test
using MLJ

pkgs = keys(MLJ.metadata())
@test "MLJ" in pkgs
@test "DecisionTree" in pkgs

@test "DecisionTreeClassifier" in models()["DecisionTree"]
@test "ConstantClassifier" in models()["MLJ"]

## if you put these back, need to add DecisionTree and MLJModels to
## [extras] and [targets]:
# @load DecisionTreeClassifier
# @test @isdefined DecisionTreeClassifier

end # module
true
