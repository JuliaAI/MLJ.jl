# It is suggested that test code for MLJ.jl include files be placed in
# a file of the same name under "test/" (and included below) and that
# this test code be wrapped in a module. Any new module name will do -
# eg, `module TestDatasets` for code testing `datasets.jl`.

using MLJ
# using Revise
using Test

@constant junk=KNNRegressor()
properties(KNNRegressor)
operations(KNNRegressor)
inputs_can_be(KNNRegressor)
outputs_are(KNNRegressor)

include("TestMetrics.jl")
include("TestDatasets.jl")
include("TestKNN.jl")
include("TestLearningNetworks.jl")
include("TestTransformers.jl")
include("TestDecisionTree.jl")


