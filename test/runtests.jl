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

@testset "metrics" begin
  @test include("metrics.jl")
end

@testset "datasets" begin
  @test include("datasets.jl")
end

@testset "KNN" begin
  @test include("KNN.jl")
end

@testset "parameters" begin
  @test include("parameters.jl")
end

@testset "networks" begin
  @test include("networks.jl")
end

@testset "Transformers" begin
  @test include("Transformers.jl")
end

@testset "DecisionTree" begin
  @test include("DecisionTree.jl")
end

@testset "TrainableModels" begin
  @test include("trainable_models.jl")
end

