# It is suggested that test code for MLJ.jl include files be placed in
# a file of the same name under "test/" (and included below) and that
# this test code be wrapped in a module. Any new module name will do -
# eg, `module TestDatasets` for code testing `datasets.jl`.

using MLJ
import MLJBase

# using Revise
using Test

@constant junk=KNNRegressor()
a = rand(3,4)'
@test MLJBase.matrix(dataframe(a)) == collect(a)

@testset "metrics" begin
  @test include("metrics.jl")
end

@testset "datasets" begin
  @test include("datasets.jl")
end

@testset "KNN" begin
  @test include("KNN.jl")
end

@testset "Constant" begin
    @test include("Constant.jl")
end

@testset "parameters" begin
  @test include("parameters.jl")
end

@testset "Transformers" begin
  @test include("Transformers.jl")
end

@testset "DecisionTree" begin
  @test include("DecisionTree.jl")
end

@testset "Machines" begin
  @test include("machines.jl")
end

@testset "networks" begin
  @test include("networks.jl")
end

@testset "composites" begin
  @test include("composites.jl")
end

@testset "resampling" begin
    @test include("resampling.jl")
end

@testset "tuning" begin
    @test include("tuning.jl")
end

@testset "ensembles" begin
    @test include("ensembles.jl")
end

@testset "XGBoost" begin
  @test_broken include("XGBoost.jl")
end

@testset "MultivariateStats" begin
  @test include("LocalMultivariateStats.jl")
end

@testset "GaussianProcesses" begin
  @test include("GaussianProcesses.jl")
end

@testset "Clustering" begin
  @test include("Clustering.jl")
end
