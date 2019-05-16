# It is suggested that test code for MLJ.jl include files be placed in
# a file of the same name under "test/" (and included below) and that
# this test code be wrapped in a module. Any new module name will do -
# eg, `module TestDatasets` for code testing `datasets.jl`.

using MLJ
using Test

@constant junk=KNNRegressor()

@testset "utilities" begin
  @test include("utilities.jl")
end

@testset "metrics" begin
  @test include("metrics.jl")
end

@testset "KNN" begin
  @test include("KNN.jl")
end

@testset "ridge" begin
  @test include("ridge.jl")
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

@testset "loading" begin
    @test include("loading.jl")
end

@testset "tasks" begin
  @test include("tasks.jl")
end


## TEST THE EXAMPLES

const exdir = joinpath(MLJ.srcdir, "..", "examples")

# uncomment remaining code to test examples:
# @testset "/examples" begin
#     @test include(joinpath(exdir, "using_tasks.jl"))
#     @test include(joinpath(exdir, "random_forest.jl"))
#     @test include(joinpath(exdir, "two_parameter_tune.jl"))
# end

       
