# It is suggested that test code for MLJ.jl include files be placed in
# a file of the same name under "test/" (and included below) and that
# this test code be wrapped in a module. Any new module name will do -
# eg, `module TestDatasets` for code testing `datasets.jl`.

using Distributed
addprocs(2)

@everywhere begin
using MLJ
using MLJBase
using Test
using Random
end

@testset "utilities" begin
  @test include("utilities.jl")
end

@testset "tuning" begin
    @test include("tuning.jl")
end

@testset "learning_curves" begin
    @test include("learning_curves.jl")
end

@testset "ensembles" begin
    @test include("ensembles.jl")
end

@testset "matching models to data" begin
    @test include("model_matching.jl")
end

@testset "tasks" begin
  @test include("tasks.jl")
end

@testset "scitypes" begin
    @test include("scitypes.jl")
end


