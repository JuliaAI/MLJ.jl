using Distributed
addprocs(2)

@everywhere begin
    using MLJ
    using MLJBase
    using Test
    using Random
end

@testset "version" begin
    @test include("version.jl")
end

# TODO: restore after MLJNearestNeighbors #1 is resolved:
# @testset "ensembles" begin
#    @test include("ensembles.jl")
# end

@testset "matching models to data" begin
    @test include("model_matching.jl")
end

@testset "scitypes" begin
    @test include("scitypes.jl")
end
