using Distributed
addprocs(2)

@everywhere begin
    using MLJ
    using MLJBase
    using Test
    using Random
end

@testset "ensembles" begin
    @test include("ensembles.jl")
end

@testset "matching models to data" begin
    @test include("model_matching.jl")
end

@testset "scitypes" begin
    @test include("scitypes.jl")
end
