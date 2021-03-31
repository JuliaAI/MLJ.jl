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

@testset "ensembles" begin
   @test include("ensembles.jl")
end

@testset "scitypes" begin
    @test include("scitypes.jl")
end
