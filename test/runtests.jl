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

@testset "exported_names" begin
    @test include("exported_names.jl")
end

@testset "scitypes" begin
    @test include("scitypes.jl")
end

if parse(Bool, get(ENV, "MLJ_TEST_INTEGRATION", "false"))
    @testset "integration" begin
        @test include("integration.jl")
    end
else
    @info "Integration tests skipped. Set environment variable "*
        "MLJ_TEST_INTEGRATION = \"true\" to include them.\n"*
        "Integration tests take at least one hour. "
end
