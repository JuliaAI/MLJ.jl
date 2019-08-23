module TestModelSearch

# using Revise
using Test
using MLJ

metadata_file = joinpath(@__DIR__, "..", "src", "registry", "Metadata.toml")
pca = MLJ.Handle("PCA", "MultivariateStats")
cnst = MLJ.Handle("ConstantRegressor", "MLJ")
i = MLJ.info_given_handle(metadata_file)[cnst]

@testset "localmodels" begin
    tree = model("DecisionTreeRegressor")
    @test cnst in localmodels(mod=TestModelSearch)
    @test !(tree in localmodels(mod=TestModelSearch))
    import MLJModels
    import DecisionTree
    import MLJModels.DecisionTree_.DecisionTreeRegressor
    @test tree in localmodels(mod=TestModelSearch)
end

@testset "models() and localmodels" begin
    t(handle) = info(handle)[:is_pure_julia]
    mods = models(t)
    @test pca in mods
    @test cnst in mods
    @test !(model("SVC") in mods)
    mods = localmodels(t, mod=TestModelSearch)
    @test cnst in mods
    @test !(pca in mods)
end

end
true
