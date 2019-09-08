module TestModelSearch

# using Revise
using Test
using MLJ

pca = traits("PCA", pkg="MultivariateStats")
cnst = traits("ConstantRegressor", pkg="MLJ")

@test_throws ArgumentError MLJ.traits("Julia")

@test traits(ConstantRegressor) == cnst
@test traits(Standardizer()) == traits("Standardizer", pkg="MLJ")

@testset "localmodels" begin
    tree = traits("DecisionTreeRegressor")
    @test cnst in localmodels(modl=TestModelSearch)
    @test !(tree in localmodels(modl=TestModelSearch))
    import MLJModels
    import DecisionTree
    import MLJModels.DecisionTree_.DecisionTreeRegressor
    @test tree in localmodels(modl=TestModelSearch)
end

@testset "models() and localmodels" begin
    t(model) = model.is_pure_julia
    mods = models(t)
    @test pca in mods
    @test cnst in mods
    @test !(traits("SVC") in mods)
    mods = localmodels(t, modl=TestModelSearch)
    @test cnst in mods
    @test !(pca in mods)
    u(model) = !(model.is_supervised)
    @test pca in models(u, t)
    @test !(cnst in models(u, t))
end

end
true
