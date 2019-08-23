module TestLoading

# using Revise
using Test
using MLJ

metadata_file = joinpath(@__DIR__, "..", "src", "registry", "Metadata.toml")
pca = MLJ.Handle("PCA", "MultivariateStats")
cnst = MLJ.Handle("ConstantRegressor", "MLJ")
i = MLJ.info_given_handle(metadata_file)[cnst]

@testset "building INFO_GIVEN_HANDLE" begin
    @test MLJ.info_given_handle(metadata_file)[pca][:name] == "PCA"
    @test MLJ.info_given_handle(metadata_file)[cnst] == info(ConstantRegressor)
end

h = Vector{Any}(undef, 7)
h[1] = MLJ.Handle("1", "a")
h[3] = MLJ.Handle("1", "b")
h[2] = MLJ.Handle("2", "b")
h[4] = MLJ.Handle("3", "b")
h[5] = MLJ.Handle("4", "c")
h[6] = MLJ.Handle("3", "d")
h[7] = MLJ.Handle("5", "e")
info_given_handle = Dict([h[j]=>i for j in 1:7]...)

@testset "building AMBIGUOUS_NAMES" begin
    @test Set(MLJ.ambiguous_names(info_given_handle)) == Set(["1", "3"])
end

@testset "building PKGS_GIVEN_NAME" begin
    d = MLJ.pkgs_given_name(info_given_handle)
    @test Set(d["1"]) == Set(["a", "b"])
    @test d["2"]==["b",]
    @test Set(d["3"]) == Set(["b", "d"])
    @test d["4"] == ["c",]
    @test d["5"] == ["e",]
end

@testset "building NAMES" begin
    model_names = MLJ.model_names(info_given_handle)
    @test Set(model_names) == Set(["1", "2", "3", "4", "5"])
end

@testset "Handle constructors" begin
    @test MLJ.Handle("PCA") == MLJ.Handle("PCA", "MultivariateStats")
    @test MLJ.model("PCA") == MLJ.Handle("PCA", "MultivariateStats")
    @test_throws ArgumentError MLJ.model("Julia")
    # TODO: add tests here when duplicate model names enter registry
end

@testset "info" begin
    @test info(model("PCA"))[:name] == "PCA"
    @test info("PCA") == info(model("PCA"))
end

@testset "localmodels" begin
    tree = model("DecisionTreeRegressor")
    @test cnst in localmodels(mod=TestLoading)
    @test !(tree in localmodels(mod=TestLoading))
    import MLJModels
    import DecisionTree
    import MLJModels.DecisionTree_.DecisionTreeRegressor
    @test tree in localmodels(mod=TestLoading)
end

@testset "models() and localmodels" begin
    t(handle) = info(handle)[:is_pure_julia]
    mods = models(t)
    @test pca in mods
    @test cnst in mods
    @test !(model("SVC") in mods)
    mods = localmodels(t, mod=TestLoading)
    @test cnst in mods
    @test !(pca in mods)
end

@testset "loading of model implementations" begin
    @load DecisionTreeClassifier pkg=DecisionTree verbosity=1
    @test (@isdefined DecisionTreeClassifier)
    @test_logs((:info, r"^A model named"),
               load_implementation("DecisionTreeClassifier", mod=TestLoading))
end

end # module
true
