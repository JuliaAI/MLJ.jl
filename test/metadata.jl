module TestMetadata

# using Revise
using Test
using MLJ
import MLJBase

metadata_file = joinpath(@__DIR__, "..", "src", "registry", "Metadata.toml")
pca = MLJ.Handle("PCA", "MultivariateStats")
cnst = MLJ.Handle("ConstantRegressor", "MLJ")
i = MLJ.info_given_handle(metadata_file)[cnst]


@testset "building INFO_GIVEN_HANDLE" begin
    @test isempty(MLJ.localmodeltypes(MLJBase))
    @test issubset(Set([MLJ.SimpleDeterministicCompositeModel,
                        FooBarRegressor,
                        KNNRegressor,                                
                        MLJ.Constant.DeterministicConstantClassifier,
                        MLJ.Constant.DeterministicConstantRegressor, 
                        MLJ.DeterministicEnsembleModel,              
                        MLJ.DeterministicTunedModel,                 
                        ConstantClassifier,                          
                        ConstantRegressor,                           
                        MLJ.ProbabilisticEnsembleModel,              
                        MLJ.ProbabilisticTunedModel,                 
                        Resampler,                                   
                        FeatureSelector,                             
                        OneHotEncoder,                               
                        Standardizer,                                
                        UnivariateBoxCoxTransformer,
                        UnivariateStandardizer]), MLJ.localmodeltypes(MLJ))
    @test MLJ.info_given_handle(metadata_file)[pca][:name] == "PCA"
    @test MLJ.info_given_handle(metadata_file)[cnst] == MLJBase.info(ConstantRegressor)
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

end
true
