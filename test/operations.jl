module TestOperations

using MLJ
using MLJBase
using Test
using Random

if VERSION ≥ v"1.3.0-"
    @testset "|> syntax for pipelines" begin
        Random.seed!(142)
        @load RidgeRegressor pkg="MultivariateStats"
        @load KNNRegressor pkg="NearestNeighbors"
        X = MLJBase.table(randn(500, 5))
        y = abs.(randn(500))
        train, test = partition(eachindex(y), 0.7)

        # Feeding data directly to a supervised model
        knn = KNNRegressor(K=10)
        ŷ   = (X, y) |> knn
        fit!(ŷ, rows=train)

        # Describing a full pipeline using |> syntax.
        Xs, ys = source.((X, y))

        # "first layer"
        W = Xs |> Standardizer()
        z = ys |> UnivariateBoxCoxTransformer()
        # "second layer"
        ẑ = (W, z) |> RidgeRegressor(lambda=0.1)
        # "output layer"
        ŷ = ẑ |> inverse_transform(z)

        fit!(ŷ, rows=train)

        @test isapprox(rms(ŷ(rows=test), ys(rows=test)), 0.627123, rtol=1e-4)
    end
end # version

end
true
