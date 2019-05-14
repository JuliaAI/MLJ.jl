module TestMultivariateStats

# using Revise
using Test
using MLJ, MLJBase
using LinearAlgebra
import Random.seed!

seed!(1234)

@testset "Ridge" begin
    ## SYNTHETIC DATA TEST

    # Define some linear, noise-free, synthetic data:
    bias = -42.0
    coefficients = Float64[1, 3, 7]
    n = 1000
    A = randn(n, 3)
    Xtable = MLJBase.table(A)
    y = A*coefficients

    # Train model on all data with no regularization and no
    # standardization of target:
    ridge = SimpleRidgeRegressor(lambda=0.0)

    fitresult, report, cache = fit(ridge, 0, Xtable, y)

    # Training error:
    yhat = predict(ridge, fitresult, Xtable)
    @test norm(yhat - y)/sqrt(n) < 1e-12

    # Get the true bias?
    fr = fitted_params(ridge, fitresult)
    @test norm(fr - coefficients) < 1e-10

    info(ridge)

end

@testset "PCA" begin
    task = load_crabs()

    X, y = X_and_y(task)

    barepca = PCA(pratio=0.9999)
    info(barepca)

    fitresult, cache, report = MLJBase.fit(barepca, 1, X)

    Xtr = MLJBase.matrix(MLJBase.transform(barepca, fitresult, X))

    X_array = MLJBase.matrix(X)

    # home made PCA (the sign flip is irrelevant)
    Xac = X_array .- mean(X_array, dims=1)
    U, S, _ = svd(Xac)
    Xtr_ref = abs.(U .* S')
    @test abs.(Xtr) ≈ Xtr_ref
end

end
true
