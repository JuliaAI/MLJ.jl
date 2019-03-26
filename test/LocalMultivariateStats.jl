module TestMultivariateStats

# using Revise
using Test
using MLJ, MLJBase
using LinearAlgebra

@testset "Ridge" begin
    ## SYNTHETIC DATA TEST

    # Define some linear, noise-free, synthetic data:
    bias = -42.0
    coefficients = Float64[1, 3, 7]
    A = randn(1000, 3)
    Xtable = MLJBase.table(A)
    y = A*coefficients

    # Train model on all data with no regularization and no
    # standardization of target:
    ridge = RidgeRegressor(lambda=0.0)

    ridgeM = machine(ridge, Xtable, y)
    fit!(ridgeM)

    # Training error:
    @test rms(predict(ridgeM, Xtable), y) < 1e-12

    # Get the true bias?
    @test abs(ridgeM.fitresult.bias) < 1e-10
    @test norm(ridgeM.fitresult.coefficients - coefficients) < 1e-10

    info(ridge)


    ## TEST OF OTHER METHODS ON REAL DATA

    # Load some data and define train/test rows:
    Xtable, y = datanow() # boston
    #y = log.(y) # log of the SalePrice
    train, test = partition(eachindex(y), 0.7); # 70:30 split

    # Instantiate a model:
    ridge = RidgeRegressor(lambda=0.1)

    # Build a machine:
    ridgeM = machine(ridge, Xtable, y)

    fit!(ridgeM, rows=train)

    # # tune lambda:
    # lambdas, rmserrors = @curve λ map(x->10^x, (range(-6, stop=-2, length=100))) begin
    #     ridge.lambda = λ
    #     fit!(ridgeM, verbosity=0)
    #     rms(predict(ridgeM, Xtable[test,:]), y[test])
    # end

    # # set lambda to the optimal value and do final train:
    # ridge.lambda = lambdas[argmin(rmserrors)]
    # fit!(ridgeM, rows=train)
    # rms(predict(ridgeM, Xtable[test,:]), y[test])

    # TODO: check this score is reasonable
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

    # machinery
    pca = machine(barepca, X)
    fit!(pca)

    Xtr2 = MLJBase.matrix(transform(barepca, fitresult, X))
    @test abs.(Xtr2) ≈ Xtr_ref
end

end
true
