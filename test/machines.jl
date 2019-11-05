module TestMachines

using MLJ
using MLJBase
using Test
using Statistics

@load KNNRegressor

N=50
X = (a=rand(N), b=rand(N), c=rand(N))
y = 2*X.a - X.c + 0.05*rand(N)

train, test = partition(eachindex(y), 0.7);

t = Machine(KNNRegressor(K=4), X, y)
@test_logs (:info, r"Training") fit!(t)
@test_logs (:info, r"Training") fit!(t, rows=train)
@test_logs (:info, r"Not retraining") fit!(t, rows=train)
@test_logs (:info, r"Training") fit!(t)
MLJ.recursive_setproperty!(t, :(model.K),  5)
@test_logs (:info, r"Updating") fit!(t)

predict(t, selectrows(X,test))
@test rms(predict(t, selectrows(X, test)), y[test]) < std(y)

mach = machine(ConstantRegressor(), X, y)
@test_logs (:info, r"Training") fit!(mach)
yhat = predict_mean(mach, X)

n = nrows(X)
rms(yhat, y) ≈ std(y)*sqrt(1 - 1/n)

# test an unsupervised univariate case:
mach = machine(UnivariateStandardizer(), float.(1:5))
@test_logs (:info, r"Training") fit!(mach)
@test isempty(params(mach))

# test a frozen NodalMachine
stand = machine(Standardizer(), source((x1=rand(10),)))
freeze!(stand)
@test_logs (:warn, r"not trained as it is frozen\.$") fit!(stand)

mutable struct Scale <: MLJBase.Static
    scaling::Float64
end

@testset "static transformations" begin

    X = rand(5, 3)

    (s::Scale)(X) = s.scaling*X
    MLJBase.inv(s::Scale, X) = X/s.scaling

    double = Scale(2)
    double(X)      # <----- crash: method amiguity

    mach = fit!(machine(double, X), X)
    @test transform(mach) ≈ 2*X

end

end # module
true
