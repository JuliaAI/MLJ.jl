module TestMachines

# using Revise
using MLJ
import MLJBase
using Test
using Statistics

task = load_boston()
X, y = task();
train, test = partition(eachindex(y), 0.7);

t = Machine(KNNRegressor(K=4), X, y)
fit!(t, rows=train)
fit!(t)

predict(t, X[test,:])
@test rms(predict(t, X[test,:]), y[test]) < std(y)

mach = machine(ConstantRegressor(target_type=Float64), task)
fit!(mach)
yhat = predict_mean(mach, X)

n = nrows(X)
rms(yhat, y) ≈ std(y)*sqrt(1 - 1/n)

# test an unsupervised univariate case:
mach = machine(UnivariateStandardizer(), float.(1:5))
fit!(mach)


end # module
true
