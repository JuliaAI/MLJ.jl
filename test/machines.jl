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
@notest_logs (:info, r"Training") fit!(t)
@notest_logs (:info, r"Training") fit!(t, rows=train)
@notest_logs (:info, r"Not retraining") fit!(t, rows=train)
@notest_logs (:info, r"Training") fit!(t)
setproperty!(t, :(model.K),  5)
@notest_logs (:info, r"Updating") fit!(t)

predict(t, X[test,:])
@test rms(predict(t, X[test,:]), y[test]) < std(y)

mach = machine(ConstantRegressor(), task)
@notest_logs (:info, r"Training") (:info, r"Fitted") fit!(mach)
yhat = predict_mean(mach, X)

n = nrows(X)
rms(yhat, y) ≈ std(y)*sqrt(1 - 1/n)

# test an unsupervised univariate case:
mach = machine(UnivariateStandardizer(), float.(1:5))
@notest_logs (:info, r"Training") fit!(mach)
@test isempty(params(mach))

# test a frozen NodalMachine
stand = machine(Standardizer(), source((x1=rand(10),)))
freeze!(stand)
@notest_logs (:warn, r"not trained as it is frozen\.$") fit!(stand)

end # module
true
