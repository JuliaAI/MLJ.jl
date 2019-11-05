module TestMachines

# using Revise
using MLJ
using MLJBase
using Test
using Statistics

@load DecisionTreeRegressor

N=50
X = (a=rand(N), b=rand(N), c=rand(N))
y = 2*X.a - X.c + 0.05*rand(N)

train, test = partition(eachindex(y), 0.7);

tree = DecisionTreeRegressor(max_depth=5)

t = Machine(tree, X, y)
@test_logs (:info, r"Training") fit!(t)
@test_logs (:info, r"Training") fit!(t, rows=train)
@test_logs (:info, r"Not retraining") fit!(t, rows=train)
@test_logs (:info, r"Training") fit!(t)
MLJ.recursive_setproperty!(t, :(model.max_depth),  1)
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

@testset "warnings" begin
    @test_logs((:warn, r"DecisionTreeRegressor does not support"),
               machine(tree, X, y, rand(N)))
    @test_throws DimensionMismatch machine(tree, X, y[1:end-1])
    @test_throws DimensionMismatch machine(tree, X, y, rand(N-1))
    @test_logs((:warn, r"The scitype of `y`"),
               machine(tree, X, categorical(1:N)))
    @test_logs((:warn, r"The scitype of `X`"),
               machine(tree, (x=categorical(1:N),), y))
end # module
true
