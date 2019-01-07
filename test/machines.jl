module TestMachines

# using Revise
using MLJ
using Test
using Statistics

X_frame, y = datanow();
X = MLJ.matrix(X_frame)
train, test = partition(eachindex(y), 0.7);

t = Machine(KNNRegressor(K=4), X, y)
fit!(t, rows=train)
fit!(t)

predict(t, X[test,:])
@test rms(predict(t, X[test,:]), y[test]) < std(y)

end # module
true
