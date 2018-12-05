module TestTrainableModels

# using Revise
using MLJ
using Test
using Statistics

X_frame, y = datanow();
X = matrix(X_frame)
train, test = partition(eachindex(y), 0.7);

t = TrainableModel(KNNRegressor(K=4), X, y)
fit!(t, rows=train)
fit!(t)

predict(t, X[test,:])
@test rms(predict(t, X[test,:]), y[test]) < std(y)

end # module
