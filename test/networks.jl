module TestLearningNetworks

# using Revise
using Test
using MLJ
using CategoricalArrays


# TRAINABLE MODELS

X_frame, y = datanow();  # boston data
X = matrix(X_frame)

knn_ = KNNRegressor(K=7)

# split the rows:
allrows = eachindex(y);
train, valid, test = partition(allrows, 0.7, 0.15);
@test vcat(train, valid, test) == allrows

Xtrain = X[train,:]
ytrain = y[train]

Xs = source(Xtrain)
ys = source(ytrain)

knn1 = trainable(knn_, Xs, ys)
fit!(knn1, verbosity=3)
knn_.K = 5
fit!(knn1, rows=train[1:end-10], verbosity=2)
fit!(knn1, verbosity=2)
yhat = predict(knn1, Xs)
yhat(X[test,:])
rms(yhat(X[test,:]), y[test])

@test MLJ.is_stale(knn1) == false

# TODO: compare to constant regressor and check it's significantly better


## NODES

tape = MLJ.get_tape
@test isempty(tape(nothing))
@test isempty(tape(knn1))

XX = source(X_frame[train,:])
yy = source(y[train])
@test !MLJ.is_stale(XX)

# construct a transformer to standardize the target:
uscale_ = UnivariateStandardizer()
uscale = trainable(uscale_, yy)

# get the transformed inputs, as if `uscale` were already fit:
z = transform(uscale, yy)

# construct a transformer to standardize the inputs:
scale_ = Standardizer() 
scale = trainable(scale_, XX) # no need to fit

# get the transformed inputs, as if `scale` were already fit:
Xt = transform(scale, XX)

# convert DataFrame Xt to an array:
Xa = matrix(Xt)

# choose a learner and make it trainable:
knn_ = KNNRegressor(K=7) # just a container for hyperparameters
knn = trainable(knn_, Xa, z) # no need to fit

# get the predictions, as if `knn` already fit:
zhat = predict(knn, Xa)

# inverse transform the target:
yhat = inverse_transform(uscale, zhat)

# fit-through training:
fit!(yhat, rows=1:50, verbosity=2)
fit!(yhat, rows=:, verbosity=2) # will retrain 
fit!(yhat, verbosity=2) # will not retrain; nothing changed
knn_.K =4
fit!(yhat, verbosity=2) # will retrain; new hyperparameter
@test !MLJ.is_stale(XX) # sources always fresh

rms(yhat(X_frame[test,:]), y[test])

end
true
