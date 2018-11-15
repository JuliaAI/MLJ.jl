module TestLearningNetworks

@warn "More testing at \"test/networks.jl\" is required."

using Revise
using Test
using MLJ


# TRAINABLE MODELS

X_frame, y = datanow();  # boston data
X = array(X_frame)

knn_ = KNNRegressor(K=7)

# split the rows:
allrows = eachindex(y)
train, valid, test = partition(allrows, 0.7, 0.15)
@test vcat(train, valid, test) == allrows

Xtrain = X[train,:]
ytrain = y[train]

Xs = node(Xtrain)
ys = node(ytrain)

knn1 = trainable(knn_, Xs, ys)
fit!(knn1)
rms(predict(knn1, Xs(X[test,:])), ys(y[test]))

# TODO: compare to constant regressor and check it's significantly better


## LEARNING NODES

tape = MLJ.get_tape
@test isempty(tape(nothing))
@test isempty(tape(knn1))

@constant XX = node(X_frame[train,:])
@constant yy = node(y[train])

# construct a transformer to standardize the target:
uscale_ = UnivariateStandardizer()
@constant uscale = trainable(uscale_, yy)

# get the transformed inputs, as if `uscale` were already fit:
@constant z = transform(uscale, yy)

# construct a transformer to standardize the inputs:
scale_ = Standardizer() 
@constant scale = trainable(scale_, XX) # no need to fit

# get the transformed inputs, as if `scale` were already fit:
@constant Xt = transform(scale, XX)

# convert DataFrame Xt to an array:
@constant Xa = array(Xt)

# choose a learner and make it trainable:
knn_ = KNNRegressor(K=7) # just a container for hyperparameters
@constant knn = trainable(knn_, Xa, z) # no need to fit

# get the predictions, as if `knn` already fit:
@constant zhat = predict(knn, Xa)

# inverse transform the target:
@constant yhat = inverse_transform(uscale, zhat)

# fit-through training:
fit!(yhat)
freeze!

rms(yhat(X_frame[test,:]), y[test])


# another test

# Xall, yall = X_and_y(load_iris())
# train, test = partition(eachindex(yall), 0.7)
# Xtrain = Xall[train,:]
# ytrain = yall[train]

end
