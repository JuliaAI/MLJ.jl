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
fit!(yhat, rows=1:100, verbosity=2)
fit!(yhat, rows=:, verbosity=2) # will retrain 
fit!(yhat, verbosity=2) # will not retrain; nothing changed
knn_.K =4
fit!(yhat, verbosity=2) # will retrain; new hyperparameter
@test !MLJ.is_stale(XX) # sources always fresh

rms(yhat(X_frame[test,:]), y[test])


## MAKE A COMPOSITE MODEL

import MLJ: Supervised, Unsupervised, Node, NodalTrainableModel, MLJType

mutable struct WetSupervised{L<:Supervised,
                             TX<:Unsupervised,
                             Ty<:Unsupervised} <: Supervised{Node}
    learner::L
    transformer_X::TX
    transformer_y::Ty
end

import DecisionTree

tree_ = DecisionTreeClassifier(target_type=Int)
selector_ = FeatureSelector()
encoder_ = ToIntTransformer()

composite = WetSupervised(tree_, selector_, encoder_)

Xin, yin = X_and_y(load_iris());
train, test = partition(eachindex(yin), 0.7);
true
Xtrain = Xin[train,:];
ytrain = yin[train];

import MLJ: fit, predict, update
function fit(composite::WetSupervised, verbosity, Xtrain, ytrain)

    X = source(Xtrain) # instantiates a source node
    y = source(ytrain)
    
    t_X = trainable(composite.transformer_X, X)
    t_y = trainable(composite.transformer_y, y)

    Xt = transform(t_X, X)
    yt = transform(t_y, y)

    l = trainable(composite.learner, Xt, yt)
    zhat = predict(l, Xt)

    yhat = inverse_transform(t_y, zhat)
    fit!(yhat, verbosity=verbosity)

    fitresult = yhat
    report = l.report
    cache = l
    
    return fitresult, cache, report

end

function update(composite::WetSupervised, verbosity, fitresult, cache,
                X, y)
    fit!(fitresult; verbosity=verbosity)
    return fitresult, cache, cache.report
end

predict(composite::WetSupervised, fitresult, Xnew) = fitresult(Xnew)

# let's train the composite:
fitresult, cache, report = fit(composite, 3, Xtrain, ytrain)

# to check internals:
encoder = fitresult.trainable
tree = fitresult.args[1].trainable
selector = fitresult.args[1].trainable

# this should trigger no retraining:
fitresult, cache, report = update(composite, 3, fitresult, cache, Xtrain, ytrain);

# this should trigger retraining of encoder and tree
encoder_.initial_label = 14
fitresult, cache, report = update(composite, 3, fitresult, cache, Xtrain, ytrain);

# this should trigger retraining of selector and tree:
selector_.features = [:petal_length, :petal_width] 
fitresult, cache, report = update(composite, 2, fitresult, cache, Xtrain, ytrain)

# this should trigger retraining of tree only:
tree_.max_depth = 2
fitresult, cache, report = update(composite, 2, fitresult, cache, Xtrain, ytrain)

# this should trigger retraining of all parts:
encoder_.initial_label = 43
selector_.features = []
fitresult, cache, report = update(composite, 2, fitresult, cache, Xtrain, ytrain)

predict(composite, fitresult, Xin[test,:])

XXX = source(Xin[train,:])
yyy = source(yin[train])

composite_ = trainable(composite, XXX, yyy)
yhat = predict(composite_, XX)
fit!(yhat, verbosity=3)
composite.transformer_X.features = [:petal_length]
fit!(yhat, verbosity=3)
yhat(Xin[test,:])

end
true
