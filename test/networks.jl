module TestLearningNetworks

# using Revise
using Test
using MLJ
import MLJBase
using CategoricalArrays


# TRAINABLE MODELS

X_frame, y = datanow();  # boston data
# X = MLJBase.matrix(X_frame)

knn_ = KNNRegressor(K=7)

# split the rows:
allrows = eachindex(y);
train, valid, test = partition(allrows, 0.7, 0.15);
@test vcat(train, valid, test) == allrows

Xtrain = X_frame[train,:]
ytrain = y[train]

Xs = source(Xtrain)
ys = source(ytrain)

knn1 = machine(knn_, Xs, ys)
@test_logs (:info, r"Training") fit!(knn1, verbosity=3)
knn_.K = 5
@test_logs (:info, r"Training") fit!(knn1, rows=train[1:end-10], verbosity=2)
@test_logs (:info, r"Training") fit!(knn1, verbosity=2)
yhat = predict(knn1, Xs)
yhat(X_frame[test,:])
rms(yhat(X_frame[test,:]), y[test])

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
uscale = machine(uscale_, yy)

# get the transformed inputs, as if `uscale` were already fit:
z = transform(uscale, yy)

# construct a transformer to standardize the inputs:
scale_ = Standardizer() 
scale = machine(scale_, XX) # no need to fit

# get the transformed inputs, as if `scale` were already fit:
Xt = transform(scale, XX)

# convert DataFrame Xt to an array:
Xa = node(MLJBase.matrix, Xt)

# choose a learner and make it machine:
knn_ = KNNRegressor(K=7) # just a container for hyperparameters
knn = machine(knn_, Xa, z) # no need to fit

# get the predictions, as if `knn` already fit:
zhat = predict(knn, Xa)

# inverse transform the target:
yhat = inverse_transform(uscale, zhat)

# fit-through training:
@test_logs((:info, r"Training"),
           (:info, r"Training"),
           (:info, r"Features standarized: "),
           (:info, r" *:Crim"),
           (:info, r" *:Zn"),
           (:info, r" *:Indus"),
           (:info, r" *:NOx"),
           (:info, r" *:Rm"),
           (:info, r" *:Age"),
           (:info, r" *:Dis"),
           (:info, r" *:Rad"),
           (:info, r" *:Tax"),
           (:info, r" *:PTRatio"),
           (:info, r" *:Black"),
           (:info, r" *:LStat"),
           (:info, r"Training"),
           fit!(yhat, rows=1:50, verbosity=2))
@test_logs(# will retrain
           (:info, r"Not retraining"),
           (:info, r"Not retraining"),
           (:info, r"Not retraining"),
           fit!(yhat, rows=:, verbosity=2))
@test_logs(# will not retrain; nothing changed
           (:info, r"Not retraining"),
           (:info, r"Not retraining"),
           (:info, r"Not retraining"),
           fit!(yhat, verbosity=2))
knn_.K =4
@test_logs(# will retrain; new hyperparameter
           (:info, r"Not retraining"),
           (:info, r"Not retraining"),
           (:info, r"Training"),
           fit!(yhat, verbosity=2))
@test !MLJ.is_stale(XX) # sources always fresh

rms(yhat(X_frame[test,:]), y[test])

end
true
