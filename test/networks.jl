module TestLearningNetworks

# using Revise
using Test
using MLJ
import MLJBase
using CategoricalArrays

# TRAINABLE MODELS

X_frame, y = datanow();  # boston data

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



# FIRST TEST OF NETWORK TRAINING

task = load_reduced_ames()
X = source(task.X)
y = source(task.y)
hot = OneHotEncoder()
hotM = machine(hot, X)
W = transform(hotM, X)
knn = KNNRegressor()
knnM = machine(knn, W, y)
yhat = predict(knnM, W)

# should get "Training" for both:
@test_logs (:info, r"^Training") (:info, r"^S") (:info, r"^S") (:info, r"^Training") fit!(yhat)

# should get "Not retraining" for both:
@test_logs (:info, r"^Not retraining") (:info, r"^Not retraining") fit!(yhat)

# should get "Updating" for first, "Training" for second:
hot.drop_last = true
@test_logs (:info, r"^Updating")  (:info, r"^S") (:info, r"^S") (:info, r"^Training") fit!(yhat)

# should get "Not retraining" for both:
@test_logs (:info, r"^Not retraining") (:info, r"^Not retraining") fit!(yhat)

# should get "Not retraining" for first, "Updating for second":
knn.K = 17
@test_logs (:info, r"^Not retraining") (:info, r"^Updating") fit!(yhat)

# should get "Training" for both:
@test_logs (:info, r"^Training")  (:info, r"^S") (:info, r"^S") (:info, r"^Training") fit!(yhat, rows=1:100)

# should get "Training" for both"
@test_logs (:info, r"^Training")  (:info, r"^S") (:info, r"^S") (:info, r"^Training") fit!(yhat)

# should get "Training" for both"
@test_logs (:info, r"^Training")  (:info, r"^S") (:info, r"^S") (:info, r"^Training") fit!(yhat, force=true)


## SECOND TEST OF NETWORK TRAINING

# test smart updating of an ensemble within a network:
forest = EnsembleModel(atom=ConstantRegressor(), n=4)
forestM = machine(forest, W, y)
zhat = predict(forestM, W)
@test_logs (:info, r"^Not") (:info, r"^Train") fit!(zhat)
forest.n = 6
@test_logs (:info, r"^Not") (:info, r"^Updating") (:info, r"Build.*length 4") fit!(zhat)


## THIRD TEST OF NETWORK TRAINING

X_frame, y = datanow();  # boston data

tape = MLJ.get_tape
@test isempty(tape(nothing))
@test isempty(tape(knn1))

XX = source(X_frame[train,:])
yy = source(y[train])

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

# do nothing to the DataFrame
Xa = node(identity, Xt)

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
@test_logs(
           (:info, r"Not retraining"),
           (:info, r"Not retraining"),
           (:info, r"Not retraining"),
           fit!(yhat, rows=1:50, verbosity=1))
@test_logs(
           (:info, r"Training"),
           (:info, r"Training"),
           (:info, r"Training"),
           fit!(yhat, verbosity=1))
knn_.K =67
@test_logs(
           (:info, r"Not retraining"),
           (:info, r"Not retraining"),
           (:info, r"Updating"),
           fit!(yhat, verbosity=1))

end


## TEST REBINDING OF SOURCE DATA



true
