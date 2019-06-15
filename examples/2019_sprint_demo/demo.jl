# # MLJ Demo from the June 2019 MLJ/sktime Sprint

# ### Load environment and seed RNG

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using Random
Random.seed!(1234);


# ## 1. Basics

# ### Constructing a task

using MLJ
models()

#- 

using RDatasets
boston = dataset("MASS", "Boston");
first(boston)

#-

scitypes(boston)

#-

task = supervised(data=boston,
                  target=:MedV,
                  ignore=:Chas,
                  types=Dict(:Rad=>Continuous,:Tax=>Continuous),
                  is_probabilistic=false)

#-

shuffle!(task)

#-

models(task)

#-

task.is_probabilistic=true
models(task)


# ### Binding a task to a model

@load OLSRegressor
model = OLSRegressor()

#-
mach = machine(model, task)


# ### Evaluation

evaluate!(mach, resampling=CV(nfolds=5), measure=rms)

# ### Fitting


train, test = partition(1:nrows(task), 0.7)

#-

fit!(mach, rows=train)


# ### Predicting on training data

yhat = predict(mach, rows=train);
yhat = predict(mach, task[train]);
yhat[1:3]

#-

y = task[train].y;
y[1:3]

#-

rms(yhat, y)


# ### Predicting on "new" data

X = task[test].X; 
yhat = predict(mach, X);
yhat[1:4]


# ### Specicial to probabilistic models

mean.(yhat[1:4])

#-

mean.(yhat) == predict_mean(mach, X)

#-                   

import Distributions
[Distributions.cdf(d, 10.0) for d in yhat[1:4]]


# ## 2. Random Forests - Case Study in Ensembling and Tuning

# ### Quick look at single tree

task = load_iris() # built-in task
shuffle!(task)
@load DecisionTreeClassifier
tree = DecisionTreeClassifier()
mach = machine(tree, task)
evaluate!(mach)

#-

yhat = predict(mach, rows=1:3)

#-

[pdf(d, "virginica") for d in yhat]

#-

predict_mode(mach, rows=1:3)

# ### A random forest

forest = EnsembleModel(atom=tree)

#-


mach = machine(forest, task)
r = range(forest, :n, lower=10, upper=1000, scale=:log10)
iterator(r,5)

#-

curves = learning_curve!(mach,
                         resampling=Holdout(fraction_train=0.8),
                         nested_range=(n=r,), 
                         measure=cross_entropy, n=4,
                         verbosity=0)
#-

using Plots
plot(curves.parameter_values, curves.measurements,
     xlab="number of trees", ylab="rms ")

#-

forest.n = 100; # 750 better

#-

params(forest) # all hyperparameters, as a named tuple

#-

r1 = range(tree, :n_subfeatures, lower=1, upper=4);
r2 = range(forest, :bagging_fraction, lower=0.4, upper=1.0);
nested_ranges = (atom=(n_subfeatures=r1,), 
                 bagging_fraction=r2)

#-

tuned_forest = TunedModel(model=forest, 
                          tuning=Grid(resolution=12),
                          resampling=CV(nfolds=6),
                          nested_ranges=nested_ranges, 
                          measure=cross_entropy)
#-

params(tuned_forest)

#-

mach = machine(tuned_forest, task)

#-

evaluate!(mach, resampling=Holdout(fraction_train=0.7),
          measure=[cross_entropy, misclassification_rate], verbosity=2)

#-

fitted_params(mach)

#-

best = fitted_params(mach).best_model
@show best.bagging_fraction best.atom.n_subfeatures;

#-

plot(mach)


# ## 3. Learning Networks

task = load_reduced_ames()
show(task, 1)

#-

X, y = task.X, task.y;
scitypes(X)

#-

models(task)


# ## Workflow without a pipeline (static data)

hot = OneHotEncoder()
hotM = machine(hot, X)
fit!(hotM)
Xc = transform(hotM, X)

knn = KNNRegressor()
knnM = machine(knn, Xc, y)
fit!(knnM)
yhat= predict(knnM, Xc)

#-

# ## Pipelining (dynamic data)

X = source(X)
y = source(y)

# Identical code to build network (`fit!`'s can be dropped) 

hot = OneHotEncoder()
hotM = machine(hot, X)
Xc = transform(hotM, X)

knn = KNNRegressor()
knnM = machine(knn, Xc, y)
yhat = predict(knnM, Xc)

# Fit the network in one go:

fit!(yhat)

#-

hot.drop_last=true

#-

fit!(yhat)

#-

knn.K = 7

#-

fit!(yhat)

# Instead of `yhat[1:10]` we have

yhat(rows=1:10)

# Or call on new data:

yhat(task.X)

# The new data is "plugged into" the orgin node, which must be unique:

origins(yhat) == [X,]


# ## Exporting pipeline as stand-alone model

comp = @from_network SmartKNN(one_hot_encoding=hot, knn_regressor=knn) <= (X, y, yhat)

#-
params(comp)

#-

mach = machine(comp, task)
evaluate!(mach, measure=rms)


# ## A more complicated example

using CategoricalArrays
x1 = rand(6);
x2 = categorical([mod(rand(Int),2) for i in 1:6]);
x3 = rand(6);
y = exp.(x1 -2x3 + 0.1*rand(6))
X = (x1=x1, x2=x2, x3=x3) 

# Here's a learning network that: (i) One-hot encodes the input table `X`; (ii)
# Log transforms the continuous target `y`; (iii) Fits specified
# K-nearest neighbour and ridge regressor models to the data; (iv)
# Computes an average of individual model predictions; and (v) Inverse
# transforms (exponentiates) the blended predictions.

@load RidgeRegressor

X = source(X)
y = source(y)

hot = machine(OneHotEncoder(), X)

# `W`, `z`, `zhat` and `yhat` are nodes in the network:
    
W = transform(hot, X) # one-hot encode the input
z = log(y) # transform the target

ridge = RidgeRegressor(lambda=0.1)
knn = KNNRegressor()

ridgeM = machine(ridge, W, z) 
knnM = machine(knn, W, z)

# Average the predictions of the KNN and ridge models:

zhat = 0.5*predict(ridgeM, W) + 0.5*predict(knnM, W) 

# Inverse the target transformation:

yhat = exp(zhat) 

# A tree "splat" of the learning network terminating at `yhat`:

MLJ.tree(yhat)

#-

blend = @from_network Blen(ridge=ridge, knn=knn) <= (X, y, yhat)
blend.ridge.lambda = 0.2
mach = machine(blend, load_reduced_ames())
evaluate!(mach, measure=rms)




