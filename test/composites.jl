module TestComposites

# using Revise
using Test
using MLJ

Xin, yin = datanow(); # boston

train, test = partition(eachindex(yin), 0.7);
Xtrain = Xin[train,:];
ytrain = yin[train];

knn_model = KNNRegressor(K=4)
selector_model = FeatureSelector()

composite = SimpleCompositeModel(model=knn_model, transformer=selector_model)

fitresult, cache, report = MLJ.fit(composite, 3, Xtrain, ytrain)

# to check internals:
knn = fitresult.args[1].machine
selector = fitresult.args[1].machine

# this should trigger no retraining:
fitresult, cache, report = MLJ.update(composite, 3, fitresult, cache, Xtrain, ytrain);

# this should trigger retraining of selector and knn:
selector_model.features = [:Crim, :Rm] 
fitresult, cache, report = MLJ.update(composite, 2, fitresult, cache, Xtrain, ytrain)

# this should trigger retraining of knn only:
knn_model.K = 3
fitresult, cache, report = MLJ.update(composite, 2, fitresult, cache, Xtrain, ytrain)

predict(composite, fitresult, Xin[test,:]);

XXX = source(Xin[train,:])
yyy = source(yin[train])

mach = machine(composite, XXX, yyy)
yhat = predict(mach, XXX)
fit!(yhat, verbosity=3)
composite.transformer.features = [:NOx, :Zn]
fit!(yhat, verbosity=3)
fit!(yhat, rows=1:20, verbosity=3)
yhat(Xin[test,:])


## EXPORTING LEARNING NETWORKS AS RE-USABLE STAND-ALONE MODELS

mutable struct WrappedKNN <: Deterministic{Node}
    K::Int
end

function MLJ.fit(model::WrappedKNN, X, y)
    Xs = source(X)
    ys = source(y)

    stand = Standardizer()
    standM = machine(stand, Xs)
    W = transform(standM, Xs)
    
    boxcox = UnivariateBoxCoxTransformer()
    boxcoxM = machine(boxcox, ys)
    z = transform(boxcoxM, ys)
        
    ridge = KNNRegressor(K=model.K)
    ridgeM = machine(ridge, W, z)
    zhat = predict(ridgeM, W)
    yhat = inverse_transform(boxcoxM, zhat)

    fit!(yhat)
    return yhat
end

X, y = datanow()

model = WrappedKNN(2)
mach = machine(model, X, y)
fit!(mach)
yhat = predict(mach, X)
model.K = 7
@test predict(mach, X) != yhat


end
true
