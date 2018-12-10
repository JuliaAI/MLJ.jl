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
boxcox_model = UnivariateBoxCoxTransformer()

composite = SimpleCompositeRegressor(regressor=knn_model,
                               transformer_X=selector_model,
                               transformer_y=boxcox_model)

fitresult, cache, report = MLJ.fit(composite, 3, Xtrain, ytrain)

# to check internals:
boxcox = fitresult.machine
knn = fitresult.args[1].machine
selector = fitresult.args[1].machine

# this should trigger no retraining:
fitresult, cache, report = MLJ.update(composite, 3, fitresult, cache, Xtrain, ytrain);

# this should trigger retraining of boxcox and knn
boxcox_model.n = 14
fitresult, cache, report = MLJ.update(composite, 3, fitresult, cache, Xtrain, ytrain);

# this should trigger retraining of selector and knn:
selector_model.features = [:Crim, :Rm] 
fitresult, cache, report = MLJ.update(composite, 2, fitresult, cache, Xtrain, ytrain)

# this should trigger retraining of knn only:
knn_model.K = 3
fitresult, cache, report = MLJ.update(composite, 2, fitresult, cache, Xtrain, ytrain)

# this should trigger retraining of all parts:
boxcox_model.n = 19
selector_model.features = []
fitresult, cache, report = MLJ.update(composite, 2, fitresult, cache, Xtrain, ytrain)

predict(composite, fitresult, Xin[test,:])

XXX = source(Xin[train,:])
yyy = source(yin[train])

composite_model = machine(composite, XXX, yyy)
yhat = predict(composite_model, XXX)
fit!(yhat, verbosity=3)
composite.transformer_X.features = [:NOx, :Zn]
fit!(yhat, verbosity=3)
yhat(Xin[test,:])

# TODO: test above with a linear model and use
# transform_y=UnivariateStandarizer() and check this has no impact on
# predictions.

end
true
