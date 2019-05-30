module TestComposites

# using Revise
using Test
using MLJ

Xin, yin = datanow(); # boston

train, test = partition(eachindex(yin), 0.7);
Xtrain = Xin[train,:];
ytrain = yin[train];

ridge_model = SimpleRidgeRegressor(lambda=0.1)
selector_model = FeatureSelector()

composite = MLJ.SimpleDeterministicCompositeModel(model=ridge_model, transformer=selector_model)

fitresult, cache, report = MLJ.fit(composite, 3, Xtrain, ytrain)

# to check internals:
ridge = fitresult.tape[2]
selector = fitresult.tape[1]
ridge_old = deepcopy(ridge)
selector_old = deepcopy(selector)

# this should trigger no retraining:
fitresult, cache, report =
    @test_logs(
        (:info, r"^Not"),
        (:info, r"^Not"),
        MLJ.update(composite, 2, fitresult, cache, Xtrain, ytrain))
@test ridge.fitresult == ridge_old.fitresult
@test selector.fitresult == selector_old.fitresult

# this should trigger update of selector and training of ridge:
selector_model.features = [:Crim, :Rm] 
fitresult, cache, report =
    @test_logs(
        (:info, r"^Updating"),
        (:info, r"^Training"),
        MLJ.update(composite, 2, fitresult, cache, Xtrain, ytrain))
@test ridge.fitresult != ridge_old.fitresult
@test selector.fitresult != selector_old.fitresult
ridge_old = deepcopy(ridge)
selector_old = deepcopy(selector)

# this should trigger updating of ridge only:
ridge_model.lambda = 1.0
fitresult, cache, report =
    @test_logs(
            (:info, r"^Not"),
            (:info, r"^Updating"),
            MLJ.update(composite, 2, fitresult, cache, Xtrain, ytrain))
@test ridge.fitresult != ridge_old.fitresult
@test selector.fitresult == selector_old.fitresult

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

mutable struct WrappedRidge <: DeterministicNetwork
    ridge
end

function MLJ.fit(model::WrappedRidge, X, y)
    Xs = source(X)
    ys = source(y)

    stand = Standardizer()
    standM = machine(stand, Xs)
    W = transform(standM, Xs)
    
    boxcox = UnivariateBoxCoxTransformer()
    boxcoxM = machine(boxcox, ys)
    z = transform(boxcoxM, ys)
        
    ridgeM = machine(model.ridge, W, z)
    zhat = predict(ridgeM, W)
    yhat = inverse_transform(boxcoxM, zhat)

    fit!(yhat)
    return yhat
end

X, y = datanow()

ridge = SimpleRidgeRegressor(lambda=0.1)
model = WrappedRidge(ridge)
mach = machine(model, X, y)
fit!(mach)
yhat=predict(mach, X)
ridge.lambda = 1.0
fit!(mach)
@test predict(mach, X) != yhat

end
true
