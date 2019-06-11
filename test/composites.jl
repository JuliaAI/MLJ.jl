module TestComposites

# using Revise
using Test
using MLJ
using CategoricalArrays

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


## EXPORTING A NETWORK BY HAND

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


## EXPORTING THE LEARNING NETWORK DEFINED BY A NODE

using CategoricalArrays

x1 = map(n -> mod(n,3), rand(UInt8, 100)) |> categorical
x2 = randn(100)
X = (x1=x1, x2=x2)
y = x2.^2

@load DecisionTreeRegressor

Xs = source(X)
ys = source(y)
z = log(ys)
stand = UnivariateStandardizer()
standM = machine(stand, z)
u = transform(standM, z)
hot = OneHotEncoder()
hotM = machine(hot, Xs)
W = transform(hotM, Xs)
knn = KNNRegressor()
knnM = machine(knn, W, u)
oak = DecisionTreeRegressor()
oakM = machine(oak, W, u)
uhat = 0.5*(predict(knnM, W) + predict(oakM, W))
zhat = inverse_transform(standM, uhat)
yhat = exp(zhat)

# test that state changes after fit:
@test sum(MLJ.state(yhat) |> MLJ.flat_values) == 0
fit!(yhat)
@test sum(MLJ.state(W) |> MLJ.flat_values) == 1

# test that a node can be reconstructed from its tree representation:
t = MLJ.tree(yhat)
yhat2 = MLJ.reconstruct(t)
@test models(yhat) == models(yhat2)
@test sources(yhat) == sources(yhat2)
@test MLJ.tree(yhat) == MLJ.tree(yhat2)



end
true
