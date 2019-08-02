module TestComposites

# using Revise
using Test
using MLJ
using CategoricalArrays

Xin, yin = datanow(); # boston

train, test = partition(eachindex(yin), 0.7);
Xtrain = Xin[train,:];
ytrain = yin[train];

ridge_model = FooBarRegressor(lambda=0.1)
selector_model = FeatureSelector()

composite = MLJ.SimpleDeterministicCompositeModel(model=ridge_model, transformer=selector_model)

fitresult, cache, report = MLJ.fit(composite, 3, Xtrain, ytrain)

# to check internals:
ridge = MLJ.machines(fitresult)[1]
selector = MLJ.machines(fitresult)[2]
ridge_old = deepcopy(ridge)
selector_old = deepcopy(selector)

# this should trigger no retraining:
fitresult, cache, report =
    @notest_logs(
        (:info, r"^Not"),
        (:info, r"^Not"),
        MLJ.update(composite, 2, fitresult, cache, Xtrain, ytrain))
@test ridge.fitresult == ridge_old.fitresult
@test selector.fitresult == selector_old.fitresult

# this should trigger update of selector and training of ridge:
selector_model.features = [:Crim, :Rm] 
fitresult, cache, report =
    @notest_logs(
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
    @notest_logs(
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

ridge = FooBarRegressor(lambda=0.1)
model = WrappedRidge(ridge)
mach = machine(model, X, y)
fit!(mach)
yhat=predict(mach, X)
ridge.lambda = 1.0
fit!(mach)
@test predict(mach, X) != yhat


## EXPORTING THE LEARNING NETWORK DEFINED BY A NODE

x1 = map(n -> mod(n,3), rand(UInt8, 100)) |> categorical;
x2 = randn(100);
X = (x1=x1, x2=x2);
y = x2.^2;

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

hot2 = deepcopy(hot)
knn2 = deepcopy(knn)
ys2 = source(nothing)

# duplicate a network:
############################
yhat2 = replace(yhat, hot=>hot2, knn=>knn2, ys=>source(ys.data))

@notest_logs((:info, r"^Train.*OneHot"),
           (:info, r"^Spawn"),
           (:info, r"^Train.*Univ"),
           (:info, r"^Train.*KNN"),
           (:info, r"^Train.*Dec"), fit!(yhat2))
@test length(MLJ.machines(yhat)) == length(MLJ.machines(yhat2))
@test models(yhat) == models(yhat2)
@test sources(yhat) == sources(yhat2)
@test MLJ.tree(yhat) == MLJ.tree(yhat2)
fit!(yhat)
@test yhat() ≈ yhat2()

# this change should trigger retraining of all machines except the
# univariate standardizer:
hot2.drop_last = true
@notest_logs((:info, r"^Updating.*OneHot"),
           (:info, r"^Spawn"),
           (:info, r"^Not.*Univ"),
           (:info, r"^Train.*KNN"),
           (:info, r"^Train.*Dec"), fit!(yhat2))

# export a supervised network:
model = @from_network Composite(knn_rgs=knn, one_hot_enc=hot) <= (Xs, ys, yhat)
mach = machine(model, Xs, ys)
@notest_logs((:info, r"^Train.*Composite"),
           (:info, r"^Train.*OneHot"),
           (:info, r"^Spawn"),
           (:info, r"^Train.*Univ"),
           (:info, r"^Train.*KNN"),
           (:info, r"^Train.*Dec"), fit!(mach))
model.knn_rgs.K = 5
@notest_logs((:info, r"^Updat.*Composite"),
           (:info, r"^Not.*OneHot"),
           (:info, r"^Not.*Univ"),
           (:info, r"^Updat.*KNN"),
           (:info, r"^Not.*Dec"), fit!(mach))

# export an unsupervised model
multistand = Standardizer()
multistandM = machine(multistand, W)
W2 = transform(multistandM, W)
model = @from_network Transf(one_hot=hot) <= (Xs, W2)
mach = machine(model, Xs)
@notest_logs((:info, r"^Training.*Transf"),
           (:info, r"^Train.*OneHot"),
           (:info, r"^Spawn"),
           (:info, r"Train.*Stand"), fit!(mach))
model.one_hot.drop_last=true
@notest_logs((:info, r"^Updating.*Transf"),
           (:info, r"^Updating.*OneHot"),
           (:info, r"^Spawn"),
           (:info, r"Train.*Stand"), fit!(mach))


end
true
