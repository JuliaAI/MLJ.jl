module TestConstant

# using Revise
using Test
using MLJ
using CategoricalArrays


## REGRESSOR

X = nothing # X is never used by constant regressors/classifiers
y = Union{Missing,Float32}[1.0, 1.0, 2.0, 2.0, missing]

model = ConstantRegressor(target_type=Union{Missing,Float32})
fitresult, cache, report = MLJ.fit(model, 1, X, y)
@test fitresult ≈ 1.5
@test predict(model, fitresult, ones(10,2)) ≈ fill(1.5, 10)

X = nothing # X is never used by constant regressors/classifiers
y = Union{Missing,Float32}[1.0, 1.0, 2.0, 2.0, missing]


## CLASSIFIER

X = nothing # X is never used by constant regressors/classifiers
yraw = Union{Missing,String}["Perry", "Antonia", "Perry", "Skater", missing]
y = categorical(yraw)

model = ConstantClassifier(target_type=Union{Missing,String})
fitresult, cache, report = MLJ.fit(model, 1, X, y)
d = MLJ.UnivariateNominal(["Perry", "Antonia", "Skater"], [0.5, 0.25, 0.25]) 
@test fitresult == d

yhat = predict_mode(model, fitresult, ones(10, 2))
@test yhat[5] == y[1] 
@test length(yhat) == 10

yhat = predict(model, fitresult, ones(10, 2))
@test yhat == fill(d, 10)

end # module
true
