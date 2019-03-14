module TestConstant

# using Revise
using Test
using MLJ
using CategoricalArrays
using DataFrames
import Distributions

## REGRESSOR

X = DataFrame(rand(10,3)) # X is never used by constant regressors/classifiers
y = [1.0, 1.0, 2.0, 2.0]

model = ConstantRegressor(target_type=Float64, distribution_type=Distributions.Normal{Float64})
fitresult, cache, report = MLJ.fit(model, 1, X, y)

d=Distributions.Normal(1.5, 0.5)
@test fitresult == d
@test predict(model, fitresult, DataFrame(ones(10,2))) == fill(d, 10)
@test predict_mean(model, fitresult, DataFrame(ones(10,2))) == fill(1.5, 10)

@show model
info(model)
info(DeterministicConstantRegressor)


## CLASSIFIER

yraw = ["Perry", "Antonia", "Perry", "Skater"]
y = categorical(yraw)

model = ConstantClassifier(target_type=String)
fitresult, cache, report = MLJ.fit(model, 1, X, y)
d = MLJ.UnivariateNominal(["Perry", "Antonia", "Skater"], [0.5, 0.25, 0.25]) 
@test fitresult == d

yhat = predict_mode(model, fitresult, DataFrame(ones(10, 2)))
@test levels(yhat) == levels(y)
@test yhat[5] == y[1] 
@test length(yhat) == 10

yhat = predict(model, fitresult, DataFrame(ones(10, 2)))
@test yhat == fill(d, 10)

@show model
info(model)
info(DeterministicConstantClassifier)

end # module
true
