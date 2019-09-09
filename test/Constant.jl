module TestConstant

# using Revise
using Test
using MLJ
import MLJBase
using CategoricalArrays
import Distributions

## REGRESSOR

# Any X will do for constant models:
X = NamedTuple{(:x1,:x2,:x3)}((rand(10), rand(10), rand(10)))
y = [1.0, 1.0, 2.0, 2.0]

model = ConstantRegressor(distribution_type=
                          Distributions.Normal{Float64})
fitresult, cache, report = MLJ.fit(model, 1, X, y)

d=Distributions.Normal(1.5, 0.5)
@test fitresult == d
@test predict(model, fitresult, X) == fill(d, 10)
@test predict_mean(model, fitresult, X) == fill(1.5, 10)

MLJBase.info_dict(model)
MLJBase.info_dict(MLJ.DeterministicConstantRegressor)


## CLASSIFIER

yraw = ["Perry", "Antonia", "Perry", "Skater"]
y = categorical(yraw)

model = ConstantClassifier()
fitresult, cache, report =  MLJ.fit(model, 1, X, y)
d = MLJ.UnivariateFinite([y[1], y[2], y[4]], [0.5, 0.25, 0.25]) 
@test fitresult == d

yhat = predict_mode(model, fitresult, X)
@test MLJ.classes(yhat[1]) == MLJ.classes(y[1])
@test yhat[5] == y[1] 
@test length(yhat) == 10

yhat = predict(model, fitresult, X)
@test yhat == fill(d, 10)

MLJBase.info_dict(model)
MLJBase.info_dict(MLJ.DeterministicConstantClassifier)

end # module
true
