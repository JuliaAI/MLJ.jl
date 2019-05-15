# Defines a simple deterministic regressor for MLJ testing purposes
# only. MLJ users should use RidgeRegressor from MultivariateStats.

import MLJBase
using LinearAlgebra

export SimpleRidgeRegressor

mutable struct SimpleRidgeRegressor <: MLJBase.Deterministic
    lambda::Float64
end

function SimpleRidgeRegressor(; lambda=0.0)
    simpleridgemodel = SimpleRidgeRegressor(lambda)
    message = MLJBase.clean!(simpleridgemodel)
    isempty(message) || @warn message
    return simpleridgemodel
end

function MLJ.clean!(model::SimpleRidgeRegressor)
    warning = ""
    if model.lambda < 0
        warning *= "Need lambda ≥ 0. Resetting lambda=0. "
        model.lambda = 0
    end
    return warning
end

function MLJBase.fitted_params(::SimpleRidgeRegressor, fitresult)
    return (coefficients=fitresult)
end

function MLJBase.fit(model::SimpleRidgeRegressor, verbosity::Int, X, y)
    x = MLJBase.matrix(X)
    fitresult = (x'x - model.lambda*I)\(x'y)
    cache = nothing
    report = NamedTuple()
    return fitresult, cache, report
end


function MLJBase.predict(model::SimpleRidgeRegressor, fitresult, Xnew)
    x = MLJBase.matrix(Xnew)
    return x*fitresult
end

# to hide from models generated from calls to models()
MLJBase.is_wrapper(::Type{<:SimpleRidgeRegressor}) = true


# metadata:
MLJBase.load_path(::Type{<:SimpleRidgeRegressor}) = "MLJ.SimpleRidgeRegressor"
MLJBase.package_name(::Type{<:SimpleRidgeRegressor}) = "MLJ"
MLJBase.package_uuid(::Type{<:SimpleRidgeRegressor}) = ""
MLJBase.is_pure_julia(::Type{<:SimpleRidgeRegressor}) = true
MLJBase.input_scitype_union(::Type{<:SimpleRidgeRegressor}) = MLJBase.Continuous
MLJBase.target_scitype_union(::Type{<:SimpleRidgeRegressor}) = MLJBase.Continuous
