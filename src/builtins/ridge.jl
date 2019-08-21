# Defines a simple deterministic regressor for MLJ testing purposes
# only. MLJ users should use RidgeRegressor from MultivariateStats.

import MLJBase
using LinearAlgebra
using ScientificTypes

export FooBarRegressor

mutable struct FooBarRegressor <: MLJBase.Deterministic
    lambda::Float64
end

function FooBarRegressor(; lambda=0.0)
    simpleridgemodel = FooBarRegressor(lambda)
    message = MLJBase.clean!(simpleridgemodel)
    isempty(message) || @warn message
    return simpleridgemodel
end

function MLJ.clean!(model::FooBarRegressor)
    warning = ""
    if model.lambda < 0
        warning *= "Need lambda ≥ 0. Resetting lambda=0. "
        model.lambda = 0
    end
    return warning
end

function MLJBase.fitted_params(::FooBarRegressor, fitresult)
    return (coefficients=fitresult)
end

function MLJBase.fit(model::FooBarRegressor, verbosity::Int, X, y)
    x = MLJBase.matrix(X)
    fitresult = (x'x - model.lambda*I)\(x'y)
    cache = nothing
    report = NamedTuple()
    return fitresult, cache, report
end


function MLJBase.predict(model::FooBarRegressor, fitresult, Xnew)
    x = MLJBase.matrix(Xnew)
    return x*fitresult
end

# to hide from models generated from calls to models()
MLJBase.is_wrapper(::Type{<:FooBarRegressor}) = true

# metadata:
MLJBase.load_path(::Type{<:FooBarRegressor}) = "MLJ.FooBarRegressor"
MLJBase.package_name(::Type{<:FooBarRegressor}) = "MLJ"
MLJBase.package_uuid(::Type{<:FooBarRegressor}) = ""
MLJBase.is_pure_julia(::Type{<:FooBarRegressor}) = true
MLJBase.input_scitype(::Type{<:FooBarRegressor}) = Table(Continuous)
MLJBase.target_scitype(::Type{<:FooBarRegressor}) = AbstractVector{Continuous}
