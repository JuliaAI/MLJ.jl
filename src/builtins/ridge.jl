import MLJBase
using LinearAlgebra

export SimpleRidgeRegressor

struct LinearFitresult{F} <: MLJBase.MLJType
    coefficients::Vector{F}
    bias::F
end

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
    return (coefficients=fitresult.coefficients, bias=fitresult.bias)
end

function MLJBase.fit(model::SimpleRidgeRegressor, verbosity::Int, X, y)
    x = MLJBase.matrix(X)
    x = hcat(ones(size(x, 1)), x)
    fitresult = (x'x - model.lambda*I)\(x'y)
    coefficients = fitresult[2:end]
    bias = fitresult[1]
    cache = nothing
    report = NamedTuple()
    return LinearFitresult(coefficients, bias), cache, report
end


function MLJBase.predict(model::SimpleRidgeRegressor, fitresult, Xnew)
    x = MLJBase.matrix(Xnew)
    return x*fitresult.coefficients .+ fitresult.bias
end

MLJBase.load_path(::Type{<:SimpleRidgeRegressor}) = "MLJ.SimpleRidgeRegressor"
MLJBase.package_name(::Type{<:SimpleRidgeRegressor}) = "MLJ"
MLJBase.package_uuid(::Type{<:SimpleRidgeRegressor}) = ""
MLJBase.is_pure_julia(::Type{<:SimpleRidgeRegressor}) = true
MLJBase.input_scitype_union(::Type{<:SimpleRidgeRegressor}) = MLJBase.Continuous
MLJBase.target_scitype_union(::Type{<:SimpleRidgeRegressor}) = MLJBase.Continuous