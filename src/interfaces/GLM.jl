module GLM_

import MLJBase
import MLJ

export OLSRegressor, OLS, LinearRegression

import GLM

const OLSFitResult = GLM.LinearModel

OLSFitResult(coefs::Vector, b=nothing) = OLSFitResult(coefs, b)

####
#### OLSRegressor
####

mutable struct OLSRegressor <: MLJBase.Deterministic{OLSFitResult}
    fit_intercept::Bool
#    allowrankdeficient::Bool
end

function OLSRegressor(;fit_intercept=true)
    return OLSRegressor(fit_intercept)
end

# synonyms
const OLS = OLSRegressor
const LinearRegression = OLSRegressor

####
#### fit/predict OLSRegressor
####

function MLJBase.fit(model::OLSRegressor, verbosity::Int, X, y::Vector)

    Xmatrix = MLJBase.matrix(X)
    features = MLJBase.schema(X).names
    
    if model.fit_intercept
        fitresult = GLM.lm(hcat(Xmatrix, ones(eltype(Xmatrix), size(Xmatrix, 1), 1)), y)
    else
        fitresult = GLM.lm(Xmatrix, y)
    end

    coefs = GLM.coef(fitresult)

    ## TODO: add feature importance curve to report using `features`
    report = Dict(:coef => coefs[1:end-Int(model.fit_intercept)]
                , :intercept => ifelse(model.fit_intercept, coefs[end], nothing)
                , :deviance => GLM.deviance(fitresult)
                , :dof_residual => GLM.dof_residual(fitresult)
                , :stderror => GLM.stderror(fitresult)
                , :vcov => GLM.vcov(fitresult))
    cache = nothing

    return fitresult, cache, report
end

function MLJBase.predict(model::OLSRegressor, fitresult::OLSFitResult, Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    model.fit_intercept && (Xmatrix = hcat(Xmatrix, ones(eltype(Xmatrix), size(Xmatrix, 1), 1)))
    return GLM.predict(fitresult, Xmatrix)
end

# metadata:
MLJBase.load_path(::Type{<:OLS}) = "MLJ.OLS"
MLJBase.package_name(::Type{<:OLS}) = "GLM"
MLJBase.package_uuid(::Type{<:OLS}) = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
MLJBase.package_url(::Type{<:OLS}) = "https://github.com/JuliaStats/GLM.jl"
MLJBase.is_pure_julia(::Type{<:OLS}) = :yes
MLJBase.input_kinds(::Type{<:OLS}) = [:continuous, ]
MLJBase.output_kind(::Type{<:OLS}) = :continuous
MLJBase.output_quantity(::Type{<:OLS}) = :univariate

end # module

using .GLM_
export OLSRegressor, OLS, LinearRegression
