module GLM_

import MLJBase
import MLJ

export OLSRegressor, OLS, LinearRegression

import GLM

const OLSFitresult = GLM.LinearModel

OLSFitResult(coefs::Vector, b=nothing) = OLSFitResult(coefs, b)

####
#### OLSRegressor
####

mutable struct OLSRegressor <: MLJBase.Deterministic{OLSFitresult}
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
#### Data preparation
####

MLJBase.coerce(model::OLSRegressor, Xtable) = (MLJBase.matrix(Xtable), MLJ.retrieve(Xtable, MLJ.Schema).names)

function MLJBase.getrows(model::OLSRegressor, X, r)
    matrix, col_names = X
    return (matrix[r,:], col_names)
end

####
#### fit/predict OLSRegressor
####

function MLJBase.fit(model::OLSRegressor, verbosity::Int, Xplus, y::Vector)
    X, features = Xplus

    if model.fit_intercept
        fitresult = GLM.lm(hcat(X, ones(eltype(X), size(X, 1), 1)), y)
    else
        fitresult = GLM.lm(X, y)
    end

    coefs = GLM.coef(fitresult)

    report = Dict(:coef => coefs[1:end-Int(model.fit_intercept)]
                , :intercept => ifelse(model.fit_intercept, coefs[end], nothing)
                , :deviance => GLM.deviance(fitresult)
                , :dof_residual => GLM.dof_residual(fitresult)
                , :stderror => GLM.stderror(fitresult)
                , :vcov => GLM.vcov(fitresult))
    cache = nothing

    return fitresult, cache, report
end

function MLJBase.predict(model::OLSRegressor, fitresult::OLSFitresult, Xnew)
    X, features = Xnew
    model.fit_intercept && (X = hcat(X, ones(eltype(X), size(X, 1), 1)))
    return GLM.predict(fitresult, X)
end

# metadata:
MLJBase.package_name(::Type{OLS}) = "GLM"
MLJBase.package_uuid(::Type{OLS}) = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
MLJBase.is_pure_julia(::Type{OLS}) = :yes
MLJBase.inputs_can_be(::Type{OLS}) = [:numeric, ]
MLJBase.target_kind(::Type{OLS}) = :numeric
MLJBase.target_quantity(::Type{OLS}) = :univariate

end # module

using .GLM_
export OLSRegressor, OLS, LinearRegression
