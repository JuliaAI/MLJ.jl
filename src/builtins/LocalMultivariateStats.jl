# this file defines *and* loads one module

module LocalMultivariateStats

export RidgeRegressor, PCA

import MLJBase
import MultivariateStats

const MS = MultivariateStats

struct LinearFitresult{F} <: MLJBase.MLJType
    coefficients::Vector{F}
    bias::F
end

####
#### RIDGE
####

mutable struct RidgeRegressor <: MLJBase.Deterministic
    lambda::Float64
end

function MLJBase.clean!(model::RidgeRegressor)
    warning = ""
    if model.lambda < 0
        warning *= "Need lambda ≥ 0. Resetting lambda=0. "
        model.lambda = 0
    end
    return warning
end

# keyword constructor
function RidgeRegressor(; lambda=0.0)

    model = RidgeRegressor(lambda)

    message = MLJBase.clean!(model)
    isempty(message) || @warn message

    return model
    
end

function MLJBase.fit(model::RidgeRegressor,
                     verbosity::Int,
                     X,
                     y)

    Xmatrix = MLJBase.matrix(X)
    features = MLJBase.schema(X).names

    weights = MS.ridge(Xmatrix, y, model.lambda)

    coefficients = weights[1:end-1]
    bias = weights[end]

    fitresult = LinearFitresult(coefficients, bias)

    report= nothing
    cache = nothing

    return fitresult, cache, report

end

MLJBase.fitted_params(::RidgeRegressor, fitresult) =
    (coefficients=fitresult.coefficients, bias=fitresult.bias)

function MLJBase.predict(model::RidgeRegressor, fitresult, Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    return Xmatrix*fitresult.coefficients .+ fitresult.bias
end

# metadata:
MLJBase.load_path(::Type{<:RidgeRegressor}) = "MLJ.RidgeRegressor"
MLJBase.package_name(::Type{<:RidgeRegressor}) = "MultivariateStats"
MLJBase.package_uuid(::Type{<:RidgeRegressor}) = "6f286f6a-111f-5878-ab1e-185364afe411"
MLJBase.package_url(::Type{<:RidgeRegressor})  = "https://github.com/JuliaStats/MultivariateStats.jl"
MLJBase.is_pure_julia(::Type{<:RidgeRegressor}) = true
MLJBase.input_scitype_union(::Type{<:RidgeRegressor}) = MLJBase.Continuous
MLJBase.target_scitype_union(::Type{<:RidgeRegressor}) = MLJBase.Continuous

####
#### PCA
####

const PCAFitResultType = MS.PCA

mutable struct PCA <: MLJBase.Unsupervised
    ncomp::Union{Nothing, Int} # number of PCA components, all if nothing
    method::Symbol  # cov or svd (auto by default, choice based on dims)
    pratio::Float64 # ratio of variances preserved in the principal subspace
    mean::Union{Nothing, Real, Vector{Float64}} # 0 if pre-centered
end

function PCA(; ncomp=nothing
             , method=:auto
             , pratio=0.99
             , mean=nothing)

    model = PCA(ncomp, method, pratio, mean)
    message = MLJBase.clean!(model)
    isempty(message) || @warn message
    return model
end

function MLJBase.clean!(model::PCA)
    warning = ""
    if model.ncomp isa Int && model.ncomp < 1
        warning *= "Need ncomp > 1. Resetting ncomp=p.\n"
        model.ncomp = nothing
    end
    if model.method ∉ [:auto, :cov, :svd]
        warning *= "Unknown method specification. Resetting to method=:auto.\n"
        model.method = :auto
    end
    if !(0.0 < model.pratio <= 1.0)
        warning *= "Need 0 < pratio < 1. Resetting to pratio=0.99.\n"
        model.pratio = 0.99
    end
    if (model.mean isa Real) && !iszero(model.mean)
        warning *= "Need mean to be nothing, zero or a vector." *
                   " Resetting to mean=nothing.\n"
        model.mean = nothing
    end
    return warning
end

function MLJBase.fit(model::PCA
                   , verbosity::Int
                   , X)

    Xarray = MLJBase.matrix(X)
    mindim = minimum(size(Xarray))

    ncomp = (model.ncomp === nothing) ? mindim : model.ncomp

    # NOTE: copy/transpose
    fitresult = MS.fit(MS.PCA, permutedims(Xarray)
                     ; method=model.method
                     , pratio=model.pratio
                     , maxoutdim=ncomp
                     , mean=model.mean)

    cache = nothing
    report = (indim=MS.indim(fitresult),
              outdim=MS.outdim(fitresult),
              mean=MS.mean(fitresult),
              principalvars=MS.principalvars(fitresult),
              tprincipalvar=MS.tprincipalvar(fitresult),
              tresidualvar=MS.tresidualvar(fitresult),
              tvar=MS.tvar(fitresult))

    return fitresult, cache, report
end

MLJBase.fitted_params(::PCA, fitresult) = (projection=fitresult,)


function MLJBase.transform(model::PCA
                         , fitresult::PCAFitResultType
                         , X)

    Xarray = MLJBase.matrix(X)
    # X is n x d, need to transpose and copy twice...
    return MLJBase.table(
                permutedims(MS.transform(fitresult, permutedims(Xarray))),
                prototype=X)
end

####
#### METADATA
####

MLJBase.load_path(::Type{<:PCA})  = "MLJ.PCA"
MLJBase.package_name(::Type{<:PCA})  = MLJBase.package_name(RidgeRegressor)
MLJBase.package_uuid(::Type{<:PCA})  = MLJBase.package_uuid(RidgeRegressor)
MLJBase.package_url(::Type{<:PCA})  = MLJBase.package_url(RidgeRegressor)
MLJBase.is_pure_julia(::Type{<:PCA}) = true
MLJBase.input_scitype_union(::Type{<:PCA}) = MLJBase.Continuous
MLJBase.output_scitype_union(::Type{<:PCA}) = MLJBase.Continuous


end # of module


## EXPOSE THE INTERFACE

using .LocalMultivariateStats

#using .MultivariateStats_
#export RidgeRegressor
