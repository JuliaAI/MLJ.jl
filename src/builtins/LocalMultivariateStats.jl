# this file defines *and* loads one module

module LocalMultivariateStats

export RidgeRegressor, PCA

import MLJBase
import MLJ
import MultivariateStats

const MS = MultivariateStats

struct LinearFitresult{F} <: MLJBase.MLJType
    coefficients::Vector{F}
    bias::F
end

# Following helper function returns a named tuple with with three elements:
#
#        name | description
# :-----------|:-------------------------------------------------
# `:index`    | indices of features used to train `fitresult`
# `:feature`  | corresponding feature labels provided by `features`
# `:coef`     | coefficients for that feature in the fitresult
#
# The rows are ordered by the absolute value of the coefficients.
function coef_info(fitresult::LinearFitresult{F}, features) where F
    coef_given_index = Dict{Int,F}()
    abs_coef_given_index = Dict{Int,F}()
    v = fitresult.coefficients
    for k in eachindex(v)
        coef_given_index[k] = v[k]
        abs_coef_given_index[k] = abs(v[k])
    end
    index = reverse(MLJ.keys_ordered_by_values(abs_coef_given_index))
    feature = [features[i] for i in index]
    coef = [coef_given_index[i] for i in index]
    return (index=index, feature=feature, coef=coef)
end

####
#### RIDGE
####

mutable struct RidgeRegressor{F<:AbstractFloat} <: MLJBase.Deterministic{LinearFitresult{F}}
    target_type::Type{F}
    lambda::Float64
end

function MLJ.clean!(model::RidgeRegressor)
    warning = ""
    if model.lambda < 0
        warning *= "Need lambda ≥ 0. Resetting lambda=0. "
        model.lambda = 0
    end
    return warning
end

# keyword constructor
function RidgeRegressor(; target_type=Float64, lambda=0.0)

    model = RidgeRegressor(target_type, lambda)

    message = MLJBase.clean!(model)
    isempty(message) || @warn message

    return model
    
end

function MLJBase.fit(model::RidgeRegressor{F},
                     verbosity::Int,
                     X,
                     y::Vector{F2}) where {F,F2}

    F == F2 || error("Model specifies target_type=$F but target type is $F2.")

    Xmatrix = MLJBase.matrix(X)
    features = MLJBase.schema(X).names

    weights = MS.ridge(Xmatrix, y, model.lambda)

    coefficients = weights[1:end-1]
    bias = weights[end]

    fitresult = LinearFitresult(coefficients, bias)

    # report on the relative strength of each feature in the fitresult:
    cinfo = coef_info(fitresult, features) # a named tuple of vectors
    u = String[]
    v = Float64[]
    for i in 1:length(cinfo.feature)
        feature, coef = (cinfo.feature[i], cinfo.coef[i])
        coef = floor(1000*coef)/1000
        if coef < 0
            label = string(feature, " (-)")
        else
            label = string(feature, " (+)")
        end
        push!(u, label)
        push!(v, abs(coef))
    end
    report= (feature_importance_curve=(u, v),)
    cache = nothing

    return fitresult, cache, report

end

MLJBase.fitted_params(::RidgeRegressor, fitresult) =
    (coefs=fitresult.coefficients, bias=fitresult.bias)

function MLJBase.predict(model::RidgeRegressor, fitresult::LinearFitresult, Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    return Xmatrix*fitresult.coefficients .+ fitresult.bias
end

# metadata:
MLJBase.load_path(::Type{<:RidgeRegressor}) = "MLJ.RidgeRegressor"
MLJBase.package_name(::Type{<:RidgeRegressor}) = "MultivariateStats"
MLJBase.package_uuid(::Type{<:RidgeRegressor}) = "6f286f6a-111f-5878-ab1e-185364afe411"
MLJBase.package_url(::Type{<:RidgeRegressor})  = "https://github.com/JuliaStats/MultivariateStats.jl"
MLJBase.is_pure_julia(::Type{<:RidgeRegressor}) = true
MLJBase.input_scitypes(::Type{<:RidgeRegressor}) = MLJBase.Continuous
MLJBase.target_scitype(::Type{<:RidgeRegressor}) = MLJBase.Continuous

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
MLJBase.input_scitypes(::Type{<:PCA}) = MLJBase.Continuous
MLJBase.output_scitypes(::Type{<:PCA}) = MLJBase.Continuous


end # of module


## EXPOSE THE INTERFACE

using .LocalMultivariateStats

#using .MultivariateStats_
#export RidgeRegressor
