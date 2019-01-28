# this file defines *and* loads one module

module LocalMultivariateStats

export RidgeRegressor, PCA

import MLJBase
import MLJ
import MultivariateStats
import DataFrames

const MS = MultivariateStats

struct LinearFitresult <: MLJBase.MLJType
    coefficients::Vector{Float64}
    bias::Float64
end

# Following helper function returns a `DataFrame` with three columns:
#
# column name | description
# :-----------|:-------------------------------------------------
# `:index`    | index of a feature used to train `fitresult`
# `:feature`  | corresponding feature label provided by `features`
# `:coef`     | coefficient for that feature in the fitresult
#
# The rows are ordered by the absolute value of the coefficients.
function coef_info(fitresult::LinearFitresult, features)
    coef_given_index = Dict{Int, Float64}()
    abs_coef_given_index = Dict{Int, Float64}()
    v = fitresult.coefficients
    for k in eachindex(v)
        coef_given_index[k] = v[k]
        abs_coef_given_index[k] = abs(v[k])
    end
    df = DataFrames.DataFrame()
    df[:index] = reverse(MLJ.keys_ordered_by_values(abs_coef_given_index))
    df[:feature] = map(df[:index]) do index
        features[index]
    end
    df[:coef] = map(df[:index]) do index
        coef_given_index[index]
    end
    return df
end

####
#### RIDGE
####

mutable struct RidgeRegressor <: MLJBase.Deterministic{LinearFitresult}
    lambda::Float64
end

# lazy keywork constructor
RidgeRegressor(; lambda=0.0) = RidgeRegressor(lambda)

MLJBase.coerce(model::RidgeRegressor, Xtable) = (MLJBase.matrix(Xtable), MLJ.schema(Xtable).names)
function MLJBase.getrows(model::RidgeRegressor, X, r)
    matrix, col_names = X
    return (matrix[r,:], col_names)
end

function MLJBase.fit(model::RidgeRegressor, verbosity::Int, Xplus, y::Vector{<:Real})

    X, features = Xplus

    weights = MS.ridge(X, y, model.lambda)

    coefficients = weights[1:end-1]
    bias = weights[end]

    fitresult = LinearFitresult(coefficients, bias)

    # report on the relative strength of each feature in the fitresult:
    report = Dict{Symbol, Any}()

    # temporary hack because fit doesn't know feature names:
    # features = [Symbol(string("_", j)) for j in 1:size(X, 2)]

    cinfo = coef_info(fitresult, features) # a DataFrame object
    u = String[]
    v = Float64[]
    for i in 1:size(cinfo, 1)
        feature, coef = (cinfo[i, :feature], cinfo[i, :coef])
        coef = floor(1000*coef)/1000
        if coef < 0
            label = string(feature, " (-)")
        else
            label = string(feature, " (+)")
        end
        push!(u, label)
        push!(v, abs(coef))
    end
    report[:feature_importance_curve] = (u, v)
    cache = nothing

    return fitresult, cache, report

end

function MLJBase.predict(model::RidgeRegressor, fitresult::LinearFitresult, Xnew)
    X, features = Xnew
    return X*fitresult.coefficients .+ fitresult.bias
end

# metadata:
MLJBase.package_name(::Type{<:RidgeRegressor}) = "MultivariateStats"
MLJBase.package_uuid(::Type{<:RidgeRegressor}) = "6f286f6a-111f-5878-ab1e-185364afe411"
MLJBase.is_pure_julia(::Type{<:RidgeRegressor}) = :yes
MLJBase.inputs_can_be(::Type{<:RidgeRegressor}) = [:numeric, ]
MLJBase.target_kind(::Type{<:RidgeRegressor}) = :numeric
MLJBase.target_quantity(::Type{<:RidgeRegressor}) = :univariate

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
    fitresult = MS.fit(MS.PCA, copy(transpose(Xarray))
                     ; method=model.method
                     , pratio=model.pratio
                     , maxoutdim=ncomp
                     , mean=model.mean)

    cache = nothing
    report = nothing

    return fitresult, cache, report
end

function MLJBase.transform(model::PCA
                         , fitresult::PCAFitResultType
                         , X)

    Xarray = MLJBase.matrix(X)
    # X is n x d, need to transpose and copy twice...
    return copy(transpose(MS.transform(fitresult, copy(transpose(Xarray)))))
end

####
#### METADATA
####

MLJBase.package_name(::Type{PCA})  = "MultivariateStats"
MLJBase.package_uuid(::Type{PCA})  = "6f286f6a-111f-5878-ab1e-185364afe411"
MLJBase.is_pure_julia(::Type{PCA}) = :yes
MLJBase.inputs_can_be(::Type{PCA}) = [:numeric,]

end # of module


## EXPOSE THE INTERFACE

using .LocalMultivariateStats

#using .MultivariateStats_
#export RidgeRegressor
