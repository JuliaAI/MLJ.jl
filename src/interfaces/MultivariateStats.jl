module MultivariateStats_

export PCA

import MLJBase
import MultivariateStats

const MS = MultivariateStats

PCAFitResultType = MS.PCA

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

    # NOTE: again a collect(transpose) this is really dumb and wasteful
    fitresult = MS.fit(MS.PCA, collect(transpose(Xarray))
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
    # X is n x d, need to transpose and collect twice...
    # NOTE: again.. collect and transpose...
    return collect(transpose(
               MS.transform(fitresult, collect(transpose(Xarray)))))
end

####
#### METADATA
####

MLJBase.package_name(::Type{PCA})  = "MultivariateStats"
MLJBase.package_uuid(::Type{PCA})  = "6f286f6a-111f-5878-ab1e-185364afe411"
MLJBase.is_pure_julia(::Type{PCA}) = :yes
MLJBase.inputs_can_be(::Type{PCA}) = [:numeric,]

end # module

using .MultivariateStats_
export PCA
