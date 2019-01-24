module MultivariateStats_

export PCA

import MLJBase

import MultivariateStats

const MS = MultivariateStats

PCAFitResultType = MS.PCA

mutable struct PCA <: MLJBase.Unsupervised
    ncomp::Union{Nothing, Int}
    solver::Symbol
    pratio::Float64 # ratio of variances preserved in the principal subspace
    mean::Union{Nothing, Int, Vector{Float64}}
end

function PCA(;
             ncomp=nothing
           , solver=:auto
           , pratio=0.99
           , mean=nothing)
    model = PCA(ncomp, solver)
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
    if model.solver ∉ [:auto, :cov, :svd]
        warning *= "Unknown solver specification. Resetting to solver=:auto.\n"
        model.solver = :auto
    end
    if !(0.0 < model.pratio < 1.0)
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

    fitresult = MS.fit(model, transpose(Xarray)
                     ; method=model.solver
                     , pratio=model.pratio
                     , maxoutdim=model.ncomp)
    cache = nothing
    report = nothing

    return fitresult, cache, report
end

end # module

using .MultivariateStats_
export PCA
