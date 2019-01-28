# NOTE: there's a `kmeans!` function that updates centers, maybe a candidate
# for the `update` machinery. Same for `kmedoids!`
# NOTE: if the prediction is done on the original array, just the assignment
# should be returned, unclear what's the best way of doing this.

module Clustering_

export KMeans
export KMedoids

import MLJBase

import Clustering

using Distances
using LinearAlgebra: norm

const C = Clustering

# ----------------------------------

const KMeansFitResultType = C.KmeansResult
const KMedoidsFitResultType = Matrix

mutable struct KMeans <: MLJBase.Unsupervised
    k::Int
end

mutable struct KMedoids{M<:SemiMetric} <: MLJBase.Unsupervised
    k::Int
    metric::M
end

function MLJBase.clean!(model::Union{KMeans, KMedoids})
    warning = ""
    if model.k < 1
        warning *= "Need k > 1. Resetting k=1.\n"
        model.k = 1
    end
    return warning
end

####
#### KMEANS: constructor, fit, transform and predict
####

function KMeans(; k=3)
    model = KMeans(k)
    message = MLJBase.clean!(model)
    isempty(message) || @warn message

    return model
end

function MLJBase.fit(model::KMeans
                   , verbosity::Int
                   , X)

    Xarray = MLJBase.matrix(X)

    # NOTE see https://github.com/JuliaStats/Clustering.jl/issues/136
    # with respect to the copy/transpose
    fitresult = C.kmeans(copy(transpose(Xarray)), model.k)

    cache = nothing
    report = Dict(:centers => transpose(fitresult.centers) # size k x p
                , :assignments => fitresult.assignments # size n
                  )

    return fitresult, cache, report
end

function MLJBase.transform(model::KMeans
                         , fitresult::KMeansFitResultType
                         , X)

    Xarray = MLJBase.matrix(X)
    # X is n × d
    # centers is d × k
    # results is n × k
    (n, d), k = size(X), model.k
    X̃ = zeros(size(X, 1), k)

    @inbounds for i ∈ 1:n
        @inbounds for j ∈ 1:k
            X̃[i, j] = norm(view(Xarray, i, :) .- view(fitresult.centers, :, j))
        end
    end
    return X̃
end

# For finding the minimum the squared norm is enough (and faster)
_norm2(x) = sum(e->e^2, x)

function MLJBase.predict(model::KMeans
                       , fitresult::KMeansFitResultType
                       , Xnew)

    Xarray = MLJBase.matrix(Xnew)
    # similar to transform except we only care about the min distance
    (n, d), k = size(Xarray), model.k
    pred = zeros(Int, n)
    @inbounds for i ∈ 1:n
        minv = Inf
        for j ∈ 1:k
            curv = _norm2(view(Xarray, i, :) .- view(fitresult.centers, :, j))
            # avoid branching (this is twice as fast as argmin because
            # the context is simpler and we have to do fewer checks)
            P       = curv < minv
            pred[i] =    j * P + pred[i] * !P # if P is true --> j
            minv    = curv * P +    minv * !P # if P is true --> curvalue
        end
    end
    return pred
end

####
#### KMEDOIDS: constructor, fit and predict
#### NOTE there is no transform in the sense of kmeans
####

function KMedoids(; k=3, metric=SqEuclidean())
    model = KMedoids(k, metric)
    message = MLJBase.clean!(model)
    isempty(message) || @warn message

    return model
end

function MLJBase.fit(model::KMedoids
                   , verbosity::Int
                   , X)

    Xarray = MLJBase.matrix(X)
    Carray = pairwise(model.metric, transpose(Xarray)) # n x n

    result = C.kmedoids(Carray, model.k)

    medoids = Xarray[result.medoids, :] # size k x p

    cache = nothing
    report = Dict(:fit_result => result
                , :medoids => medoids # size k x p
                , :assignments => result.assignments # size n
                  )

    # keep track of the actual medoids ("center") in order to predict
    fitresult = medoids

    return fitresult, cache, report
end

function MLJBase.transform(model::KMedoids
                         , fitresult::KMedoidsFitResultType
                         , X)

    Xarray = MLJBase.matrix(X)
    medoids = fitresult
    metric = model.metric
    # X is n × d
    # centers is d × k
    # results is n × k
    (n, d), k = size(X), model.k
    X̃ = zeros(size(X, 1), k)

    @inbounds for i ∈ 1:n
        @inbounds for j ∈ 1:k
            X̃[i, j] = evaluate(metric, view(Xarray, i, :), view(medoids, j, :))
        end
    end
    return X̃
end

function MLJBase.predict(model::KMedoids
                       , fitresult::KMedoidsFitResultType
                       , Xnew)

    Xarray = MLJBase.matrix(Xnew)
    medoids = fitresult

    # similar to kmeans except instead of centers we use medoids
    # kth medoid corresponds to Xarray[medoids[k], :]
    metric = model.metric
    (n, d), k = size(Xarray), model.k
    pred = zeros(Int, n)

    @inbounds for i ∈ 1:n
        minv = Inf
        for j ∈ 1:k
            curv = evaluate(metric, view(Xarray, i, :), view(medoids, j, :))
            P       = curv < minv
            pred[i] =    j * P + pred[i] * !P # if P is true --> j
            minv    = curv * P +    minv * !P # if P is true --> curvalue
        end
    end
    return pred
end

####
#### METADATA
####

MLJBase.package_name(::Type{KMeans}) = "Clustering"
MLJBase.package_uuid(::Type{KMeans}) = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
MLJBase.is_pure_julia(::Type{KMeans}) = :yes
MLJBase.inputs_can_be(::Type{KMeans}) = [:numeric,]
MLJBase.target_kind(::Type{KMeans}) = :multiclass
MLJBase.target_quantity(::Type{KMeans}) = :univariate

MLJBase.package_name(::Type{KMedoids}) = "Clustering"
MLJBase.package_uuid(::Type{KMedoids}) = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
MLJBase.is_pure_julia(::Type{KMedoids}) = :yes
MLJBase.inputs_can_be(::Type{KMedoids}) = [:numeric,]
MLJBase.target_kind(::Type{KMedoids}) = :multiclass
MLJBase.target_quantity(::Type{KMedoids}) = :univariate

end # module

using .Clustering_
export KMeans
export KMedoids
