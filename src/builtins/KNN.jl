# this file defines *and* loads one module

module KNN

export KNNRegressor

import MLJBase

using LinearAlgebra

# to be extended:

KNNFitResultType{T} = Tuple{Matrix{T},Vector{T}}

# TODO: introduce type parameters for the function fields (metric, kernel)

mutable struct KNNRegressor{F} <: MLJBase.Deterministic{KNNFitResultType{F}}
    target_type::Type{F}
    K::Int           # number of local target values averaged
    metric::Function
    kernel::Function 
end

euclidean(v1, v2) = norm(v2 - v1)
reciprocal(d) = d < eps(Float64) ? sign(d)/eps(Float64) : 1/d

# lazy keywork constructor:
function KNNRegressor(; target_type=Float64,
                      K=1,
                      metric=euclidean,
                      kernel=reciprocal)
    model = KNNRegressor(target_type, K, metric, kernel)
    message = MLJBase.clean!(model)
    isempty(message) || @warn message
    return model
end
    
function MLJBase.clean!(model::KNNRegressor)
    message = ""
    if model.K <= 0
        model.K = 1
        message = message*"K cannot be negative. K set to 1."
    end
    return message
end

function MLJBase.fit(model::KNNRegressor{F}
                     , verbosity::Int
                     , X
                     , y::Vector{F2}) where {F,F2}

    F == F2 || error("Model specifies target_type=$F but target type is $F2.")

    Xmatrix = MLJBase.matrix(X)
    
    # computing norms of rows later on is faster if we use the transpose of Xmatrix:
    fitresult = (Xmatrix', y)
    cache = nothing
    report = NamedTuple()
    
    return fitresult, cache, report 
end

first_component_is_less_than(v, w) = isless(v[1], w[1])

# TODO: there is way smarter way to do without sorting. Alternatively,
# let's get a KNN solver from external package.
function distances_and_indices_of_closest(F, K, metric, Xtrain, pattern)

    distance_index_pairs = 
        [(metric(Xtrain[:,j], pattern), j) for j in 1:size(Xtrain, 2)]

    sort!(distance_index_pairs, lt=first_component_is_less_than)
    Kprime = min(K,size(Xtrain, 2)) # in case less patterns than K
    distances = Array{F}(undef, Kprime)
    indices = Array{Int}(undef, Kprime)
    for j in 1:Kprime
        distances[j] = distance_index_pairs[j][1]
        indices[j] = distance_index_pairs[j][2]
    end

    return distances, indices    
    
end

function predict_on_pattern(model::KNNRegressor{F}, fitresult, pattern) where F
    Xtrain, ytrain = fitresult[1], fitresult[2]
    distances, indices =
        distances_and_indices_of_closest(F, model.K, model.metric, Xtrain, pattern)
    wts = [model.kernel(d) for d in distances]
    wts = wts/sum(wts)
    return sum(wts .* ytrain[indices])
end

function MLJBase.predict(model::KNNRegressor, fitresult, Xnew)
    Xmatrix = MLJBase.matrix(Xnew)
    [predict_on_pattern(model, fitresult, Xmatrix[i,:]) for i in 1:size(Xmatrix,1)]
end
    
# metadata:
MLJBase.load_path(::Type{<:KNNRegressor}) = "MLJ.KNNRegressor"
MLJBase.package_name(::Type{<:KNNRegressor}) = "MLJ"
MLJBase.package_uuid(::Type{<:KNNRegressor}) = ""
MLJBase.is_pure_julia(::Type{<:KNNRegressor}) = true
MLJBase.input_scitype_union(::Type{<:KNNRegressor}) = MLJBase.Continuous
MLJBase.target_scitype_union(::Type{<:KNNRegressor}) = MLJBase.Continuous


end # module


## EXPOSE THE INTERFACE

using .KNN



