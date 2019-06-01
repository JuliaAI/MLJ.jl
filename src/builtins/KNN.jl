# this file defines *and* loads one module

module KNN

export KNNRegressor

import MLJBase

using LinearAlgebra

# to be extended:


# TODO: introduce type parameters for the function fields (metric, kernel)

mutable struct KNNRegressor <: MLJBase.Deterministic
    K::Int           # number of local target values averaged
    metric
    kernel
end

euclidean(v1, v2) = norm(v2 - v1)
reciprocal(d) = d < eps(Float64) ? sign(d)/eps(Float64) : 1/d

# lazy keywork constructor:
function KNNRegressor(; K=2,
                      metric=euclidean,
                      kernel=reciprocal)
    model = KNNRegressor(K, metric, kernel)
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

function MLJBase.fit(model::KNNRegressor
                     , verbosity::Int
                     , X
                     , y)

    Xmatrix = MLJBase.matrix(X)
    
    # computing norms of rows later on is faster if we use the transpose of Xmatrix:
    fitresult = (Xmatrix', y)
    cache = nothing
    report = NamedTuple()
    
    return fitresult, cache, report 
end

first_component_is_less_than(v, w) = isless(v[1], w[1])

# TODO: there is way smarter way to do without sorting. Alternatively,
# let's get a KNN solver from external package: MLJ issue #87
function distances_and_indices_of_closest(K, metric, Xtrain, pattern)

    distance_index_pairs = 
        [(metric(Xtrain[:,j], pattern), j) for j in 1:size(Xtrain, 2)]

    sort!(distance_index_pairs, lt=first_component_is_less_than)
    Kprime = min(K,size(Xtrain, 2)) # in case less patterns than K

    distances = [distance_index_pairs[j][1] for j in 1:Kprime]
    indices = [distance_index_pairs[j][2] for j in 1:Kprime]

    return distances, indices    
    
end

function predict_on_pattern(model::KNNRegressor, fitresult, pattern)
    Xtrain, ytrain = fitresult[1], fitresult[2]
    distances, indices =
        distances_and_indices_of_closest(model.K, model.metric, Xtrain, pattern)
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



