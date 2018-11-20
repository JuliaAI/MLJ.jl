# note: this file defines *and* imports one module; see end
module Transformers

export FeatureSelector
export ToIntTransformer
export UnivariateStandardizer, Standardizer

import MLJ: MLJType, Transformer
import DataFrames: names, AbstractDataFrame, DataFrame, eltypes
import Distributions
using Statistics

# to be extended:
import MLJ: fit, transform, inverse_transform


## CONSTANTS

const N_VALUES_THRESH = 16 # for BoxCoxTransformation


## FOR FEATURE (COLUMN) SELECTION

"""
    FeatureSelector(features=Symbol[])

A transformer model for `DataFrame`s that returns a new `DataFrame`
with only the those features (columns) encountered during fitting the
transformer, and in the order encountered then.  Alternatively, if a
non-empty `features` is specified, then only the specified features
are used. Throws an error if a recorded or specified feature is not
present in the transformation input.

"""
mutable struct FeatureSelector <: Transformer
    features::Vector{Symbol} 
end

FeatureSelector(;features=Symbol[]) = FeatureSelector(features)

function fit(transformer::FeatureSelector, verbosity, rows, X::AbstractDataFrame)
    namesX = names(X)
    issubset(Set(transformer.features), Set(namesX)) ||
        throw(error("Attempting to select non-existent feature(s)."))
    if isempty(transformer.features)
        fitresult = namesX
    else
        fitresult = transformer.features
    end
    report = Dict{Symbol,Any}()
    report[:features_to_keep] = fitresult
    return fitresult, nothing, report
end

function transform(transformer::FeatureSelector, features, X)
    issubset(Set(features), Set(names(X))) ||
        throw(error("Supplied frame does not admit previously selected features."))
    return X[features]
end 


## FOR RELABELLING BY CONSECUTIVE INTEGERS STARTING AT 1

mutable struct ToIntTransformer <: Transformer
    sorted::Bool
    initial_label::Int # ususally 0 or 1
    map_unseen_to_minus_one::Bool # unseen inputs are transformed to -1
end

ToIntTransformer(; sorted=true, initial_label=1
                 , map_unseen_to_minus_one=false) =
                     ToIntTransformer(sorted, initial_label,
                                      map_unseen_to_minus_one)

struct ToIntFitResult{T} <: MLJType
    n_levels::Int
    int_given_T::Dict{T, Int}
    T_given_int::Dict{Int, T}
end

# null fitresult constructor:
ToIntFitResult(S::Type{T}) where T =
    ToIntFitResult{T}(0, Dict{T, Int}(), Dict{Int, T}())

function fit(transformer::ToIntTransformer
             , verbosity::Int
             , rows
             , v::AbstractVector{T}) where T

    v = v[rows]
    int_given_T = Dict{T, Int}()
    T_given_int = Dict{Int, T}()
    vals = collect(Set(v)) 
    if transformer.sorted
        sort!(vals)
    end
    n_levels = length(vals)
    if n_levels > 2^62 - 1
        error("Cannot encode with integers a vector "*
                         "having more than $(2^62 - 1) values.")
    end
    i = transformer.initial_label
    for c in vals
        int_given_T[c] = i
        T_given_int[i] = c
        i = i + 1
    end

    fitresult = ToIntFitResult{T}(n_levels, int_given_T, T_given_int)
    cache = nothing
    report = Dict{Symbol,Any}()
    report[:values] = vals

    return fitresult, cache, report

end

# scalar case:
function transform(transformer::ToIntTransformer, fitresult::ToIntFitResult{T}, x::T) where T
    ret = 0 # otherwise ret below stays in local scope
    try 
        ret = fitresult.int_given_T[x]
    catch exception
        if isa(exception, KeyError)
            if transformer.map_unseen_to_minus_one 
                ret = -1
            else
                throw(exception)
            end
        end 
    end
    return ret
end 
inverse_transform(transformer::ToIntTransformer, fitresult, y::Int) =
    fitresult.T_given_int[y]

# vector case:
function transform(transformer::ToIntTransformer, fitresult::ToIntFitResult{T},
                   v::AbstractVector{T}) where T
    return Int[transform(transformer, fitresult, x) for x in v]
end
inverse_transform(transformer::ToIntTransformer, fitresult::ToIntFitResult{T},
                  w::AbstractVector{Int}) where T = T[fitresult.T_given_int[y] for y in w]


## UNIVARIATE STANDARDIZATION

mutable struct UnivariateStandardizer <: Transformer
end

function fit(transformer::UnivariateStandardizer, verbosity, rows, v::AbstractVector{T}) where T<:Real
    v = v[rows]
    std(v) > eps(Float64) || 
        @warn "Extremely small standard deviation encountered in standardization."
    fitresult = (mean(v), std(v))
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

# for transforming single value:
function transform(transformer::UnivariateStandardizer, fitresult, x::Real)
    mu, sigma = fitresult
    return (x - mu)/sigma
end

# for transforming vector:
transform(transformer::UnivariateStandardizer, fitresult,
          v) =
              [transform(transformer, fitresult, x) for x in v]

# for single values:
function inverse_transform(transformer::UnivariateStandardizer, fitresult, y::Real)
    mu, sigma = fitresult
    return mu + y*sigma
end

# for vectors:
inverse_transform(transformer::UnivariateStandardizer, fitresult, w) =
    [inverse_transform(transformer, fitresult, y) for y in w]


## STANDARDIZATION OF ORDINAL FEATURES OF A DATAFRAME

# TODO: reimplement in simpler, safer way: fitresult is two vectors:
# one of features that are transformed, one of corresponding
# univariate trainable models. Make data container agnostic.

""" Standardizes the columns of eltype <: AbstractFloat unless non-empty `features` specfied."""
mutable struct Standardizer <: Transformer
    features::Vector{Symbol} # features to be standardized; empty means all of
end

# lazy keyword constructor:
Standardizer(; features=Symbol[]) = Standardizer(features)

struct StandardizerFitResult <: MLJType
    fitresults::Matrix{Float64}
    features::Vector{Symbol} # all the feature labels of the data frame fitted
    is_transformed::Vector{Bool}
end

# null fitresult:
StandardizerFitResult() = StandardizerFitResult(zeros(0,0), Symbol[], Bool[])

function fit(transformer::Standardizer, verbosity::Int, rows, X::DataFrame)

    X = X[rows,:]
    features = names(X)
    
    # determine indices of features to be transformed
    features_to_try = (isempty(transformer.features) ? features : transformer.features)
    is_transformed = Array{Bool}(undef, size(X, 2))
    for j in 1:size(X, 2)
        if features[j] in features_to_try && eltype(X[j]) <: AbstractFloat
            is_transformed[j] = true
        else
            is_transformed[j] = false
        end
    end

    # fit each of those features
    fitresults = Array{Float64}(undef, 2, size(X, 2))
    verbosity < 2 || @info "Features standarized: "
    for j in 1:size(X, 2)
        if is_transformed[j]
            fitresult, cache, report =
                fit(UnivariateStandardizer(), verbosity-1, :, collect(X[j]))
            fitresults[:,j] = [fitresult...]
            verbosity < 2 ||
                @info "  :$(features[j])    mu=$(fitresults[1,j])  sigma=$(fitresults[2,j])"
        else
            fitresults[:,j] = Float64[0.0, 1.0]
        end
    end
    
    fitresult = StandardizerFitResult(fitresults, features, is_transformed)
    cache = nothing
    report = Dict{Symbol,Any}()
    report[:features_transformed]=[features[is_transformed]]
    
    return fitresult, cache, report
    
end

function transform(transformer::Standardizer, fitresult, X)

    names(X) == fitresult.features ||
        error("Attempting to transform data frame with incompatible feature labels.")

    Xnew = X[1:end,:] # make a copy of X, working even for `SubDataFrames`
    univ_transformer = UnivariateStandardizer()
    for j in 1:size(X, 2)
        if fitresult.is_transformed[j]
            # extract the (mu, sigma) pair:
            univ_fitresult = (fitresult.fitresults[1,j], fitresult.fitresults[2,j])  
            Xnew[j] = transform(univ_transformer, univ_fitresult, collect(X[j]))
        end
    end
    return Xnew

end    

end # end module


## EXPOSE THE INTERFACE

using .Transformers


