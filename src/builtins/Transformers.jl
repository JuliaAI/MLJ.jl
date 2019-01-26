# note: this file defines *and* imports one module; see end

module Transformers

export FeatureSelector
export ToIntTransformer
export UnivariateStandardizer, Standardizer
export UnivariateBoxCoxTransformer

import MLJBase: MLJType, Unsupervised
import DataFrames: names, AbstractDataFrame, DataFrame, eltypes
import Distributions
using Statistics
# using Tables

import MLJBase: Rows, Cols, Schema, retrieve, table

# to be extended:
import MLJBase: fit, transform, inverse_transform


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
mutable struct FeatureSelector <: Unsupervised
    features::Vector{Symbol} 
end

FeatureSelector(;features=Symbol[]) = FeatureSelector(features)

function fit(transformer::FeatureSelector, verbosity::Int, X::AbstractDataFrame)
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


## FOR RELABELLING BY CONSECUTIVE INTEGERS
"""
    Relabel with consecutive integers
"""
mutable struct ToIntTransformer <: Unsupervised
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
             , v::AbstractVector{T}) where T

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

mutable struct UnivariateStandardizer <: Unsupervised
end

function fit(transformer::UnivariateStandardizer, verbosity::Int, v::AbstractVector{T}) where T<:Real
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


## STANDARDIZATION OF ORDINAL FEATURES OF TABULAR DATA

# TODO: reimplement in simpler, safer way: fitresult is two vectors:
# one of features that are transformed, one of corresponding
# univariate machines. Make data container agnostic.

"""
    Standardizer(; features=Symbol[])

Unsupervised model for standardizing (whitening) the columns of
tabular data. If `features` is empty then all columns of eltype
`AbstractFloat` will be standardized. For different behaviour, specify
the names of features to be standardized. Presently returns a
`DataFrame`.

    using DataFrames
    X = DataFrame(x1=[0.2, 0.3, 1.0], x2=[4, 2, 3])
    stand_model = Standardizer()
    transform(fit!(machine(stand_model, X)), X)

    3×2 DataFrame
    │ Row │ x1        │ x2    │
    │     │ Float64   │ Int64 │
    ├─────┼───────────┼───────┤
    │ 1   │ -0.688247 │ 4     │
    │ 2   │ -0.458831 │ 2     │
    │ 3   │ 1.14708   │ 3     │

    stand_model.features=[:x1, :x2]
    transform(fit!(machine(stand_model, X)), X)

    3×2 DataFrame
    │ Row │ x1        │ x2      │
    │     │ Float64   │ Float64 │
    ├─────┼───────────┼─────────┤
    │ 1   │ -0.688247 │ 1.0     │
    │ 2   │ -0.458831 │ -1.0    │
    │ 3   │ 1.14708   │ 0.0     │

"""
mutable struct Standardizer <: Unsupervised
    features::Vector{Symbol} # features to be standardized; empty means all of
end

# lazy keyword constructor:
Standardizer(; features=Symbol[]) = Standardizer(features)

# null fitresult:
StandardizerFitResult() = StandardizerFitResult(zeros(0,0), Symbol[], Bool[])

function fit(transformer::Standardizer, verbosity::Int, X::Any)
    # if using Query.jl, replace below code with
    # all_features = df |> @take(1) |> @map(fieldnames(typeof(_))) |> @mapmany(_, __)
    # Since this is a really dirty way of proceeding, I've used
    # Tables.jl for now.
    schema =  retrieve(X, Schema)
    all_features = schema.names
    
    # determine indices of all_features to be transformed
    if isempty(transformer.features)
        cols_to_fit = filter!(eachindex(all_features)|>collect) do j
            schema.eltypes[j] <: AbstractFloat
        end
    else
        cols_to_fit = filter!(eachindex(all_features)|>collect) do j
            all_features[j] in transformer.features && schema.eltypes[j] <: Real
        end
    end
    
    fitresult_given_feature = Dict{Symbol,Tuple{Float64,Float64}}()

    # fit each feature
    verbosity < 2 || @info "Features standarized: "
    for j in cols_to_fit
        col_fitresult, cache, report =
            fit(UnivariateStandardizer(), verbosity - 1, retrieve(X, Cols, j))
        fitresult_given_feature[all_features[j]] = col_fitresult
        verbosity < 2 ||
            @info "  :$(all_features[j])    mu=$(col_fitresult[1])  sigma=$(col_fitresult[2])"
    end
    
    fitresult = fitresult_given_feature
    cache = nothing
    report = Dict{Symbol,Any}()
    report[:features_fit]=keys(fitresult_given_feature)
    
    return fitresult, cache, report
    
end

function transform(transformer::Standardizer, fitresult, X)

    # `fitresult` is dict of column fitresults, keyed on feature names

    features_to_be_transformed = keys(fitresult)

    all_features = retrieve(X, Schema).names
    
    issubset(Set(features_to_be_transformed), Set(all_features)) ||
        error("Attempting to transform data with incompatible feature labels.")

    col_transformer = UnivariateStandardizer()

    cols = map(all_features) do ftr
        if ftr in features_to_be_transformed
            transform(col_transformer, fitresult[ftr], retrieve(X, Cols, ftr))
        else
            retrieve(X, Cols, ftr)
        end
    end

    named_cols = NamedTuple{all_features}(tuple(cols...))
        
    return table(named_cols, prototype=X)

end    

## UNIVARIATE BOX-COX TRANSFORMATIONS

function standardize(v)
    map(v) do x
        (x - mean(v))/std(v)
    end
end
                   
function midpoints(v::AbstractVector{T}) where T <: Real
    return [0.5*(v[i] + v[i + 1]) for i in 1:(length(v) -1)]
end

function normality(v)

    n  = length(v)
    v = standardize(convert(Vector{Float64}, v))

    # sort and replace with midpoints
    v = midpoints(sort!(v))

    # find the (approximate) expected value of the size (n-1)-ordered statistics for
    # standard normal:
    d = Distributions.Normal(0,1)
    w= map(collect(1:(n-1))/n) do x
        quantile(d, x)
    end

    return cor(v, w)

end

function boxcox(lambda, c, x::Real) 
    c + x >= 0 || throw(DomainError)
    if lambda == 0.0
        c + x > 0 || throw(DomainError)
        return log(c + x)
    end
    return ((c + x)^lambda - 1)/lambda
end

boxcox(lambda, c, v::AbstractVector{T}) where T <: Real =
    [boxcox(lambda, c, x) for x in v]    


"""
    UnivariateBoxCoxTransformer(; n=171, shift=false)

Construct a `Unsupervised` model specifying a univariate Box-Cox
transformation of a single variable taking non-negative values, with a
possible preliminary shift. Such a transformation is of the form

    x -> ((x + c)^λ - 1)/λ for λ not 0
    x -> log(x + c) for λ = 0

On fitting to data `n` different values of the Box-Cox
exponent λ (between `-0.4` and `3`) are searched to fix the value
maximizing normality. If `shift=true` and zero values are encountered
in the data then the transformation sought includes a preliminary
positive shift by `0.2` times the data mean. If there are no zero
values, then no shift is applied.

See also `BoxCoxEstimator` a transformer for selected ordinals in a
an iterable table.

"""
mutable struct UnivariateBoxCoxTransformer <: Unsupervised
    n::Int      # nbr values tried in optimizing exponent lambda
    shift::Bool # whether to shift data away from zero
end

# lazy keyword constructor:
UnivariateBoxCoxTransformer(; n=171, shift=false) = UnivariateBoxCoxTransformer(n, shift)

function fit(transformer::UnivariateBoxCoxTransformer, verbosity::Int, v::AbstractVector{T}) where T <: Real 

    m = minimum(v)
    m >= 0 || error("Cannot perform a Box-Cox transformation on negative data.")

    c = 0.0 # default
    if transformer.shift
        if m == 0
            c = 0.2*mean(v)
        end
    else
        m != 0 || error("Zero value encountered in data being Box-Cox transformed.\n"*
                        "Consider calling `fit!` with `shift=true`.")
    end
  
    lambdas = range(-0.4, stop=3, length=transformer.n)
    scores = Float64[normality(boxcox(l, c, v)) for l in lambdas]
    lambda = lambdas[argmax(scores)]

    return  (lambda, c), nothing, nothing

end

# for X scalar or vector:
transform(transformer::UnivariateBoxCoxTransformer, fitresult, X) =
    boxcox(fitresult..., X)

# scalar case:
function inverse_transform(transformer::UnivariateBoxCoxTransformer,
                           fitresult, x::Real)
    lambda, c = fitresult
    if lambda == 0
        return exp(x) - c
    else
        return (lambda*x + 1)^(1/lambda) - c
    end
end

# vector case:
function inverse_transform(transformer::UnivariateBoxCoxTransformer,
                           fitresult, w::AbstractVector{T}) where T <: Real
    return [inverse_transform(transformer, fitresult, y) for y in w]
end

end # end module


## EXPOSE THE INTERFACE

using .Transformers


