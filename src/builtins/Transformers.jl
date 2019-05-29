# note: this file defines *and* imports one module; see end

module Transformers

export FeatureSelector
# export ToIntTransformer
export UnivariateStandardizer, Standardizer
export UnivariateBoxCoxTransformer
export OneHotEncoder

import MLJBase: MLJType, Unsupervised
import MLJBase: schema, selectcols, table, scitype, scitype_union, scitypes
import MLJBase
import Distributions
using CategoricalArrays
using Statistics
using Tables

# to be extended:
import MLJBase: fit, transform, inverse_transform
import MLJBase: Found, Continuous, Multiclass
import MLJBase: OrderedFactor, Other, Finite, Infinite, Count


## CONSTANTS

const N_VALUES_THRESH = 16 # for BoxCoxTransformation
const CategoricalElement = Union{CategoricalValue,CategoricalString}

## FOR FEATURE (COLUMN) SELECTION

"""
    FeatureSelector(features=Symbol[])

An unsupervised model for filtering features (columns) of a table.
Only those features encountered during fitting will appear in
transformed tables if `features` is empty (the default).
Alternatively, if a non-empty `features` is specified, then only the
specified features are used. Throws an error if a recorded or
specified feature is not present in the transformation input.

"""
mutable struct FeatureSelector <: Unsupervised
    features::Vector{Symbol} 
end

FeatureSelector(;features=Symbol[]) = FeatureSelector(features)

function fit(transformer::FeatureSelector, verbosity::Int, X)
    namesX = MLJBase.schema(X).names
    issubset(Set(transformer.features), Set(namesX)) ||
        throw(error("Attempting to select non-existent feature(s)."))
    if isempty(transformer.features)
        fitresult = collect(namesX)
    else
        fitresult = transformer.features
    end
    report = NamedTuple()
    return fitresult, nothing, report
end

MLJBase.fitted_params(::FeatureSelector, fitresult) = (features_to_keep=fitresult,)

function transform(transformer::FeatureSelector, features, X)
    issubset(Set(features), Set(MLJBase.schema(X).names)) ||
        throw(error("Supplied frame does not admit previously selected features."))
    return MLJBase.selectcols(X, features)
end

# metadata:
MLJBase.load_path(::Type{<:FeatureSelector}) = "MLJ.FeatureSelector" 
MLJBase.package_url(::Type{<:FeatureSelector}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.package_name(::Type{<:FeatureSelector}) = "MLJ"
MLJBase.package_uuid(::Type{<:FeatureSelector}) = ""
MLJBase.is_pure_julia(::Type{<:FeatureSelector}) = true
MLJBase.input_scitype_union(::Type{<:FeatureSelector}) = Union{Missing,MLJBase.Found}
MLJBase.output_scitype_union(::Type{<:FeatureSelector}) = Union{Missing,MLJBase.Found}
MLJBase.output_is_multivariate(::Type{<:FeatureSelector}) = true



## UNIVARIATE STANDARDIZATION

"""
    UnivariateStandardizer()

Unsupervised model for standardizing (whitening) univariate data. 

"""
mutable struct UnivariateStandardizer <: Unsupervised
end

function fit(transformer::UnivariateStandardizer, verbosity::Int, v::AbstractVector{T}) where T<:Real
    std(v) > eps(Float64) || 
        @warn "Extremely small standard deviation encountered in standardization."
    fitresult = (mean(v), std(v))
    cache = nothing
    report = NamedTuple()
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

# metadata:
MLJBase.load_path(::Type{<:UnivariateStandardizer}) = "MLJ.UnivariateStandardizer" 
MLJBase.package_url(::Type{<:UnivariateStandardizer}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.package_name(::Type{<:UnivariateStandardizer}) = "MLJ"
MLJBase.package_uuid(::Type{<:UnivariateStandardizer}) = ""
MLJBase.is_pure_julia(::Type{<:UnivariateStandardizer}) = true
MLJBase.input_scitype_union(::Type{<:UnivariateStandardizer}) = Found
MLJBase.input_is_multivariate(::Type{<:UnivariateStandardizer}) = false
MLJBase.output_scitype_union(::Type{<:UnivariateStandardizer}) = Continuous
MLJBase.output_is_multivariate(::Type{<:UnivariateStandardizer}) = false


## STANDARDIZATION OF ORDINAL FEATURES OF TABULAR DATA

"""
    Standardizer(; features=Symbol[])

Unsupervised model for standardizing (whitening) the columns of
tabular data. If `features` is empty then all columns `v` for which
all elements have `Continuous` scitypes are standardized. For
different behaviour, specify the names of features to be standardized.

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

"""
mutable struct Standardizer <: Unsupervised
    features::Vector{Symbol} # features to be standardized; empty means all of
end

# lazy keyword constructor:
Standardizer(; features=Symbol[]) = Standardizer(features)

function fit(transformer::Standardizer, verbosity::Int, X::Any)

    _schema =  schema(X)
    all_features = _schema.names
    types = scitypes(X)
    
    # determine indices of all_features to be transformed
    if isempty(transformer.features)
        cols_to_fit = filter!(eachindex(all_features)|>collect) do j
            types[j] <: Continuous
        end
    else
        cols_to_fit = filter!(eachindex(all_features)|>collect) do j
            all_features[j] in transformer.features && types[j] <: Continuous
        end
    end
    
    fitresult_given_feature = Dict{Symbol,Tuple{Float64,Float64}}()

    # fit each feature
    verbosity < 2 || @info "Features standarized: "
    for j in cols_to_fit
        col_fitresult, cache, report =
            fit(UnivariateStandardizer(), verbosity - 1, selectcols(X, j))
        fitresult_given_feature[all_features[j]] = col_fitresult
        verbosity < 2 ||
            @info "  :$(all_features[j])    mu=$(col_fitresult[1])  sigma=$(col_fitresult[2])"
    end
    
    fitresult = fitresult_given_feature
    cache = nothing
    report = (features_fit=keys(fitresult_given_feature),)
    
    return fitresult, cache, report
    
end

MLJBase.fitted_params(::Standardizer, fitresult) = (mean_and_std_given_feature=fitresult,)

function transform(transformer::Standardizer, fitresult, X)

    # `fitresult` is dict of column fitresults, keyed on feature names

    features_to_be_transformed = keys(fitresult)

    all_features = schema(X).names
    
    issubset(Set(features_to_be_transformed), Set(all_features)) ||
        error("Attempting to transform data with incompatible feature labels.")

    col_transformer = UnivariateStandardizer()

    cols = map(all_features) do ftr
        if ftr in features_to_be_transformed
            transform(col_transformer, fitresult[ftr], selectcols(X, ftr))
        else
            selectcols(X, ftr)
        end
    end

    named_cols = NamedTuple{all_features}(tuple(cols...))
        
    return MLJBase.table(named_cols, prototype=X)

end    

# metadata:
MLJBase.load_path(::Type{<:Standardizer}) = "MLJ.Standardizer" 
MLJBase.package_url(::Type{<:Standardizer}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.package_name(::Type{<:Standardizer}) = "MLJ"
MLJBase.package_uuid(::Type{<:Standardizer}) = ""
MLJBase.is_pure_julia(::Type{<:Standardizer}) = true
MLJBase.input_scitype_union(::Type{<:Standardizer}) = Union{Found,Missing}
MLJBase.input_is_multivariate(::Type{<:Standardizer}) = true
MLJBase.output_scitype_union(::Type{<:Standardizer}) = Union{Found,Missing}
MLJBase.output_is_multivariate(::Type{<:Standardizer}) = true


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

Unsupervised model specifying a univariate Box-Cox
transformation of a single variable taking non-negative values, with a
possible preliminary shift. Such a transformation is of the form

    x -> ((x + c)^λ - 1)/λ for λ not 0
    x -> log(x + c) for λ = 0

On fitting to data `n` different values of the Box-Cox
exponent λ (between `-0.4` and `3`) are searched to fix the value
maximizing normality. If `shift=true` and zero values are encountered
in the data then the transformation sought includes a preliminary
positive shift `c` of `0.2` times the data mean. If there are no zero
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

    return  (lambda, c), nothing, NamedTuple()

end

fitted_params(::UnivariateBoxCoxTransformer, fitresult) =
    (λ=fitresult[1], c=fitresult[2])

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

# metadata:
MLJBase.load_path(::Type{<:UnivariateBoxCoxTransformer}) = "MLJ.UnivariateBoxCoxTransformer" 
MLJBase.package_url(::Type{<:UnivariateBoxCoxTransformer}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.package_name(::Type{<:UnivariateBoxCoxTransformer}) = "MLJ"
MLJBase.package_uuid(::Type{<:UnivariateBoxCoxTransformer}) = ""
MLJBase.is_pure_julia(::Type{<:UnivariateBoxCoxTransformer}) = true
MLJBase.input_scitype_union(::Type{<:UnivariateBoxCoxTransformer}) = MLJBase.Continuous
MLJBase.input_is_multivariate(::Type{<:UnivariateBoxCoxTransformer}) = false
MLJBase.output_scitype_union(::Type{<:UnivariateBoxCoxTransformer}) = MLJBase.Continuous
MLJBase.output_is_multivariate(::Type{<:UnivariateBoxCoxTransformer}) = false


## ONE HOT ENCODING

"""
    OneHotEncoder(; features=Symbol[], drop_last=false, ordered_factor=true)

Unsupervised model for one-hot encoding all features of `Finite`
scitype, within some table. If `ordered_factor=false` then
only `Multiclass` features are considered. The features encoded is
further restricted to those in `features`, when specified and
non-empty.

If `drop_last` is true, the column for the last level of each
categorical feature is dropped. New data to be transformed may lack
features present in the fit data, but no new features can be present.

*Warning:* This transformer assumes that the elements of a categorical
 feature in new data to be transformed point to the same
 CategoricalPool object encountered during the fit.

"""
mutable struct OneHotEncoder <: Unsupervised
    features::Vector{Symbol}
    drop_last::Bool
    ordered_factor::Bool
end

# lazy keyword constructor:
OneHotEncoder(; features=Symbol[], drop_last=false, ordered_factor=true) =
    OneHotEncoder(features, drop_last, ordered_factor)

# we store the categorical refs for each feature to be encoded and the
# corresponing feature labels generated (called
# "names"). `all_features` is stored to ensure no new features appear
# in new input data, causing potential name clashes.
struct OneHotEncoderResult <: MLJType
    all_features::Vector{Symbol} # all feature labels
    ref_name_pairs_given_feature::Dict{Symbol,Vector{Pair{<:Unsigned,Symbol}}}
end

# join feature and level into new label without clashing with anything
# in all_features:
function compound_label(all_features, feature, level)
    label = Symbol(string(feature, "__", level))
    # in the (rare) case subft is not a new feature label:
    while label in all_features
        label = Symbol(string(label,"_"))
    end
    return label
end

function fit(transformer::OneHotEncoder, verbosity::Int, X)

    all_features = Tables.schema(X).names # a tuple not vector
    specified_features =
        isempty(transformer.features) ? collect(all_features) : transformer.features

    ref_name_pairs_given_feature = Dict{Symbol,Vector{Pair{<:Unsigned,Symbol}}}()
    allowed_scitypes =
        transformer.ordered_factor == true ? Finite : Multiclass

    for j in eachindex(all_features)
        ftr = all_features[j]
        col = MLJBase.selectcols(X,j)
        T = scitype_union(col)
        if T <: allowed_scitypes && ftr in specified_features
            ref_name_pairs_given_feature[ftr] = Pair{<:Unsigned,Symbol}[]
            shift = transformer.drop_last ? 1 : 0
            levels = MLJBase.classes(first(col))
            if verbosity > 0
                @info "Spawning $(length(levels)-shift) sub-features "*
                "to one-hot encode feature :$ftr."
            end
            for level in levels[1:end-shift]
                ref = MLJBase.int(level)
                name = compound_label(all_features, ftr, level)
                push!(ref_name_pairs_given_feature[ftr], ref => name)
            end
        end
    end

    fitresult = OneHotEncoderResult(collect(all_features), ref_name_pairs_given_feature)
    report = (features_to_be_encoded=collect(keys(ref_name_pairs_given_feature)),)
    cache = nothing

    return fitresult, cache, report

end

# If v=categorical('a', 'a', 'b', 'a', 'c') and MLJBase.int(v[1]) = ref
# then `hot(v, ref) = [true, true, false, true, false]` 
hot(v::AbstractVector{<:CategoricalElement}, ref) = map(v) do c
    MLJBase.int(c) == ref
end

function transform(transformer::OneHotEncoder, fitresult, X)

    features = Tables.schema(X).names # tuple not vector
    d = fitresult.ref_name_pairs_given_feature
    
    issubset(Set(features), Set(fitresult.all_features)) ||
        error("Attempting to transform table with feature labels not seen in fit. ")

    new_features = Symbol[]
    new_cols = Vector[]
    features_to_be_transformed = keys(d)
    for ftr in features
        col = MLJBase.selectcols(X, ftr)
        if ftr in features_to_be_transformed
            append!(new_features, last.(d[ftr]))
            pairs = d[ftr]
            refs = first.(pairs)
            names = last.(pairs)
            cols_to_add = map(refs) do ref
                float.(hot(col, ref))
            end
            append!(new_cols, cols_to_add)
        else
            push!(new_features, ftr)
            push!(new_cols, col)
        end
    end

    named_cols = NamedTuple{tuple(new_features...)}(tuple(new_cols)...)

    return MLJBase.table(named_cols, prototype=X)

end

# metadata:
MLJBase.load_path(::Type{<:OneHotEncoder}) = "MLJ.OneHotEncoder" 
MLJBase.package_url(::Type{<:OneHotEncoder}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.package_name(::Type{<:OneHotEncoder}) = "MLJ"
MLJBase.package_uuid(::Type{<:OneHotEncoder}) = ""
MLJBase.is_pure_julia(::Type{<:OneHotEncoder}) = true
MLJBase.input_scitype_union(::Type{<:OneHotEncoder}) = Union{Missing,Found}
MLJBase.input_is_multivariate(::Type{<:OneHotEncoder}) = true
MLJBase.output_scitype_union(::Type{<:OneHotEncoder}) = Union{Missing,Found}
MLJBase.output_is_multivariate(::Type{<:OneHotEncoder}) = true


end # end module


## EXPOSE THE INTERFACE

using .Transformers


