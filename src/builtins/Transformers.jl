# note: this file defines *and* imports one module; see end

module Transformers

export FeatureSelector
export ToIntTransformer
export UnivariateStandardizer, Standardizer
export UnivariateBoxCoxTransformer
export OneHotEncoder

import MLJBase: MLJType, Unsupervised
import MLJBase: schema, selectcols, table
import MLJBase
import DataFrames: names, AbstractDataFrame, DataFrame, eltypes
import Distributions
using CategoricalArrays
using Statistics
using Tables

# to be extended:
import MLJBase: fit, transform, inverse_transform


## CONSTANTS

const N_VALUES_THRESH = 16 # for BoxCoxTransformation


## FOR FEATURE (COLUMN) SELECTION

# TODO - make input data agnositic!!
"""
    FeatureSelector(features=Symbol[])

An usupervised model for filtering features (columns) of a table.
Only those features encountered during fitting will appear in
transformed tables, these features appearing encountered.
Alternatively, if a non-empty `features` is specified, then only the
specified features are used. Throws an error if a recorded or
specified feature is not present in the transformation input.

"""
mutable struct FeatureSelector <: Unsupervised
    features::Vector{Symbol} 
end

FeatureSelector(;features=Symbol[]) = FeatureSelector(features)

function fit(transformer::FeatureSelector, verbosity::Int, X)
    namesX = Tables.schema(X).names
    issubset(Set(transformer.features), Set(namesX)) ||
        throw(error("Attempting to select non-existent feature(s)."))
    if isempty(transformer.features)
        fitresult = collect(namesX)
    else
        fitresult = transformer.features
    end
    report = Dict{Symbol,Any}()
    report[:features_to_keep] = fitresult
    return fitresult, nothing, report
end

function transform(transformer::FeatureSelector, features, X)
    issubset(Set(features), Set(Tables.schema(X).names)) ||
        throw(error("Supplied frame does not admit previously selected features."))
    return MLJBase.selectcols(X, features)
end

# metadata:
MLJBase.load_path(::Type{<:FeatureSelector}) = "MLJ.FeatureSelector" 
MLJBase.package_url(::Type{<:FeatureSelector}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.package_name(::Type{<:FeatureSelector}) = "MLJ"
MLJBase.package_uuid(::Type{<:FeatureSelector}) = ""
MLJBase.is_pure_julia(::Type{<:FeatureSelector}) = :yes
MLJBase.input_kinds(::Type{<:FeatureSelector}) = [:continuous, :multiclass, :ordered_factor_finite, :ordered_factor_infinite, :missing]
MLJBase.output_kind(::Type{<:FeatureSelector}) = :mixed
MLJBase.output_quantity(::Type{<:FeatureSelector}) = :multivariate


## FOR RELABELLING BY CONSECUTIVE INTEGERS

# TODO: Fix so categorical levels are maintained, even "invisible"
# ones. Or depreciate in favour of (wrapped version of)
# MLJBase.CategoricalDecoder (but which does not respect order)?
"""
    ToIntTransformer

Univariate transformer for relabelling with consecutive integers. Does
not keep all levels of a `CategoricalVector` - only those manifest in
the fitting input.

See also: MLJBase.CategoricalDecoder
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

MLJBase.load_path(::Type{<:ToIntTransformer}) = "MLJ.ToIntTransformer" 
MLJBase.package_url(::Type{<:ToIntTransformer}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.package_name(::Type{<:ToIntTransformer}) = "MLJ"
MLJBase.package_uuid(::Type{<:ToIntTransformer}) = ""
MLJBase.is_pure_julia(::Type{<:ToIntTransformer}) = :yes
MLJBase.input_kinds(::Type{<:ToIntTransformer}) = [:multiclass, :ordered_factor_finite]
MLJBase.input_quantity(::Type{<:ToIntTransformer}) = :univariate
MLJBase.output_kind(::Type{<:ToIntTransformer}) = :multiclass
MLJBase.output_quantity(::Type{<:ToIntTransformer}) = :univariate


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

# metadata:
MLJBase.load_path(::Type{<:UnivariateStandardizer}) = "MLJ.UnivariateStandardizer" 
MLJBase.package_url(::Type{<:UnivariateStandardizer}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.package_name(::Type{<:UnivariateStandardizer}) = "MLJ"
MLJBase.package_uuid(::Type{<:UnivariateStandardizer}) = ""
MLJBase.is_pure_julia(::Type{<:UnivariateStandardizer}) = :yes
MLJBase.input_kinds(::Type{<:UnivariateStandardizer}) = [:multiclass, :ordered_factor_finite]
MLJBase.input_quantity(::Type{<:UnivariateStandardizer}) = :univariate
MLJBase.output_kind(::Type{<:UnivariateStandardizer}) = :multiclass
MLJBase.output_quantity(::Type{<:UnivariateStandardizer}) = :univariate


## STANDARDIZATION OF ORDINAL FEATURES OF TABULAR DATA

"""
    Standardizer(; features=Symbol[])

Unsupervised model for standardizing (whitening) the columns of
tabular data. If `features` is empty then all columns of eltype
`AbstractFloat` will be standardized. For different behaviour, specify
the names of features to be standardized. 

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
    _schema =  schema(X)
    all_features = _schema.names
    
    # determine indices of all_features to be transformed
    if isempty(transformer.features)
        cols_to_fit = filter!(eachindex(all_features)|>collect) do j
            _schema.eltypes[j] <: AbstractFloat
        end
    else
        cols_to_fit = filter!(eachindex(all_features)|>collect) do j
            all_features[j] in transformer.features && _schema.eltypes[j] <: Real
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
    report = Dict{Symbol,Any}()
    report[:features_fit]=keys(fitresult_given_feature)
    
    return fitresult, cache, report
    
end

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
        
    return table(named_cols, prototype=X)

end    

# metadata:
MLJBase.load_path(::Type{<:Standardizer}) = "MLJ.Standardizer" 
MLJBase.package_url(::Type{<:Standardizer}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.package_name(::Type{<:Standardizer}) = "MLJ"
MLJBase.package_uuid(::Type{<:Standardizer}) = ""
MLJBase.is_pure_julia(::Type{<:Standardizer}) = :yes
MLJBase.input_kinds(::Type{<:Standardizer}) = [:continuous, :multiclass, :ordered_factor_finite, :ordered_factor_infinite, :missing]
MLJBase.input_quantity(::Type{<:Standardizer}) = :multivariate
MLJBase.output_kind(::Type{<:Standardizer}) = :mixed
MLJBase.output_quantity(::Type{<:Standardizer}) = :multivariate


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

# metadata:
MLJBase.load_path(::Type{<:UnivariateBoxCoxTransformer}) = "MLJ.UnivariateBoxCoxTransformer" 
MLJBase.package_url(::Type{<:UnivariateBoxCoxTransformer}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.package_name(::Type{<:UnivariateBoxCoxTransformer}) = "MLJ"
MLJBase.package_uuid(::Type{<:UnivariateBoxCoxTransformer}) = ""
MLJBase.is_pure_julia(::Type{<:UnivariateBoxCoxTransformer}) = :yes
MLJBase.input_kinds(::Type{<:UnivariateBoxCoxTransformer}) = [:continuous,]
MLJBase.input_quantity(::Type{<:UnivariateBoxCoxTransformer}) = :univariate
MLJBase.output_kind(::Type{<:UnivariateBoxCoxTransformer}) = :continuous
MLJBase.output_quantity(::Type{<:UnivariateBoxCoxTransformer}) = :univariate


## ONE HOT ENCODING

"""
    OneHotEncoder(; features=Symbol[], drop_last=false, ref_type=UInt32)

Unsupervised model for one-hot encoding the categorical features of
tabular data. Here "categorical" refers to any feature whose eltype is
`CategoricalVector`. Only categorical features in `features` are
transformed when this is specified and non-empty. 

If `drop_last` is true, the column for the last level of each
categorical feature is dropped. New data to be transformed may lack
features present in the fit data, but no new features can be present.

All categorical features to be transformed must have a type promotable
to `ref_type`. Usually `ref_type=UInt32` suffices, but `ref_type=Int`
will always work. 

*Warning:* This transformer assumes that a categorical feature in new
 data to be transformed will have the same pool as that encountered in the fit.

"""
mutable struct OneHotEncoder{R<:Integer} <: Unsupervised
    features::Vector{Symbol}
    drop_last::Bool
    ref_type::Type{R}
end

# lazy keyword constructor:
OneHotEncoder(; features=Symbol[], drop_last::Bool=false, ref_type=UInt32) =
    OneHotEncoder(features, drop_last, ref_type)

# we store the categorical refs for each feature to be encoded and the
# corresponing feature labels generated (called
# "names"). `all_features` is stored to ensure no new features appear
# in new input data, causing potential name clashes.
struct OneHotEncoderResult{R} <: MLJType
    all_features::Vector{Symbol} # all feature labels
    ref_name_pairs_given_feature::Dict{Symbol,Vector{Pair{R,Symbol}}}
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

function fit(transformer::OneHotEncoder{R}, verbosity::Int, X) where R

    all_features = Tables.schema(X).names # a tuple not vector
    specified_features =
        isempty(transformer.features) ? collect(all_features) : transformer.features
    all_eltypes = Tables.schema(X).types  # a tuple not vector

    ref_name_pairs_given_feature = Dict{Symbol,Vector{Pair{R,Symbol}}}()

    for j in eachindex(all_features)
        ftr = all_features[j]
        T = all_eltypes[j]
        if T <: Union{CategoricalValue,CategoricalString} && ftr in specified_features
            ref_name_pairs_given_feature[ftr] = Pair{R,Symbol}[]
            col = MLJBase.selectcols(X,j)
            shift = transformer.drop_last ? 1 : 0
            if verbosity > 0
                @info "Spawned $(length(col)-shift) sub-features to one-hot encode feature :$ftr."
            end
            for level in levels(col)[1:end-shift]
                ref = get(col.pool, level)
                name = compound_label(all_features, ftr, level)
                push!(ref_name_pairs_given_feature[ftr], ref => name)
            end
        end
    end

    fitresult = OneHotEncoderResult(collect(all_features), ref_name_pairs_given_feature)
    report = Dict{Symbol,Any}()
    report[:features_to_be_encoded] = collect(keys(ref_name_pairs_given_feature))
    cache = nothing

    return fitresult, cache, report

end

# suppose 'a' gets reference `ref` in `v=categorical('a', 'a', 'b', 'a', 'c')`.
# then `hot(v, ref) = [true, true, false, true, false]` 
hot(v::CategoricalVector, ref) = map(v.refs) do r
    r == ref
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
                hot(col, ref)
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
MLJBase.is_pure_julia(::Type{<:OneHotEncoder}) = :yes
MLJBase.input_kinds(::Type{<:OneHotEncoder}) = [:continuous, :multiclass, :ordered_factor_finite, :ordered_factor_infinite, :missing]
MLJBase.input_quantity(::Type{<:OneHotEncoder}) = :multivariate
MLJBase.output_kind(::Type{<:OneHotEncoder}) = :mixed
MLJBase.output_quantity(::Type{<:OneHotEncoder}) = :multivariate


end # end module


## EXPOSE THE INTERFACE

using .Transformers


