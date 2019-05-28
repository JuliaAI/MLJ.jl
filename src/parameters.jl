
## NESTED PARAMATER INTERFACE

function set_params!(model, key::Symbol, value)
    setfield!(model, key, value)
    return model
end

function set_params!(model, key::Symbol, params::NamedTuple)
    submodel = getfield(model, key)
    set_params!(submodel, params)
    return model
end


"""
   set_params!(model, nested_params)

Mutate the possibly nested fields of `model`, as returned by
`params(model)`, by specifying a named tuple `nested_params` matching
the pattern of `params(model)`.

    julia> rf = EnsembleModel(atom=DecisionTreeClassifier());
    julia> params(rf)
    (atom = (pruning_purity = 1.0,
             max_depth = -1,
             min_samples_leaf = 1,
             min_samples_split = 2,
             min_purity_increase = 0.0,
             n_subfeatures = 0.0,
             display_depth = 5,
             post_prune = false,
             merge_purity_threshold = 0.9,),
     weights = Float64[],
     bagging_fraction = 0.8,
     n = 100,
     parallel = true,
     out_of_bag_measure = Any[],)

    julia> set_params!(rf, (atom = (max_depth = 2,), n = 200));
    julia> params(rf)
    (atom = (pruning_purity = 1.0,
             max_depth = 2,
             min_samples_leaf = 1,
             min_samples_split = 2,
             min_purity_increase = 0.0,
             n_subfeatures = 0.0,
             display_depth = 5,
             post_prune = false,
             merge_purity_threshold = 0.9,),
     weights = Float64[],
     bagging_fraction = 0.8,
     n = 200,
     parallel = true,
     out_of_bag_measure = Any[],)

"""
function set_params!(model, params::NamedTuple)
    for k in keys(params)
        set_params!(model, k,  getproperty(params, k))
    end
    return model
end

function flat_length(params::NamedTuple)
    count = 0
    for k in keys(params)
        value = getproperty(params, k)
        if value isa NamedTuple
            count += flat_length(value)
        else
            count += 1
        end
    end
    return count
end

function flat_values(params::NamedTuple)
    values = []
    for k in keys(params)
        value = getproperty(params, k)
        if value isa NamedTuple
            append!(values, flat_values(value))
        else
            push!(values, value)
        end
    end
    return Tuple(values)
end

"""
     flat_keys(params::NamedTuple)

Use dot-concatentation to express each possibly nested key of `params`
in string form.

### Example

````
julia> flat_keys((A=(x=2, y=3), B=9)))
["A.x", "A.y", "B"]
````

"""
flat_keys(pair::Pair{Symbol}) = flat_keys(pair, last(pair))
flat_keys(pair, ::Any) = [string(first(pair)), ]
flat_keys(pair, ::NamedTuple) =
    [string(first(pair), ".", str) for str in flat_keys(last(pair))]
flat_keys(nested::NamedTuple) =
    reduce(vcat, [flat_keys(k => getproperty(nested, k)) for k in keys(nested)])


"""
    copy(params::NamedTuple, values=nothing)

Return a copy of `params` with new `values`. That is,
`flat_values(copy(params, values)) == values` is true, while the
nested keys remain unchanged.

If `values` is not specified a deep copy is returned. 

"""
function Base.copy(params::NamedTuple, values=nothing)

    values != nothing || return deepcopy(params)
    flat_length(params) == length(values) ||
        throw(DimensionMismatch("Length of NamedTuple object not matching number "*
                                "of supplied values"))

    kys = Symbol[]
    vals = Any[]
    pos = 1
    
    for oldky in keys(params)
        oldval = getproperty(params, oldky)
        if oldval isa NamedTuple
            L = flat_length(oldval)
            newval = copy(oldval, values[pos:(pos+L-1)])
            push!(kys, oldky)
            push!(vals, newval)
            pos += L
        else
            push!(kys, oldky)
            push!(vals, values[pos])
            pos += 1
        end
    end

    return NamedTuple{Tuple(kys)}(Tuple(vals))

end


## PARAMETER RANGES


#     Scale = SCALE()

# Object for dispatching on scales and functions when generating
# parameter ranges. We require different behaviour for scales and
# functions:

#      transform(Scale, scale(:log10), 100) = 2
#      inverse_transform(Scale, scale(:log10), 2) = 100

# but
#     transform(Scale, scale(log10), 100) = 100       # identity
#     inverse_transform(Scale, scale(log10), 100) = 2


# See also: strange

struct SCALE end
Scale = SCALE()
scale(s::Symbol) = Val(s)
scale(f::Function) = f
MLJ.transform(::SCALE, ::Val{:linear}, x) = x
MLJ.inverse_transform(::SCALE, ::Val{:linear}, x) = x
MLJ.transform(::SCALE, ::Val{:log}, x) = log(x)
MLJ.inverse_transform(::SCALE, ::Val{:log}, x) = exp(x)
MLJ.transform(::SCALE, ::Val{:log10}, x) = log10(x) 
MLJ.inverse_transform(::SCALE, ::Val{:log10}, x) = 10^x
MLJ.transform(::SCALE, ::Val{:log2}, x) = log2(x) 
MLJ.inverse_transform(::SCALE, ::Val{:log2}, x) = 2^x
MLJ.transform(::SCALE, f::Function, x) = x            # not a typo!
MLJ.inverse_transform(::SCALE, f::Function, x) = f(x) # not a typo!

abstract type ParamRange <: MLJType end

struct NominalRange{field,T} <: ParamRange
    values::Tuple{Vararg{T}}
end

struct NumericRange{field,T,D} <: ParamRange
    lower::T
    upper::T
    scale::D
end

function Base.show(stream::IO, object::ParamRange)
    id = objectid(object)
    field = typeof(object).parameters[1]
    description = string(typeof(object).name.name, "{$field}")
    str = "$description @ $(MLJBase.handle(object))"
    if !isempty(fieldnames(typeof(object)))
        printstyled(IOContext(stream, :color=> MLJBase.SHOW_COLOR),
                    str, color=:blue)
    else
        print(stream, str)
    end
end

""" 
   get_type(T, field::Symbol)

Returns the type of the field `field` of `DataType` T. Not a
type-stable function.  

"""
function get_type(T, field::Symbol)
    position = findfirst(fieldnames(T)) do fld
        fld == field
    end
    position != nothing || error("Type $T does not have $field as a field.")
    return T.types[position]
end

"""
    r = range(model, :hyper; values=nothing)

    Defines a `NominalRange` object for a field `hyper` of `model`,
assuming the field is a not a subtype of `Real`. Note that `r` is not
directly iterable but `iterator(r)` iterates over `values`.

    r = range(model, :hyper; upper=nothing, lower=nothing, scale=:linear)

Defines a `NumericRange` object for a `Real` field `hyper` of `model`.
Note that `r` is not directly iteratable but `iterator(r, n)` iterates
over `n` values between `lower` and `upper` values, according to the
specified `scale`. The supported scales are `:linear, :log, :log10,
:log2`. Values for `Integer` types are rounded (with duplicate values
removed, resulting in possibly less than `n` values).

Alternatively, if a function `f` is provided as `scale`, then
`iterator(r, n)` iterates over the values `[f(x1), f(x2), ... ,
f(xn)]`, where `x1, x2, ..., xn` are linearly spaced between `lower`
and `upper`.


See also: iterator

"""
function Base.range(model, field::Symbol; values=nothing,
                    lower=nothing, upper=nothing, scale::D=:linear) where D
    T = get_type(typeof(model), field)
    if T <: Real
        (lower === nothing || upper === nothing) &&
            error("You must specify lower=... and upper=... .")
        return NumericRange{field,T,D}(lower, upper, scale)
    else
        values === nothing && error("You must specify values=... .")
        return NominalRange{field,T}(Tuple(values))
    end
end

"""
    MLJ.scale(r::ParamRange)

Return the scale associated with the `ParamRange` object `r`. The
possible return values are: `:none` (for a `NominalRange`), `:linear`,
`:log`, `:log10`, `:log2`, or `:custom` (if `r.scale` is function).

"""
scale(r::NominalRange) = :none
scale(r::NumericRange) = :custom
scale(r::NumericRange{field,T,Symbol}) where {field,T} =
    r.scale


## ITERATORS FROM A PARAMETER RANGE

iterator(param_range::NominalRange) = collect(param_range.values)

function iterator(param_range::NumericRange{field,T}, n::Int) where {field,T<:Real}
    s = scale(param_range.scale) 
    transformed = range(transform(Scale, s, param_range.lower),
                stop=transform(Scale, s, param_range.upper),
                length=n)
    inverse_transformed = map(transformed) do value
        inverse_transform(Scale, s, value)
    end
    return unique(inverse_transformed)
end

# in special case of integers, round to nearest integer:
function iterator(param_range::NumericRange{field, I}, n::Int) where {field,I<:Integer}
    s = scale(param_range.scale) 
    transformed = range(transform(Scale, s, param_range.lower),
                stop=transform(Scale, s, param_range.upper),
                length=n)
    inverse_transformed =  map(transformed) do value
        round(I, inverse_transform(Scale, s, value))
    end
    return unique(inverse_transformed)
end

    
## GRID GENERATION

"""
    unwind(iterators...)

Represent all possible combinations of values generated by `iterators`
as rows of a matrix `A`. In more detail, `A` has one column for each
iterator in `iterators` and one row for each distinct possible
combination of values taken on by the iterators. Elements in the first
column cycle fastest, those in the last clolumn slowest. 

### Example

````julia
julia> iterators = ([1, 2], ["a","b"], ["x", "y", "z"]);
julia> MLJ.unwind(iterators...)
12×3 Array{Any,2}:
 1  "a"  "x"
 2  "a"  "x"
 1  "b"  "x"
 2  "b"  "x"
 1  "a"  "y"
 2  "a"  "y"
 1  "b"  "y"
 2  "b"  "y"
 1  "a"  "z"
 2  "a"  "z"
 1  "b"  "z"
 2  "b"  "z"
````

"""
function unwind(iterators...)
    n_iterators = length(iterators)
    iterator_lengths = map(length, iterators)

    # product of iterator lengths:
    L = reduce(*, iterator_lengths)
    L != 0 || error("Parameter iterator of length zero encountered.") 

    A = Array{Any}(undef, L, n_iterators)
    n_iterators != 0 || return A

    inner = 1
    outer = L
    for j in 1:n_iterators
        outer = outer ÷ iterator_lengths[j]
        A[:,j] = repeat(iterators[j], inner=inner, outer=outer)
        inner *= iterator_lengths[j]
    end
    return A
end

"""
    iterator(model::Model, param_iterators::NamedTuple)

Iterator over all models of type `typeof(model)` defined by
`param_iterators`.

Each `name` in the nested `:name => value` pairs of `param_iterators`
should be the name of a (possibly nested) field of `model`; and each
element of `flat_values(param_iterators)` (the corresponding final
values) is an iterator over values of one of those fields.

See also `iterator` and `params`.

"""
function iterator(model::M, param_iterators::NamedTuple) where M<:Model
    iterators = flat_values(param_iterators)
    A = unwind(iterators...)

    # initialize iterator (vector) to be returned:
    L = size(A, 1)
    it = Vector{M}(undef, L)

    for i in 1:L
        params = copy(param_iterators, Tuple(A[i,:]))
        clone = deepcopy(model)
        it[i] = set_params!(clone, params)
    end

    return it

end

