## NESTED PARAMATER INTERFACE

# A `Params` object is a wrapped tuple of symbol-value pairs. In our
# application, the symbol is always the name of the field of some
# `Model` object. There are several possibilities for the values: (i)
# They are normal hyperparameter values or, in the case the field
# represents another `Model`, another `Params` object. So for us, a
# `Params` object represents nested hyper-parameter values in this
# case; (ii) The values are hyperparameter *ranges* (see below) or
# another `Params` object representing ranges of some sub-`Model`.
# object. So in this latter case we are representing nested
# hyperparameter ranges; (iii) As in (ii) but iterators instead of
# ranges.
struct Params
    pairs::Tuple{Vararg{Pair{Symbol}}}
    Params(args::Pair...) = new(args)
end

MLJBase.istoobig(::Params) = false # for display functionality

# since a "value" may be another Params object, we can build nested
# Param objects. The symbols are the names of paramet

Base.isempty(p::Params) = isempty(p.pairs)

==(params1::Params, params2::Params) = params1.pairs == params2.pairs

function Base.show(stream::IO, params::Params)
    print(stream, "Params(")
    count = 1 
    for pair in params.pairs
        show(stream, first(pair))
        print(stream, " => ")
        show(stream, last(pair))
        count == length(params.pairs) || print(stream, ", ")
        count += 1
    end
    print(stream, ")")
end

params(field) = field
function params(model::M) where M<:Model
    pairs = Pair{Symbol,Any}[]
    for field in fieldnames(M)
        value = getfield(model, field)
        push!(pairs, field => params(value))
    end
    return Params(pairs...)
end

function set_params!(model::M, pair::Pair) where M<:Model
    setfield!(model, first(pair), last(pair))
    return model
end

function set_params!(model::M, pair::Pair{Symbol, Params}) where M<:Model
    submodel = getfield(model, first(pair))
    set_params!(submodel, last(pair))
    return model
end

function set_params!(model::M, params::Params) where M<:Model
    for pair in params.pairs
        set_params!(model, pair)
    end
    return model
end

function Base.length(params::Params)
    count = 0
    for pair in params.pairs
        value = last(pair)
        if value isa Params
            count += length(value)
        else
            count += 1
        end
    end
    return count
end

function flat_values(params::Params)
    values = []
    for pair in params.pairs
        value = last(pair)
        if value isa Params
            append!(values, flat_values(value))
        else
            push!(values, value)
        end
    end
    return Tuple(values)
end

"""
    copy(params::Params, values=nothing)

Return a copy of `params` with new `values`. That is,
`flat_values(copy(params, values)) == values` is true, while
the first element of each nested pair (parameter name) is unchanged.

If `values` is not specified a deep copy is returned. 

"""
function Base.copy(params::Params, values=nothing)

    values != nothing || return deepcopy(params)
    length(params) == length(values) ||
        throw(DimensionMismatch("Length of Params object not matching number "*
                                "of supplied values"))

    pairs = []
    pos = 1
    
    for oldpair in params.pairs
        oldvalue = last(oldpair)
        if oldvalue isa Params
            L = length(oldvalue)
            newvalue = copy(oldvalue, values[pos:(pos+L-1)])
            push!(pairs, first(oldpair) => newvalue)
            pos += L
        else
            push!(pairs, first(oldpair) => values[pos])
            pos += 1
        end
    end

    return Params(pairs...)

end


## PARAMETER RANGES

""" 
    Scale = SCALE()

Object for dispatching on scales and functions when generating
parameter ranges. We require different behaviour for scales and
functions:

     transform(Scale, scale(:log), 100) = 2
     inverse_transform(Scale, scale(:log), 2) = 100

but
    transform(Scale, scale(log), 100) = 100       # identity
    inverse_transform(Scale, scale(log), 100) = 2 


See also: strange

"""
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

struct NominalRange{T} <: ParamRange
    values::Tuple{Vararg{T}}
end

struct NumericRange{T,D} <: ParamRange
    lower::T
    upper::T
    scale::D
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

Defines a `NominalRange` object for a field `hyper` of `model`. Note
that `r` is not directly iterable but `iterator(r)` iterates over
`values`.

    r = range(model, :hyper; upper=nothing, lower=nothing, scale=:linear)

Defines a `NumericRange` object for a field `hyper` of `model`.  Note
that `r` is not directly iteratable but `iterator(r, n)` iterates over
`n` values between `lower` and `upper` values, according to the
specified `scale`. The supported scales are `:linear, :log, :log10,
:log2`. Values for `Integer` types are rounded (with duplicate values
removed, resulting in possibly less than `n` values).

Alternatively, if a function `f` is provided as `scale`, then
`iterator(r, n)` iterates over the values `[f(x1), f(x2), ... ,
f(xn)]`, where `x1, x2, ..., xn` are linearly spaced between `lower`
and `upper`.

See also: strange, iterator

"""
function Base.range(model::MLJType, field::Symbol; values=nothing,
                    lower=nothing, upper=nothing, scale::D=:linear) where D
    T = get_type(typeof(model), field)
    if T <: Real
        lower !=nothing && upper != nothing ||
            error("You must specify lower=... and upper=... .")
        return NumericRange{T,D}(lower, upper, scale)
    else
        values !=nothing || error("You must specify values=... .")
        return NominalRange{T}(Tuple(values))
    end
end

"""
    strange(model, :hyper; kwargs...)

Returns the pair `:hyper => range(model, :hyper; kwargs...)`;
"strange" is short for "set to range".

See also: range

"""
strange(model::Model, field::Symbol; kwargs...) = field => range(model, field; kwargs...)


## ITERATORS FROM A PARAMETER RANGE

iterator(param_range::NominalRange) = collect(param_range.values)

function iterator(param_range::NumericRange{<:Real}, n::Int)
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
function iterator(param_range::NumericRange{I}, n::Int) where I<:Integer
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
    iterator(model::Model, param_iterators::Params)

Iterator over all models of type `typeof(model)` defined by
`param_iterators`.

Each `name` in the nested `:name => value` pairs of `param_iterators`
should be the name of a (possibly nested) field of `model`; and each
element of `flat_values(param_iterators)` (the corresponding final
values) is an iterator over values of one of those fields.

See also `iterator` and `param_range`.

"""
function iterator(model::M, param_iterators::Params) where M<:Model
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

