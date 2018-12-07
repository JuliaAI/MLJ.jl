#abstract type MLJType end


## NESTED PARAMATER INTERFACE

struct Params 
    pairs::Tuple{Vararg{Pair{Symbol}}}
    Params(args::Pair...) = new(args)
end

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

get_params(field) = field
function get_params(model::M) where M<:Model
    pairs = Pair{Symbol,Any}[]
    for field in fieldnames(M)
        value = getfield(model, field)
        push!(pairs, field => get_params(value))
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

abstract type ParamRange <: MLJType end

struct NominalRange{T} <: ParamRange
    values::Tuple{Vararg{T}}
end

# for dispatching on scales (eg, transform(Val(:log10), 100) = 2):
MLJ.transform(::Val{:identity}, x) = x
MLJ.inverse_transform(::Val{:identity}, x) = x
MLJ.transform(::Val{:log}, x) = log(x)
MLJ.inverse_transform(::Val{:log}, x) = exp(x)
MLJ.transform(::Val{:log10}, x) = log10(x) 
MLJ.inverse_transform(::Val{:log10}, x) = 10^x
MLJ.transform(::Val{:log2}, x) = log2(x) 
MLJ.inverse_transform(::Val{:log2}, x) = 2^x

struct NumericRange{T} <: ParamRange
    lower::T
    upper::T
    scale::Symbol
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

function ParamRange(model, field::Symbol; values=nothing,
                    lower=nothing, upper=nothing, scale=:identity)
    T = get_type(typeof(model), field)
    if T <: Real
        lower !=nothing && upper != nothing ||
            error("You must specify lower=... and upper=... .")
        return NumericRange{T}(lower, upper, scale)
    else
        values !=nothing || error("You must specify values=... .")
        return NominalRange{T}(Tuple(values))
    end
end


## ITERATORS FROM PARAMRANGES

iterator(param_range::NominalRange) = collect(param_range.values)

function iterator(param_range::NumericRange{<:Real}, n::Int)
    scale = Val(param_range.scale) 
    transformed = range(transform(scale, param_range.lower),
                stop=transform(scale, param_range.upper),
                length=n)
    inverse_transformed = map(transformed) do value
        inverse_transform(scale, value)
    end
    return unique(inverse_transformed)
end

# in special case of integers, round to nearest integer:
function iterator(param_range::NumericRange{I}, n::Int) where I<:Integer
    scale = Val(param_range.scale) 
    transformed = range(transform(scale, param_range.lower),
                stop=transform(scale, param_range.upper),
                length=n)
    inverse_transformed =  map(transformed) do value
        round(I, inverse_transform(scale, value))
    end
    return unique(inverse_transformed)
end

    
## GRID GENERATION

# credit: the idea to use `repeat` to generate the grid comes from an earlier
# tuning algorithm of Diego Arenas

"""
    unwind(iterators...)

Represent all possible combinations of values generated by `iterators`
as rows of a matrix `A`. In more detail, `A` has one column for each
iterator in `iterators` and one row for each distinct possible
combination of values taken on by the iterators. Elements in the first
colmun cyle fastest, those in the last clolumn slowest. 

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

# TODO: Make this a sequential iterator and avoid copying `model`
# more than once.

"""
    iterator(model::Model, param_space_its::Params)

Object iterating over all models of type `typeof(model)` defined by
`param_space_its`.  

Each `name` in the nested `:name => value` pairs of `param_space_its`
should be the name of a nested field of `model`; and each element of
`flat_values(param_space_its)` (which consists of `value`s) should be
an iterator over values of some nested field of `model`.

"""
function iterator(model::M, param_space_its::Params) where M<:Model
    iterators = flat_values(param_space_its)
    A = unwind(iterators...)

    # initialize iterator (vector) to be returned:
    L = size(A, 1)
    it = Vector{M}(undef, L)

    for i in 1:L
        params = copy(param_space_its, Tuple(A[i,:]))
        clone = deepcopy(model)
        it[i] = set_params!(clone, params)
    end

    return it

end


