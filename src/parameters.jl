abstract type MLJType end


## NESTED PARAMATER INTERFACE

struct Params 
    pairs::Tuple{Vararg{Pair{Symbol}}}
    Params(args::Pair...) = new(args)
end

==(pairs1::Params, pairs2::Params) = pairs1.pairs == pairs2.pairs

function Base.show(stream::IO, pairs::Params)
    print(stream, "Params(")
    count = 1 
    for pair in pairs.pairs
        show(stream, first(pair))
        print(stream, " => ")
        show(stream, last(pair))
        count == length(pairs.pairs) || print(stream, ", ")
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

function set_params!(model::M, pairs::Params) where M<:Model
    for pair in pairs.pairs
        set_params!(model, pair)
    end
    return model
end

function Base.length(pairs::Params)
    count = 0
    for pair in pairs.pairs
        value = last(pair)
        if value isa Params
            count += length(value)
        else
            count += 1
        end
    end
    return count
end

function flat_values(pairs::Params)
    values = []
    for pair in pairs.pairs
        value = last(pair)
        if value isa Params
            append!(values, flat_values(value))
        else
            push!(values, value)
        end
    end
    return Tuple(values)
end

function Base.copy(tree::Params, values=nothing)

    values != nothing || return deepcopy(tree)
    length(tree) == length(values) ||
        throw(DimensionMismatch("Length of Params object not matching number "*
                                "of supplied values"))

    pairs = []
    pos = 1
    
    for oldpair in tree.pairs
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

struct NumericRange{T} <: ParamRange
    lower::T
    upper::T
    transform
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
                    lower=nothing, upper=nothing, transform=identity)
    T = get_type(typeof(model), field)
    if T <: Real
        lower !=nothing && upper != nothing ||
            error("You must specify lower=... and upper=... .")
        return NumericRange{T}(lower, upper, transform)
    else
        values !=nothing || error("You must specify values=... .")
        return NominalRange{T}(Tuple(values))
    end
end

# deterministic iterators:
iterator(param_range::NominalRange) = param_range.values
function iterator(param_range::NumericRange{T}, n::Int) where T
    raw = range(param_range.lower, stop=param_range.upper, length=n)
    return map(raw) do value
        round(T, param_range.transform(value))
    end
end

    

