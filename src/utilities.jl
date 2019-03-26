
# NOTE: The next three functions should be identical to those defined
# in MLJRegistry/src/MLJRegistry.jl (not explicitly shared for
# dependency reasons).

# for decoding metadata:
function decode_dic(s::String)
    if !isempty(s)
        if  s[1] == ':'
            return Symbol(s[2:end])
        elseif s[1] == '`'
            return eval(Meta.parse(s[2:end-1]))
        else
            return s
        end
    else
        return ""
    end
end
decode_dic(v::Vector) = decode_dic.(v)
function decode_dic(d::Dict)
    ret = Dict()
    for (k, v) in d
        ret[decode_dic(k)] = decode_dic(v)
    end
    return ret
end

# the inverse of a multivalued dictionary is a mulitvalued
# dictionary:
function inverse(d::Dict{S,Set{T}}) where {S,T}
    dinv = Dict{T,Set{S}}()
    for key in keys(d)
        for val in d[key]
            if val in keys(dinv)
                push!(dinv[val], key)
            else
                dinv[val] = Set([key,])
            end
        end
    end
    return dinv
end

macro colon(p)
    Expr(:quote, p)
end

function keys_ordered_by_values(d::Dict{T,S}) where {T, S<:Real}

    items = collect(d) # 1d array containing the (key, value) pairs
    sort!(items, by=pair->pair[2], alg=QuickSort)

    return T[pair[1] for pair in items]

end

