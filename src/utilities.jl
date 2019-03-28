
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

"""
## `@curve`

The code,

    @curve var range code

evaluates `code`, replacing appearances of `var` therein with each
value in `range`. The range and corresponding evaluations are returned
as a tuple of arrays. For example,

    @curve  x 1:3 (x^2 + 1)

evaluates to

    ([1,2,3], [2, 5, 10])

This is convenient for plotting functions using, eg, the `Plots` package:

    plot(@curve x 1:3 (x^2 + 1))

A macro `@pcurve` parallelizes the same behaviour.  A two-variable
implementation is also available, operating as in the following
example:

    julia> @curve x [1,2,3] y [7,8] (x + y)
    ([1,2,3],[7 8],[8.0 9.0; 9.0 10.0; 10.0 11.0])

    julia> ans[3]
    3×2 Array{Float64,2}:
      8.0   9.0
      9.0  10.0
     10.0  11.0

N.B. The second range is returned as a *row* vector for consistency
with the output matrix. This is also helpful when plotting, as in:

    julia> u1, u2, A = @curve x range(0, stop=1, length=100) α [1,2,3] x^α
    julia> u2 = map(u2) do α "α = "*string(α) end
    julia> plot(u1, A, label=u2)

which generates three superimposed plots - of the functions x, x^2 and x^3 - each
labels with the exponents α = 1, 2, 3 in the legend.

"""
macro curve(var1, range, code)
    quote
        output = []
        N = length($(esc(range)))
        for i in eachindex($(esc(range)))
            local $(esc(var1)) = $(esc(range))[i]
            print((@colon $var1), "=", $(esc(var1)), "                   \r")
            flush(stdout)
            # print(i,"\r"); flush(stdout)
            push!(output, $(esc(code)))
        end
        collect($(esc(range))), [x for x in output]
    end
end

macro curve(var1, range1, var2, range2, code)
    quote
        output = Array{Float64}(undef, length($(esc(range1))), length($(esc(range2))))
        for i1 in eachindex($(esc(range1)))
            local $(esc(var1)) = $(esc(range1))[i1]
            for i2 in eachindex($(esc(range2)))
                local $(esc(var2)) = $(esc(range2))[i2]
                # @dbg $(esc(var1)) $(esc(var2))
                print((@colon $var1), "=", $(esc(var1)), " ")
                print((@colon $var2), "=", $(esc(var2)), "                    \r")
                flush(stdout)
                output[i1,i2] = $(esc(code))
            end
        end
        collect($(esc(range1))), permutedims(collect($(esc(range2)))), output
    end
end

macro pcurve(var1, range, code)
    quote
        N = length($(esc(range)))
        pairs = @distributed vcat for i in eachindex($(esc(range)))
            local $(esc(var1)) = $(esc(range))[i]
#            print((@colon $var1), "=", $(esc(var1)), "                    \r")
#            flush(stdout)
#            print(i,"\r"); flush(stdout)
            [( $(esc(range))[i], $(esc(code)) )]
        end
        sort!(pairs, by=first)
        collect(map(first,pairs)), collect(map(last, pairs))
    end
end
