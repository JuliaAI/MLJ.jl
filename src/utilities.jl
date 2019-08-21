"""
$SIGNATURES

Viewing a nested named tuple as a tree and return, as a tuple, the values
at the leaves, in the order they appear in the original tuple.

```julia-repl
julia> t = (X = (x = 1, y = 2), Y = 3)
julia> flat_values(t)
(1, 2, 3)
```
"""
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


## FOR ENCODING AND DECODING MODEL METADATA


# This method needs revisiting when new scitypes are introduced:
function encode_dic(s)
    if s isa Symbol
        return string(":", s)
    elseif s isa AbstractString
        return string(s)
    else # we have some more complicated object
        prestring = string("`", s, "`")
        # hack for objects with gensyms in their string representation:
        str = replace(prestring, '#'=>'_')
        return str
    end
end

encode_dic(v::Vector) = encode_dic.(v)
function encode_dic(d::AbstractDict)
    ret = LittleDict{}()
    for (k, v) in d
        ret[encode_dic(k)] = encode_dic(v)
    end
    return ret
end

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
function decode_dic(d::AbstractDict)
    ret = LittleDict()
    for (k, v) in d
        ret[decode_dic(k)] = decode_dic(v)
    end
    return ret
end

# the inverse of a multivalued dictionary is a multivalued
# dictionary:
function inverse(d::LittleDict{S,Set{T}}) where {S,T}
    dinv = LittleDict{T,Set{S}}()
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


## SOME GENERAL PURPOSE MACROS

macro colon(p)
    Expr(:quote, p)
end


"""
$SIGNATURES

Take a dictionary of type `{T, <:Real}` and return the keys sorted according to
the value associated with them.

```julia-repl
julia> d = Dict("abc"=>5, "def"=>7, "ghi"=>2)
julia> keys_ordered_by_values(d)
3-element Array{String,1}:
 "ghi"
 "abc"
 "def"
```
"""
keys_ordered_by_values(d::Dict{T,<:Real} where T) = sort(collect(keys(d)), by=k->d[k])


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


## FOR PRETTY PRINTING COLUMN TABLES

function pretty_table(X; showtypes=true, alignment=:l, kwargs...)
    names = schema(X).names |> collect
    if showtypes
        types = schema(X).types |> collect
        scitypes = schema(X).scitypes |> collect
        header = hcat(names, types, scitypes) |> permutedims
    else
        header  = names
    end
    try
        PrettyTables.pretty_table(MLJBase.matrix(X),
                                  header; alignment=alignment, kwargs...)
    catch
        println("Trouble displaying evaluation results.")
    end
end

