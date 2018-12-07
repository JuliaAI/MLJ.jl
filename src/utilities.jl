"""
    partition(rows::AbstractVector{Int}, fractions...)

Splits the vector `rows` into a tuple of vectors whose lengths are
given by the corresponding `fractions` of `length(rows)`. The last
fraction is not provided, as it is inferred from the preceding
ones. So, for example,

    julia> partition(1:1000, 0.2, 0.7)
    (1:200, 201:900, 901:1000)

"""
function partition(rows::AbstractVector{Int}, fractions...)
    rows = collect(rows)
    rowss = []
    if sum(fractions) >= 1
        throw(DomainError)
    end
    n_patterns = length(rows)
    first = 1
    for p in fractions
        n = round(Int, p*n_patterns)
        n == 0 ? (@warn "A split has only one element"; n = 1) : nothing
        push!(rowss, rows[first:(first + n - 1)])
        first = first + n
    end
    if first > n_patterns
        @warn "Last vector in the split has only one element."
        first = n_patterns
    end
    push!(rowss, rows[first:n_patterns])
    return tuple(rowss...)
end

macro colon(p)
    Expr(:quote, p)
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

function readlibsvm(fname::String, shape)
    dmx = zeros(Float32, shape)
    label = Float32[]
    fi = open(fname, "r")
    cnt = 1
    for line in eachline(fi)
        line = split(line, " ")
        push!(label, parse(Float64, line[1]))
        line = line[2:end]
        for itm in line
            itm = split(itm, ":")
            dmx[cnt, parse(Int, itm[1]) + 1] = float(parse(Int, itm[2]))
        end
        cnt += 1
    end
    close(fi)
    return (dmx, label)
end
