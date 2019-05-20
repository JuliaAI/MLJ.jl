## ROW INDEXING

function Base.getindex(task::SupervisedTask, r)
    X = selectrows(task.X, r)
    y = task.y[r]
    is_probabilistic = task.is_probabilistic
    input_scitypes = scitypes(X)
    target = task.target
    input_scitype_union = Union{input_scitypes...}
    target_scitype_union = scitype_union(y)
    input_is_multivariate = task.input_is_multivariate
    return SupervisedTask(X,
                          y,
                          is_probabilistic,
                          input_scitypes,
                          target,
                          input_scitype_union,
                          target_scitype_union,
                          input_is_multivariate)
end


## ROWS SHUFFLING

function Random.shuffle!(rng::AbstractRNG, task::SupervisedTask)
    rows = shuffle!(rng, Vector(1:nrows(task)))
    task.X = selectrows(task.X, rows)
    task.y = selectrows(task.y, rows)
    return task
end

function Random.shuffle!(task::SupervisedTask)
    rows = shuffle!(Vector(1:nrows(task)))
    task.X = selectrows(task.X, rows)
    task.y = selectrows(task.y, rows)
    return task
end

Random.shuffle(rng::AbstractRNG, task::SupervisedTask) =
    task[shuffle!(rng, Vector(1:nrows(task)))]
Random.shuffle(task::SupervisedTask) = task[shuffle!(Vector(1:nrows(task)))]


## COERCION

_coerce_missing_warn(T) =
    @warn "Missing values encountered. Coerced to Union{Missing,$T} instead of $T."


# Vector to Continuous
"""
    coerce(T, v::AbstractVector)

Coerce the machine types of elements of `v` to ensure the returned
vector has `T` as its `scitype_union`, or `Union{Missing,T}`, if `v` has
missing values.

    julia> v = coerce(Continuous, [1, missing, 5])
    3-element Array{Union{Missing, Float64},1}:
     1.0     
     missing
     5.0  

    julia> scitype_union(v)
    Union{Missing,Continuous}

See also scitype, scitype_union, scitypes

"""
coerce(T::Type{Continuous}, y::AbstractVector{<:Number}) = float(y)
function coerce(T::Type{Continuous}, y::V) where {N<:Number,
                                                  V<:AbstractVector{Union{N,Missing}}}
    _coerce_missing_warn(T)
    return float(y)
end
function coerce(T::Type{Continuous}, y::AbstractVector{S}) where S
    for el in y
        if ismissing(el)
            _coerce_missing_warn(T)
            break
        end
    end
    return float.(y)
end

# Vector to Count
_int(::Missing) = missing
_int(x) = Int(x)

coerce(T::Type{Count}, y::AbstractVector{<:Integer}) = y
function coerce(T::Type{Count}, y::V) where {R<:Real,
                                             V<:AbstractVector{Union{R,Missing}}}
    _coerce_missing_warn(T)
    return convert(Vector{Union{Missing,Int}}, y)
end
function coerce(T::Type{Count}, y::V) where {S,V<:AbstractVector{S}}
    for el in y
        if ismissing(el)
            _coerce_missing_warn(T)
            break
        end
    end
    return _int.(y)
end

# Vector to Multiclass and OrderedFactor
for (T, ordered) in ((Multiclass, false), (OrderedFactor, true))
    @eval function coerce(::Type{$T}, y)
        su = scitype_union(y)
        if su >: Missing
            _coerce_missing_warn($T)
        end
        if su <: $T
            return y
        else
            return categorical(y, true, ordered = $ordered)
        end
    end
end

# Coerce table
function _coerce_col(X, name, types)
    y = selectcols(X, name)
    if haskey(types, name)
        return coerce(types[name], y)
    else
        return y
    end
end

"""
    coerce(d::Dict, X)

Return a copy of the table `X` with columns named in the keys of `d`
coerced to have `scitype_union` equal to the corresponding value. 

"""
function coerce(types::Dict, X)
    names = schema(X).names
    coltable = NamedTuple{names}(_coerce_col(X, name, types) for name in names)
    return MLJBase.table(coltable, prototype=X)
end

# Attempt to coerce a vector using a dictionary with a single key (corner case):
function coerce(types::Dict, v::AbstractVector)
    kys = keys(types)
    length(kys) == 1 || error("Cannot coerce a vector using a multi-keyed dictionary of types. ")
    key = first(kys)
    return coerce(types[key], v)
end


## TASK CONSTRUCORS WITH OPTIONAL TYPE COERCION

"""
    task = supervised(data=nothing, 
                      types=Dict(), 
                      target=nothing,  
                      ignore=Symbol[], 
                      is_probabilistic=false, 
                      verbosity=1)

Construct a supervised learning task with input features `X` and
target `y`, where: `y` is the column vector from `data` named
`target`, if this is a single symbol, or a vector of tuples, if
`target` is a vector; `X` consists of all remaining columns of `data`
not named in `ignore`, and is a table unless it has only one column, in
which case it is a vector.

The data types of elements in a column of `data` named as a key of the
dictionary `types` are coerced to have a scientific type given by the
corresponding value. Possible values are `Continuous`, `Multiclass`,
`OrderedFactor` and `Count`. So, for example,
`types=Dict(:x1=>Count)` means elements of the column of `data` named
`:x1` will be coerced into integers (whose scitypes are always `Count`).

    task = supervised(X, y; 
                      input_is_multivariate=true, 
                      is_probabilistic=false, 
                      verbosity=1)

A more customizable constructor, this returns a supervised learning
task with input features `X` and target `y`, where: `X` must be a
table or vector, according to whether it is multivariate or
univariate, while `y` must be a vector whose elements are scalars, or
tuples scalars (of constant length for ordinary multivariate
predictions, and of variable length for sequence prediction). Table
rows must correspond to patterns and columns to features. Type
coercion is not available for this constructor (but see also `coerce`).

    X, y = task()

Returns the input `X` and target `y` of the task, also available as
`task.X` and `task.y`.

"""
supervised(; data=nothing, types=Dict(), kwargs...) =
	    SupervisedTask(; data = coerce(types, data), kwargs...)
supervised(X, y; kwargs...) = SupervisedTask(X, y; kwargs...)
"""
    task = unsupervised(data=nothing, types=Dict(), ignore=Symbol[], verbosity=1)

Construct an unsupervised learning task with given input `data`, which
should be a table or, in the case of univariate inputs, a single
vector. 

The data types of elements in a column of `data` named as a key of the
dictionary `types` are coerced to have a scientific type given by the
corresponding value. Possible values are `Continuous`, `Multiclass`,
`OrderedFactor` and `Count`. So, for example,
`types=Dict(:x1=>Count)` means elements of the column of `data` named
`:x1` will be coerced into integers (whose scitypes are always `Count`).

Rows of `data` must correspond to patterns and columns to
features. Columns in `data` whose names appear in `ignore` are
ignored.

    X = task()

Return the input data in form to be used in models.

See also [`scitype`](@ref), [`scitype_union`](@ref), [`scitypes`](@ref).

"""
unsupervised(; data=nothing, types=Dict(), kwargs...) =
	    UnsupervisedTask(; data = coerce(types, data), kwargs...)
