function Base.getindex(task::SupervisedTask, r)
    X = selectrows(task.X, r)
    y = task.y[r]
    is_probabilistic = task.is_probabilistic
    input_scitypes = scitypes(X)
    input_scitype_union = Union{input_scitypes...}
    target_scitype_union = scitype_union(y)
    input_is_multivariate = task.input_is_multivariate
    return SupervisedTask(X,
                          y,
                          is_probabilistic,
                          input_scitypes,
                          input_scitype_union,
                          target_scitype_union,
                          input_is_multivariate)
end

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

Random.shuffle(rng::AbstractRNG, task::SupervisedTask) = task[shuffle!(rng, Vector(1:nrows(task)))]
Random.shuffle(task::SupervisedTask) = task[shuffle!(Vector(1:nrows(task)))]

## Coercion
_coerce_missing_warn(T) =
    @warn "Missing values encountered. Coerced to Union{Missing,$T} instead of $T."

# Vector to Continuous
coerce(T::Type{Continuous}, y::AbstractVector{<:Number}) = float(y)
function coerce(T::Type{Continuous}, y::AbstractVector{Union{<:Number,Missing}})
    _coerce_missing_warn(T)
    return float(y)
end
function coerce(T::Type{Continuous}, y::AbstractVector)
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
function coerce(T::Type{Count}, y::AbstractVector{Union{<:Real,Missing}})
    _coerce_missing_warn(T)
    return convert(Vector{Missing,Int}, y)
end
function coerce(T::Type{Count}, y::AbstractVector)
    for el in y
        if ismissing(el)
            _coerce_missing_warn(T)
            break
        end
    end
    return _int.(y)
end

# Vector to Multiclass
function coerce(T::Type{Multiclass}, y)
    su = scitype_union(y)
    if su >: Missing
        _coerce_missing_warn(T)
    end
    if su <: T
        return y
    else
        return categorical(y, ordered = false)
    end
end

# Vector to FiniteOrderedFactor
function coerce(T::Type{FiniteOrderedFactor}, y)
    su = scitype_union(y)
    if su >: Missing
        _coerce_missing_warn(T)
    end
    if su <: T
        return y
    else
        return categorical(y, ordered = true)
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

function coerce(types::Dict, X)
    names = schema(X).names
    coltable = NamedTuple{names}(_coerce_col(X, name, types) for name in names)
    return MLJBase.table(coltable, prototype=X)
end
