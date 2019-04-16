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


coerce(T::Type{Continuous}, y) = float(y)
coerce(T::Type{Count}, y)      = convert(Vector{Int}, y)
coerce(T::Type{Count}, y::AbstractVector{<:Integer}) = y
function coerce(T::Type{Multiclass}, y)
    if scitype_union(y) <: T
        return y
    else
        return categorical(y, ordered = false)
    end
end
function coerce(T::Type{FiniteOrderedFactor}, y)
    if scitype_union(y) <: T
        return y
    else
        return categorical(y, ordered = true)
    end
end

function coerce(types::Dict{Symbol, Type}, X)
    names = schema(X).names
    coltable = NamedTuple{names}(coerce(types[name], selectcols(X, name))
                                 for name in names)
    return MLJBase.table(coltable, prototype=X)
end

supervisedtask(; data=nothing, types=nothing, kwargs...) =
    SupervisedTask(; data = coerce(types, data), kwargs...)
unsupervisedtask(; data=nothing, types=nothing, kwargs...) =
    UnsupervisedTask(; data = coerce(types, data), kwargs...)
