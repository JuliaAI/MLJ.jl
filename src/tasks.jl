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


coerce(T::Type{Continuous}, y) = convert(Vector{Float64}, y)
coerce(T::Type{Count}, y)      = convert(Vector{Int}, y)
coerce(T::Type{Multiclass}, y)          = categorical(y, ordered = false)
coerce(T::Type{FiniteOrderedFactor}, y) = categorical(y, ordered = true)

function coerce(types::Dict{Symbol, Type}, X)
    names = schema(X).names
    NamedTuple{names}(coerce(types[name], selectcols(X, name)) for name in names)
end
