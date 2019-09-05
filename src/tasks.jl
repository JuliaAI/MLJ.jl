## ROW INDEXING

function Base.getindex(task::SupervisedTask, r)
    X = selectrows(task.X, r)
    y = selectrows(task.y, r)
    is_probabilistic = task.is_probabilistic
    input_scitypes =  NamedTuple{schema(X).names}(schema(X).scitypes)
    target = task.target
    input_scitype = scitype(X)
    target_scitype = scitype(y)
    return SupervisedTask(X,
                          y,
                          is_probabilistic,
                          input_scitypes,
                          target,
                          input_scitype,
                          target_scitype)
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
                      is_probabilistic=false,
                      verbosity=1)

A more customizable constructor, this returns a supervised learning
task with input features `X` and target `y`, where: `X` is a table or
vector (univariate inputs), while `y` must be a vector whose elements
are scalars, or tuples scalars (of constant length for ordinary
multivariate predictions, and of variable length for sequence
prediction). Table rows must correspond to patterns and columns to
features. Type coercion is not available for this constructor (but see
also [`coerce`](@ref)).

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

See also [`scitype`](@ref), [`scitype_union`](@ref) 

"""
unsupervised(; data=nothing, types=Dict(), kwargs...) =
	    UnsupervisedTask(; data = coerce(types, data), kwargs...)
