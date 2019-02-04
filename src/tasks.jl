# TODO: add evaluation metric:
# TODO: add `input_kinds` and `outputs_are`
# TODO: add multiple targets

abstract type Task <: MLJType end

struct UnsupervisedTask <: Task
    data
    ignore::Vector{Symbol}
    operation::Function    # transform, inverse_transform, etc
    properties::Tuple
end

function UnsupervisedTask(
    ; data=nothing
    , ignore=Symbol[]
    , operation=predict
    , properties=())

    data != nothing         || throw(error("You must specify data=..."))

    return SupervisedTask(data, ignore, operation, properties)
end

struct SupervisedTask <: Task
    data
    target::Symbol
    ignore::Vector{Symbol}
    operation::Function    # predict, predict_proba, etc
    properties::Tuple
end

function SupervisedTask(
    ; data=nothing
    , target=nothing
    , ignore=Symbol[]
    , operation=predict
    , properties=())

    data != nothing         || throw(error("You must specify data=..."))
    target != nothing       || throw(error("You must specify target=..."))
    target in names(data)   || throw(error("Supplied data does not have $target as field."))

    return SupervisedTask(data, target, ignore, operation, properties)
end


## RUDIMENTARY TASK OPERATIONS

nrows(task::Task) = schema(task.data).nrows
Base.eachindex(task::Task) = Base.OneTo(nrows(task))

features(task::Task) = filter!(schema(task.data).names |> collect) do ftr
    !(ftr in task.ignore)
end

features(task::SupervisedTask) = filter(schema(task.data).names |> collect) do ftr
    ftr != task.target && !(ftr in task.ignore)
end

X_and_y(task::SupervisedTask) = (selectcols(task.data, features(task)),
                                 selectcols(task.data, task.target))
