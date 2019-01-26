# TODO: add evaluation metric:
# TODO: add `inputs_can_be` and `outputs_are`
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

nrows(task::Task) = select(task.data, Schema).nrows
Base.eachindex(task::Task) = Base.OneTo(nrows(task))

features(task::Task) = filter!(select(task.data, Schema).names |> collect) do ftr
    !(ftr in task.ignore)
end

features(task::SupervisedTask) = filter(select(task.data, Schema).names |> collect) do ftr
    ftr != task.target && !(ftr in task.ignore)
end

X_and_y(task::SupervisedTask) = (select(task.data, Cols, features(task)),
                                 select(task.data, Cols, task.target))
