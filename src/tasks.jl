function Base.getindex(task::SupervisedTask{U}, r) where U
    X = selectrows(task.X, r)
    y = selectrows(task.y, r)
    is_probabilistic = task.is_probabilistic
    target_scitype = task.target_scitype
    input_scitypes = task.input_scitypes
    input_is_multivariate = task.input_is_multivariate
    return SupervisedTask{U}(X,
                             y,
                             is_probabilistic,
                             target_scitype,
                             input_scitypes,
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



    



    
