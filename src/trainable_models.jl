abstract type AbstractTrainableModel{M} <: MLJType end

mutable struct TrainableModel{M<:Model} <: AbstractTrainableModel{M}

    model::M
    fitresult
    cache
    args::Tuple
    report
    rows # remember last rows used for convenience
    
    function TrainableModel{M}(model::M, args...) where M<:Model

        # check number of arguments for model subtypes:
        !(M <: Supervised) || length(args) == 2 ||
            throw(error("Wrong number of arguments. "*
                        "Use NodalTrainableModel(model, X, y) for supervised learner models."))
        !(M <: Unsupervised) || length(args) == 1 ||
            throw(error("Wrong number of arguments. "*
                        "Use NodalTrainableModel(model, X) for an unsupervised learner model."))
        
        trainable_model = new{M}(model)

        # if M <: Supervised
        #     X = coerce(model, args[1])
        #     y = args[2]
        #     trainable_model.args = (X, y)
        # else
        #     trainable_model.args = args
        # end
        trainable_model.args = args
        
        trainable_model.report = Dict{Symbol,Any}()

        return trainable_model

    end
end

# automatically detect type parameter:
TrainableModel(model::M, args...) where M<:Model = TrainableModel{M}(model, args...)

# constructor for tasks instead of bare data:
# TrainableModel(model::Model, task::SupervisedTask) = TrainableModel(model, X_and_y(task)...)
# TrainableModel(model::Model, task::UnsupervisedTask) = TrainableModel(model, task.data)

# TODO: The fit code below is almost identical to NodalTrainableModel
# fit code in networks.jl and we ought to combine the two by, say,
# making generic data and vectors callable on rows.

# fit method, general case (no coercion of arguments):
function fit!(trainable_model::TrainableModel; rows=nothing, verbosity=1)

    warning = clean!(trainable_model.model)
    isempty(warning) || verbosity < 0 || @warn warning 
    
#    verbosity < 1 || @info "Training $trainable_model whose model is $(trainable_model.model)."
    verbosity < 1 || @info "Training $trainable_model."

    if !isdefined(trainable_model, :fitresult)
        if rows == nothing
            rows = (:) # error("An untrained TrainableModel requires rows to fit.")
        end
        args = [arg[Rows, rows] for arg in trainable_model.args]
        trainable_model.fitresult, trainable_model.cache, report =
            fit(trainable_model.model, verbosity, args...)
        trainable_model.rows = rows
    else
        if rows == nothing # (ie rows not specified) update:
            args = [arg[Rows, trainable_model.rows] for arg in trainable_model.args]
            trainable_model.fitresult, trainable_model.cache, report =
                update(trainable_model.model, verbosity, trainable_model.fitresult,
                       trainable_model.cache, args...)
        else # retrain from scratch:
            args = [arg[Rows, rows] for arg in trainable_model.args]
            trainable_model.fitresult, trainable_model.cache, report =
                fit(trainable_model.model, verbosity, args...)
            trainable_model.rows = rows
        end
    end

    if report != nothing
        merge!(trainable_model.report, report)
    end

    return trainable_model

end

# fit method, supervised case (input data coerced):
function fit!(trainable_model::TrainableModel{M};
              rows=nothing, verbosity=1) where M<:Supervised

    warning = clean!(trainable_model.model)
    isempty(warning) || verbosity < 0 || @warn warning 
    
#    verbosity < 1 || @info "Training $trainable_model whose model is $(trainable_model.model)."
    verbosity < 1 || @info "Training $trainable_model."

    args = trainable_model.args
    if !isdefined(trainable_model, :fitresult)
        if rows == nothing
            rows = (:) # error("An untrained TrainableModel requires rows to fit.")
        end
        X = coerce(trainable_model.model, args[1][Rows, rows])
        y = args[2][rows]
        trainable_model.fitresult, trainable_model.cache, report =
            fit(trainable_model.model, verbosity, X, y)
        trainable_model.rows = rows
    else
        if rows == nothing # (ie rows not specified) update:
            X = coerce(trainable_model.model, args[1][Rows, trainable_model.rows])
            y = args[2][trainable_model.rows]
            trainable_model.fitresult, trainable_model.cache, report =
                update(trainable_model.model, verbosity, trainable_model.fitresult,
                       trainable_model.cache, X, y)
        else # retrain from scratch:
            X = coerce(trainable_model.model, args[1][Rows, rows])
            y = args[2][rows]
            trainable_model.fitresult, trainable_model.cache, report =
                fit(trainable_model.model, verbosity, X, y)
            trainable_model.rows = rows
        end
    end

    if report != nothing
        merge!(trainable_model.report, report)
    end

    return trainable_model

end

trainable(model::Model, args...) = TrainableModel(model, args...)


