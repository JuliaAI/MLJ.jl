abstract type AbstractMachine{M} <: MLJType end

mutable struct Machine{M<:Model} <: AbstractMachine{M}

    model::M
    fitresult
    cache
    args::Tuple
    report
    rows # remember last rows used for convenience
    
    function Machine{M}(model::M, args...) where M<:Model

        # check number of arguments for model subtypes:
        !(M <: Supervised) || length(args) == 2 ||
            throw(error("Wrong number of arguments. "*
                        "Use NodalMachine(model, X, y) for supervised learner models."))
        !(M <: Unsupervised) || length(args) == 1 ||
            throw(error("Wrong number of arguments. "*
                        "Use NodalMachine(model, X) for an unsupervised learner model."))
        
        machine = new{M}(model)

        # if M <: Supervised
        #     X = coerce(model, args[1])
        #     y = args[2]
        #     machine.args = (X, y)
        # else
        #     machine.args = args
        # end
        machine.args = args
        
        machine.report = Dict{Symbol,Any}()

        return machine

    end
end

# automatically detect type parameter:
Machine(model::M, args...) where M<:Model = Machine{M}(model, args...)

# constructor for tasks instead of bare data:
# Machine(model::Model, task::SupervisedTask) = Machine(model, X_and_y(task)...)
# Machine(model::Model, task::UnsupervisedTask) = Machine(model, task.data)

# TODO: The fit code below is almost identical to NodalMachine
# fit code in networks.jl and we ought to combine the two by, say,
# making generic data and vectors callable on rows.

# fit method, general case (no coercion of arguments):
function fit!(machine::Machine; rows=nothing, verbosity=1)

    warning = clean!(machine.model)
    isempty(warning) || verbosity < 0 || @warn warning 
    
#    verbosity < 1 || @info "Training $machine whose model is $(machine.model)."
    verbosity < 1 || @info "Training $machine."

    if !isdefined(machine, :fitresult)
        if rows == nothing
            rows = (:) # error("An untrained Machine requires rows to fit.")
        end
        args = [arg[Rows, rows] for arg in machine.args]
        machine.fitresult, machine.cache, report =
            fit(machine.model, verbosity, args...)
        machine.rows = rows
    else
        if rows == nothing # (ie rows not specified) update:
            args = [arg[Rows, machine.rows] for arg in machine.args]
            machine.fitresult, machine.cache, report =
                update(machine.model, verbosity, machine.fitresult,
                       machine.cache, args...)
        else # retrain from scratch:
            args = [arg[Rows, rows] for arg in machine.args]
            machine.fitresult, machine.cache, report =
                fit(machine.model, verbosity, args...)
            machine.rows = rows
        end
    end

    if report != nothing
        merge!(machine.report, report)
    end

    return machine

end

# fit method, supervised case (input data coerced):
function fit!(machine::Machine{M};
              rows=nothing, verbosity=1) where M<:Supervised

    warning = clean!(machine.model)
    isempty(warning) || verbosity < 0 || @warn warning 
    
#    verbosity < 1 || @info "Training $machine whose model is $(machine.model)."
    verbosity < 1 || @info "Training $machine."

    args = machine.args
    if !isdefined(machine, :fitresult)
        if rows == nothing
            rows = (:) # error("An untrained Machine requires rows to fit.")
        end
        X = coerce(machine.model, args[1][Rows, rows])
        y = args[2][rows]
        machine.fitresult, machine.cache, report =
            fit(machine.model, verbosity, X, y)
        machine.rows = rows
    else
        if rows == nothing # (ie rows not specified) update:
            X = coerce(machine.model, args[1][Rows, machine.rows])
            y = args[2][machine.rows]
            machine.fitresult, machine.cache, report =
                update(machine.model, verbosity, machine.fitresult,
                       machine.cache, X, y)
        else # retrain from scratch:
            X = coerce(machine.model, args[1][Rows, rows])
            y = args[2][rows]
            machine.fitresult, machine.cache, report =
                fit(machine.model, verbosity, X, y)
            machine.rows = rows
        end
    end

    if report != nothing
        merge!(machine.report, report)
    end

    return machine

end

machine(model::Model, args...) = Machine(model, args...)


