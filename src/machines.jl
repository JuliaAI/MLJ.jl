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
        !(M <: Supervised) || length(args) > 1 ||
            throw(error("Wrong number of arguments. "*
                        "You must provide target(s) for supervised models."))
        !(M <: Unsupervised) || length(args) == 1 ||
            throw(error("Wrong number of arguments. "*
                        "Use NodalMachine(model, X) for an unsupervised  model."))
        
        machine = new{M}(model)

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
            rows = (:) 
        end
        X = coerce(machine.model, args[1][Rows, rows])
        ys = [arg[rows] for arg in args[2:end]]
        machine.fitresult, machine.cache, report =
            fit(machine.model, verbosity, X, ys...)
        machine.rows = rows
    else
        if rows == nothing # (ie rows not specified) update:
            X = coerce(machine.model, args[1][Rows, machine.rows])
            ys = [arg[machine.rows] for arg in args[2:end]]
            machine.fitresult, machine.cache, report =
                update(machine.model, verbosity, machine.fitresult,
                       machine.cache, X, ys...)
        else # retrain from scratch:
            X = coerce(machine.model, args[1][Rows, rows])
            ys = [arg[rows] for arg in args[2:end]]
            machine.fitresult, machine.cache, report =
                fit(machine.model, verbosity, X, ys...)
            machine.rows = rows
        end
    end

    if report != nothing
        merge!(machine.report, report)
    end

    return machine

end

machine(model::Model, args...) = Machine(model, args...)


