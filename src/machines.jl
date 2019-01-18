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

function fit!(mach::AbstractMachine; rows=nothing, verbosity=1, force=false)

    if mach isa NodalMachine && mach.frozen 
        verbosity < 0 || @warn "$mach not trained as it is frozen."
        return mach
    end

    warning = clean!(mach.model)
    isempty(warning) || verbosity < 0 || @warn warning 
    
    if rows == nothing
        rows = (:) 
    end

    rows_have_changed  = (!isdefined(mach, :rows) || rows != mach.rows)

    if mach isa NodalMachine && !is_stale(mach) && !rows_have_changed && !force
        return mach
    end
    
    verbosity < 1 || @info "Training $mach."

    if mach.model isa Supervised
            X = coerce(mach.model, mach.args[1][Rows, rows])
            ys = [arg[Rows, rows] for arg in mach.args[2:end]]
        args = (X, ys...)
    else
        args = [arg[Rows, rows] for arg in mach.args]
    end

    if !isdefined(mach, :fitresult) || rows_have_changed || force 
        mach.fitresult, mach.cache, report =
            fit(mach.model, verbosity, args...)
    else # call `update`:
        mach.fitresult, mach.cache, report =
            update(mach.model, verbosity, mach.fitresult, mach.cache, args...)
    end

    if rows_have_changed
        mach.rows = deepcopy(rows)
    end

    if mach isa NodalMachine
        mach.previous_model = deepcopy(mach.model)
    end
    
    if report != nothing
        merge!(mach.report, report)
    end

    return mach

end

machine(model::Model, args...) = Machine(model, args...)


