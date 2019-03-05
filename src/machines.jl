abstract type AbstractMachine{M} <: MLJType end

mutable struct Machine{M<:Model} <: AbstractMachine{M}

    model::M
    fitresult
    cache
    args::Tuple
    report
    rows # remember last rows used for convenience
    
    function Machine{M}(model::M, args...) where M<:Model

        # checks on args:
        if M <: Supervised
            if  (length(args) == 1 && !(args[1] isa SupervisedTask)) ||
                (length(args) == 2 && !Tables.istable(args[1])) ||
                length(args) > 2
                error("Use machine(model, task) or machine(model, X, y) "*
                      "for a supervised model.")
            end
            if length(args) == 2
                X, y = args
                scitype(X) <: input_scitypes(model) ||
                        error("The scitype of X in machine(model, X, y) should be a subtype of $(input_scitypes(model)). ")
                scitype(y) <: target_scitype(model) ||
                    error("The scitype of y in machine(model, X, y) should be a subtype of $(target_scitype(model)). ")
                T =  target_scitype(model)
                if T <: Tuple
                    Tables.istable(y) ||
                        error("y in machine(model, X, y) should be a table. ")
                else
                    if scitype(y)  <: Union{Continuous,Count}
                        y isa Vector ||
                            error("y in machine(model, X, y) should be a Vector. ")
                    elseif scitype(y) <: Union{Multiclass,FiniteOrderedFactor}
                        y isa CategoricalArray ||
                            error("y in machine(model, X, y) should be a CategoricalVector. ")
                    end
                end
            end
        end
        if M <: Unsupervised
            length(args) == 1 ||
                error("Wrong number of arguments. "*
                      "Use machine(model, X) or machine(model, task) for an unsupervised model.")
            Tables.istable(args[1]) || args[1] isa UnsupervisedTask ||
                error("X in machine(model, X) should be a table or  UnsupervisedTask. "*
                      "Use MLJ.table(X) to wrap an abstract matrix X as a table. ")
            if Tables.istable(args[1]) && !(scitype(args[1]) <: input_scitypes(model))
                error("The scitype of X in machine(model, X) should be a subtype of $(input_scitypes(model)). ")
            end
        end

        machine = new{M}(model)

        if args[1] isa MLJTask
            machine.args = args[1]()
        else
            machine.args = args
        end
        
        return machine

    end
end

# automatically detect type parameter:
Machine(model::M, args...) where M<:Model = Machine{M}(model, args...)

# Note: The following method is written to fit `NodalMachine`s
# defined in networks.jl, in addition to `Machine`s defined above.

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

    args = [selectrows(arg, rows) for arg in mach.args]
    
    if !isdefined(mach, :fitresult) || rows_have_changed || force 
        verbosity < 1 || @info "Training $mach."
        mach.fitresult, mach.cache, mach.report =
            fit(mach.model, verbosity, args...)
    else # call `update`:
        verbosity < 1 || @info "Updating $mach."
        mach.fitresult, mach.cache, mach.report =
            update(mach.model, verbosity, mach.fitresult, mach.cache, args...)
    end

    if rows_have_changed
        mach.rows = deepcopy(rows)
    end

    if mach isa NodalMachine
        mach.previous_model = deepcopy(mach.model)
    end
    
    return mach

end

machine(model::Model, args...) = Machine(model, args...)


