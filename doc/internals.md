# MLJ Internals

## The machine interface, simplified

The following is simplified description of the `Machine` interface, as
it applies to `Supervised` models.

````julia 
abstract type AbstractMachine{M}

mutable struct Machine{M<Supervised} 

    model::M
    fitresult
    cache
    args::Tuple    # (Xtable, y) 
    report
    rows # remember last rows used 
    
    function Machine{M}(model::M, Xtable, y) where M<:Supervised
        machine = new{M}(model)
        machine.args = (Xtable, y)
        machine.report = Dict{Symbol,Any}()
        return machine
    end
end

# constructor:
machine(model::M, Xtable, y) = Machine{M}(model, Xtable, y)

# fit method:
function fit!(machine::Machine{<:Supervised}; rows=nothing, verbosity=1) 

    warning = clean!(machine.model)
    isempty(warning) || verbosity < 0 || @warn warning 
    
    verbosity < 1 || @info "Training $machine."

    args = machine.args
    if !isdefined(machine, :fitresult) # then train for first time:
        if rows == nothing
            rows = (:) # error("An untrained Machine requires rows to fit.")
        end
        X = coerce(machine.model, args[1][Rows, rows])
        y = args[2][rows]
        machine.fitresult, machine.cache, report =
            fit(machine.model, verbosity, X, y)
        machine.rows = rows
    else 
        if rows == nothing # (ie rows not specified) then update using previous rows:
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

# predict method:
function predict(machine::Machine{<:Supervised}, Xnew)
            if isdefined(machine, :fitresult)
                return predict(machine.model,
                               machine.fitresult,
                               coerce(machine.model, Xnew))
            else
                throw(error("$machine is not trained and so cannot predict."))
            end
        end

# predict_proba method:
function predict_proba(machine::Machine{<:Supervised}, Xnew)
            if isdefined(machine, :fitresult)
                return predict_proba(machine.model,
                               machine.fitresult,
                               coerce(machine.model, Xnew))
            else
                throw(error("$machine is not trained and so cannot predict."))
            end
        end

````


