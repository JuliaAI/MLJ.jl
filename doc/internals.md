# MLJ Internals

## The machine interface, simplified

The following is simplified description of the `Machine` interface, as
it applies to `Supervised` models.

### The types

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
````
    
### Constructor

````julia
machine(model::M, Xtable, y) = Machine{M}(model, Xtable, y)
````

### fit! and predict

````julia
function fit!(machine::Machine{<:Supervised}; rows=nothing, verbosity=1) 

    warning = clean!(mach.model)
    isempty(warning) || verbosity < 0 || @warn warning 

    if rows == nothing
        rows = (:) 
    end

    rows_have_changed  = (!isdefined(mach, :rows) || rows != mach.rows)

    X = coerce(mach.model, mach.args[1][Rows, rows])
    ys = [arg[Rows, rows] for arg in mach.args[2:end]]

    if !isdefined(mach, :fitresult) || rows_have_changed || force 
        mach.fitresult, mach.cache, report =
            fit(mach.model, verbosity, X, ys...)
    else # call `update`:
        mach.fitresult, mach.cache, report =
            update(mach.model, verbosity, mach.fitresult, mach.cache, X, ys...)
    end

    if rows_have_changed
        mach.rows = deepcopy(rows)
    end

    if report != nothing
        merge!(mach.report, report)
    end

    return mach

end

function predict(machine::Machine{<:Supervised}, Xnew)
    if isdefined(machine, :fitresult)
        return predict(machine.model,
                       machine.fitresult,
                       coerce(machine.model, Xnew))
    else
        throw(error("$machine is not trained and so cannot predict."))
    end
end
````




