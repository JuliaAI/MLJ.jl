# Internals

## The machine interface, simplified

The following is simplified description of the `Machine`
interface. See also the [Glossary](glossary.md)

### The Machine type

````julia
mutable struct Machine{M<Model}

    model::M
    fitresult
    cache
    args::Tuple    # e.g., (X, y) for supervised models
    report
    previous_rows # remember last rows used

    function Machine{M}(model::M, args...) where M<:Model
        machine = new{M}(model)
        machine.args = args
        machine.report = Dict{Symbol,Any}()
        return machine
    end

end
````

### Constructor

````julia
machine(model::M, Xtable, y) = Machine{M}(model, Xtable, y)
````

### fit! and predict/transform

````julia
function fit!(machine::Machine; rows=nothing, force=false, verbosity=1)

    warning = clean!(mach.model)
    isempty(warning) || verbosity < 0 || @warn warning

    if rows === nothing
        rows = (:)
    end

    rows_have_changed  = (!isdefined(mach, :previous_rows) ||
	    rows != mach.previous_rows)

    args = [MLJ.selectrows(arg, rows) for arg in mach.args]

    if !isdefined(mach, :fitresult) || rows_have_changed || force
        mach.fitresult, mach.cache, report =
            fit(mach.model, verbosity, args...)
    else # call `update`:
        mach.fitresult, mach.cache, report =
            update(mach.model, verbosity, mach.fitresult, mach.cache, args...)
    end

    if rows_have_changed
        mach.previous_rows = deepcopy(rows)
    end

    if report !== nothing
        merge!(mach.report, report)
    end

    return mach

end

function predict(machine::Machine{<:Supervised}, Xnew)
    if isdefined(machine, :fitresult)
        return predict(machine.model, machine.fitresult, Xnew))
    else
        throw(error("$machine is not trained and so cannot predict."))
    end
end

function transform(machine::Machine{<:Unsupervised}, Xnew)
    if isdefined(machine, :fitresult)
        return transform(machine.model, machine.fitresult, Xnew))
    else
        throw(error("$machine is not trained and so cannot transform."))
    end
end
````
