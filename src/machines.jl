abstract type AbstractMachine{M} <: MLJType end


# TODO: write out separate method for machine(::Model, ::MLJTask) to simplify logic. 
mutable struct Machine{M<:Model} <: AbstractMachine{M}

    model::M
    previous_model::M
    fitresult
    cache
    args::Tuple
    report
    rows # remember last rows used for convenience

    function Machine{M}(model::M, args...) where M<:Model

        # checks on args:
        if M <: Supervised
            (length(args) == 2) ||
                error("Use machine(model, X, y) for a supervised model. ")
            X, y = args
            T =
                input_is_multivariate(model) ? Union{scitypes(X)...} : scitype_union(X)
            T <: input_scitype_union(model) ||
                error("The scitypes of elements of X, in machine(model, X, y), "*
                      "should be a subtype of $(input_scitype_union(model)). ")
            y isa AbstractVector ||
                error("The y, in machine(model, X, y), should be an AbstractVector "*
                      "(possibly of tuples). ")
            scitype_union(y) <: target_scitype_union(model) || 
                error("The scitype of elements of y, in machine(model, X, y), "*
                      "should be a subtype of $(target_scitype_union(model)). ")
        end
        if M <: Unsupervised
            length(args) == 1 ||
                error("Wrong number of arguments. "*
                      "Use machine(model, X) for an unsupervised model.")
            X = args[1]
            container_type(X) in [:table, :sparse] || args[1] isa AbstractVector ||
                error("The X, in machine(model, X), should be a table, sparse table or AbstractVector. "*
                      "Use MLJ.table(X) to wrap an AbstractMatrix X as a table. ")
            U =
                input_is_multivariate(model) ?  Union{scitypes(X)...} : scitype_union(X)
            U <: input_scitype_union(model) || 
                error("The scitype of elements of X, in machine(model, X), should be a subtype "*
                      "of $(input_scitype_union(model)). ")
        end

        machine = new{M}(model)
        machine.args = args
        
        return machine

    end
end

# automatically detect type parameter:
Machine(model::M, args...) where M<:Model = Machine{M}(model, args...)

machine(model::Model, task::SupervisedTask) = machine(model, task.X, task.y)
machine(model::Model, task::UnsupervisedTask) = machine(model, task.X)


# Note: The following method is written to fit `NodalMachine`s
# defined in networks.jl, in addition to `Machine`s defined above.

"""
    fit!(mach::Machine; rows=nothing, verbosity=1, force=false)

When called for the first time, call `MLJBase.fit` on `mach.model` and
store the returned fit-result and report. Subsequent calls do nothing
unless: (i) `force=true`, or (ii) the specified `rows` are different
from those used the last time a fit-result was computed, or (iii)
`mach.model` has changed since the last time a fit-result was computed
(the machine is *stale*). In cases (i) or (ii) `MLJBase.fit` is 
called on `mach.model`. Otherwise, `MLJBase.update` is called.

    fit!(mach::NodalMachine; rows=nothing, verbosity=1, force=false)

When called for the first time, attempt to call `MLJBase.fit` on
`fit.model`. This will fail if a machine in the dependency tape of
`mach` has not been trained yet, which can be resolved by fitting any
downstream node instead. Subsequent `fit!` calls do nothing unless:
(i) `force=true`, or (ii) some machine in the dependency tape of
`mach` has computed a new fit-result since `mach` last computed its
fit-result, or (iii) the specified `rows` have changed since the last
time a fit-result was last computed, or (iv) `mach` is stale (see
below). In cases (i), (ii) or (iii), `MLJBase.fit` is
called. Otherwise `MLJBase.update` is called.

A machine `mach` is *stale* if `mach.model` has changed since the last
time a fit-result was computed, or if if one of its training arguments
is `stale`. A node `N` is stale if `N.machine` is stale or one of its
arguments is stale. Source nodes are never stale. 

Note that a nodal machine obtains its training data by *calling* its
node arguments on the specified `rows` (rather *indexing* its arguments
on those rows) and that this calling is a recursive operation on nodes
upstream of those arguments.

"""
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

    if mach isa NodalMachine
        # determine if concrete data to be used in training may have changed:
        upstream_state = broadcast(m -> m.state, mach.tape)
        data_has_changed = rows_have_changed || (upstream_state != mach.upstream_state)
    else
        data_has_changed = rows_have_changed
    end

    args = [selectrows(arg, rows) for arg in mach.args]

    if !isdefined(mach, :fitresult) || data_has_changed || force
        # fit the model:
        verbosity < 1 || @info "Training $mach."
        mach.fitresult, mach.cache, mach.report =
            fit(mach.model, verbosity, args...)
    elseif !is_stale(mach)
        # don't fit the model
        if verbosity > 0
            @info "Not retraining $mach.\n It appears up-to-date. "*
            "Use force=true to force retraining."
        end
        return mach
    else
        # update the model:
        verbosity < 1 || @info "Updating $mach."
        mach.fitresult, mach.cache, mach.report =
            update(mach.model, verbosity, mach.fitresult, mach.cache, args...)
    end

    if rows_have_changed
        mach.rows = deepcopy(rows)
    end

    mach.previous_model = deepcopy(mach.model)

    if mach isa NodalMachine
        mach.upstream_state = upstream_state
        mach.state = mach.state + 1
    end
    
    return mach

end

is_stale(mach::Machine) =
    !isdefined(mach, :fitresult) || (mach.model != mach.previous_model)

machine(model::Model, args...) = Machine(model, args...)

params(mach::AbstractMachine) = params(mach.model)
report(mach::AbstractMachine) = mach.report


