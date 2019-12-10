abstract type AbstractMachine{M<:Model} <: MLJType end

mutable struct Machine{M<:Model} <: AbstractMachine{M}

    model::M
    previous_model::M
    fitresult
    cache
    args::Tuple
    report
    previous_rows

    function Machine{M}(model, args...) where M
        machine = new{M}(model)
        machine.args = args
        return machine
    end

end

# automatically detect type parameter:
Machine(model::M, args...) where M <: Model = Machine{M}(model, args...)

# public constructor:
function machine(model::M, args...) where M <: Model

    nargs = length(args)

    if M <: Supervised

        # checks on args:
        if nargs == 2
            X, y = args
        elseif nargs == 3
            supports_weights(model) ||
                @info("$(typeof(model)) does not support sample weights and "*
                      "the supplied weights will be ignored in "*
                      "training.\n However, supplied weights will be passed "*
                      "to weight-supporting measures on calls to `evaluate!` "*
                      "and in tuning. ")
            X, y, w = args
            w isa AbstractVector{<:Real} ||
                throw(ArgumentError("Weights must be real. "))
            nrows(w) == nrows(y) ||
                throw(DimensionMismatch("Weights and target differ in length."))
        else
            error("Use `machine(model, X, y)` or `machine(model, X, y, w)` "*
                  "for a supervised model. ")
        end

        # checks on input type:
        input_scitype(model) <: Unknown ||
            scitype(X) <: input_scitype(model) ||
            @warn "The scitype of `X`, in `machine(model, X, y)` or " *
                    "`machine(model, X, y, w)` is "*
                    "incompatible with `model`:\n"*
                    "scitype(X) = $(scitype(X))\n"*
                    "input_scitype(model) = $(input_scitype(model)). "

        # checks on target type:
        target_scitype(model) <: Unknown ||
            scitype(y) <: target_scitype(model) ||
            @warn "The scitype of `y`, in `machine(model, X, y)` "*
                    "or `machine(model, X, y, w)` is "*
                    "incompatible with `model`:\n"*
                    "scitype(y) = $(scitype(y))\n"*
                    "target_scitype(model) = $(target_scitype(model)). "

        # checks on dimension matching:
        nrows(X) == nrows(y) ||
            throw(DimensionMismatch("Differing number of observations "*
                                    "in input and target. "))

    else # M <: Unsupervised

        if !(M <: Static)

            length(args) == 1 ||
                error("Wrong number of arguments. " *
                      "Use machine(model, X) for an unsupervised model.")
            X = args[1]

            input_scitype(model) <: Unknown ||
                scitype(X) <: input_scitype(model) ||
                @warn "The scitype of `X`, in `machine(model, X)` is "*
            "incompatible with `model`:\n"*
            "scitype(X) = $(scitype(X))\n"*
            "input_scitype(model) = $(input_scitype(model)). "

        end

    end
    return Machine(model, args...)
end

# to be depreciated:
machine(model::Model, task::SupervisedTask) = machine(model, task.X, task.y)
machine(model::Model, task::UnsupervisedTask) = machine(model, task.X)


# Note: The following method is written to fit `NodalMachine`s
# defined in networks.jl, in addition to `Machine`s defined above.

"""
    fit!(mach::Machine; rows=nothing, verbosity=1, force=false)

When called for the first time, call 

    MLJBase.fit(mach.model, verbosity, mach.args...)

storing the returned fit-result and report in `mach`. Subsequent calls
do nothing unless: (i) `force=true`, or (ii) the specified `rows` are
different from those used the last time a fit-result was computed, or
(iii) `mach.model` has changed since the last time a fit-result was
computed (the machine is *stale*). In cases (i) or (ii) `MLJBase.fit`
is called again. Otherwise, `MLJBase.update` is called.

    fit!(mach::NodalMachine; rows=nothing, verbosity=1, force=false)

When called for the first time, attempt to call 

    MLJBase.fit(mach.model, verbosity, mach.args...)

This will fail if an argument of the machine depends ultimately on
some other untrained machine for successful calling, but this is
resolved by instead calling `fit!` any node `N` for which
`mach in machines(N)` is true, which trains all necessary machines in
an appropriate order. Subsequent `fit!` calls do nothing unless: (i)
`force=true`, or (ii) some machine on which `mach` depends has
computed a new fit-result since `mach` last computed its fit-result,
or (iii) the specified `rows` have changed since the last time a
fit-result was last computed, or (iv) `mach` is stale (see below). In
cases (i), (ii) or (iii), `MLJBase.fit` is called. Otherwise
`MLJBase.update` is called.

A machine `mach` is *stale* if `mach.model` has changed since the last
time a fit-result was computed, or if if one of its training arguments
is `stale`. A node `N` is stale if `N.machine` is stale or one of its
arguments is stale. Source nodes are never stale.

Note that a nodal machine obtains its training data by *calling* its
node arguments on the specified `rows` (rather than *indexing* its arguments
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

    if rows === nothing
        rows = (:)
    end

    rows_have_changed = !isdefined(mach, :previous_rows) ||
        rows != mach.previous_rows

    if mach isa NodalMachine
        # determine if concrete data to be used in training may have changed:
        upstream_state = Tuple([state(arg) for arg in mach.args])
        data_has_changed =
            rows_have_changed || (upstream_state != mach.upstream_state)
        previously_fit = (mach.state > 0)
        args = [arg(rows=rows) for arg in mach.args]
    else
        data_has_changed = rows_have_changed
        previously_fit = isdefined(mach, :fitresult)
        args = [selectrows(arg, rows) for arg in mach.args]
    end

    if !previously_fit || data_has_changed || force
        # fit the model:
        verbosity < 1 || @info "Training $mach."
        mach.fitresult, mach.cache, mach.report =
            fit(mach.model, verbosity, args...)

    elseif !is_stale(mach)
        # don't fit the model
        if verbosity > 0
            @info "Not retraining $mach.\n It appears up-to-date. " *
                  "Use `force=true` to force retraining."
        end
        return mach
    else
        # update the model:
        verbosity < 1 || @info "Updating $mach."
        mach.fitresult, mach.cache, mach.report =
            update(mach.model, verbosity, mach.fitresult, mach.cache, args...)

    end

    if rows_have_changed
        mach.previous_rows = deepcopy(rows)
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

params(mach::AbstractMachine) = params(mach.model)
report(mach::AbstractMachine) = mach.report
