abstract type AbstractMachine{M} <: MLJType end


# TODO: write out separate method for machine(::Model, ::MLJTask) to simplify logic. 
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
                length(args) > 2
                error("Use machine(model, task) or machine(model, X, y) "*
                      "for a supervised model.")
            elseif length(args) == 2 && !(container_type(args[1]) in [:table, :sparse])
                error("The X in machine(model, X, y) should be a table. ")
            end
            if length(args) == 2
                X, y = args
                T =
                    input_is_multivariate(model) ? Union{scitypes(X)...} : scitype_union(X)
                T <: input_scitype_union(model) ||
                    error("The scitypes of elements of X, in machine(model, X, y), should be a subtype of $(input_scitype_union(model)). ")
                y isa Vector || y isa CategoricalVector ||
                    error("The y, in machine(model, X, y), should be a vector "*
                          "(of tuples for multivariate targets) or a categorical vector. ")
                scitype_union(y) <: target_scitype_union(model) || 
                    error("The scitype of elements of y, in machine(model, X, y), should be a subtype of $(target_scitype_union(model)). ")
            end
        end
        if M <: Unsupervised
            length(args) == 1 ||
                error("Wrong number of arguments. "*
                      "Use machine(model, X) or machine(model, task) for an unsupervised model.")

            container_type(args[1]) in [:table, :sparse] || args[1] isa UnsupervisedTask || args[1] isa Vector || args[1] isa CategoricalVector ||
                error("The X, in machine(model, X), should be a vector, categorical vector, table or UnsupervisedTask. "*
                      "Use MLJ.table(X) to wrap an abstract matrix X as a table. ")
            if container_type(args[1]) in [:table, :sparse]
                X = args[1]
                U =
                    input_is_multivariate(model) ?  Union{scitypes(X)...} : scitype_union(X)
                U <: input_scitype_union(model) || 
                error("The scitype of elements of X, in machine(model, X), should be a subtype of $(input_scitype_union(model)). ")
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

"""
    fit!(mach::Machine; rows=nothing, verbosity=1)

Train the machine `mach` using the algorithm and hyperparameters
specified by `mach.model`, using those rows of the wrapped data having
indices in `rows`.

    fit!(mach::NodalMachine; rows=nothing, verbosity=1)

A nodal machine is trained in the same way as a regular machine with
one difference: Instead of training the model on the wrapped data
*indexed* on `rows`, it is trained on the wrapped nodes *called* on
`rows`, with calling being a recursive operation on nodes within a
learning network.

"""
function fit!(mach::AbstractMachine; rows=nothing, verbosity=1,
force=false)

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

params(mach::AbstractMachine) = params(mach.model)
report(mach::AbstractMachine) = mach.report


