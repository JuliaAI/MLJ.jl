## FUNCTIONS TO INSPECT METADATA OF REGISTERED MODELS AND TO
## FACILITATE MODEL SEARCH

# Notes:

# - A "handle" (defined in src/metadata.jl) is a named tuple like
#   `(name="KMeans", pkg="Clustering"). 

# - `Handle` is an alias for such tuples.

# - `Handle(name::String)` returns the appropriate handle, with
#    pkg=missing in the case of duplicate names. See also `model`
#    below, exported for use by user.

function model(name::String; pkg=nothing)
    name in NAMES ||
        throw(ArgumentError("There is no model named \"$name\" in "*
                            "the registry. \n Run `models()` to view all "*
                            "registered models."))
    if pkg == nothing
        handle  = Handle(name)
        if ismissing(handle.pkg)
            pkgs = PKGS_GIVEN_NAME[name]
            message = "Ambiguous model name. Use pkg=...\n"*
            "The model $model is provided by these packages: $pkgs.\n"
            throw(ArgumentError(message))
        end
    else
        handle = Handle(name, pkg)
        handle in keys(INFO_GIVEN_HANDLE) ||
            throw(ArgumentError("$handle does not exist in the registry. \n"*
                  "Use models() to list all models. "))
    end
    return handle
end

"""
   info(model::Model)

Return the dictionary of metadata associated with the specified
`model`.

   info(name::String, pkg=nothing)
   info((name=name, pkg=pkg))

In the first instance, return the same dictionary given only the
`name` of the model type (which does not need to be loaded). If more
than one package implements a model with that name then some
`pkg::String` will need to be specified, or the second form used.

"""
MLJBase.info(handle::Handle) = INFO_GIVEN_HANDLE[handle]
MLJBase.info(name::String; pkg=nothing) = info(model(name, pkg=pkg))
    
"""
    models()

List all models in the MLJ registry. Here and below, a *model* is any
named tuple of strings of the form `(name=..., pkg=...)`.

    models(task::MLJTask)

List all models matching the specified `task`. 

    models(condition)

List all models matching a given condition. A *condition* is any
`Bool`-valued function on models.

Excluded in the listings are the built-in model-wraps `EnsembleModel`,
`TunedModel`, and `IteratedModel`.

### Example

If

    task(model) = info(model)[:is_supervised] && info(model)[:is_probabilistic]

then `models(task)` lists all supervised models making probabilistic
predictions.

See also: [`localmodels`](@ref).

"""
models(condition) =
    sort!(filter(condition, keys(INFO_GIVEN_HANDLE) |> collect))

models() = models(x->true)

function models(task::SupervisedTask)
    ret = Dict{String, Any}()
    function condition(handle)
        i = info(handle)
        return i[:is_supervised] &&
            i[:is_wrapper] == false &&
            task.target_scitype <: i[:target_scitype] &&
            task.input_scitype <: i[:input_scitype] &&
            task.is_probabilistic == i[:is_probabilistic]
    end
    return models(condition)
end

function models(task::UnsupervisedTask)
    ret = Dict{String, Any}()
    function condition(handle)
        i  = info(handle)
        return i[:is_wrapper] == false &&
            task.input_scitype <: i[:input_scitype]
    end
    return models(condition)
end

"""
    localmodels(; mod=Main)
    localmodels(task::MLJTask; mod=Main)
    localmodels(condition; mod=Main)
 

List all models whose names are in the namespace of the specified
module `mod`, additionally solving the `task`, or meeting the
`condition`, if specified.

See also [models](@ref)

"""
function localmodels(args...; mod=Main)
    localmodels = filter(MLJBase.finaltypes(Model)) do model
        name = info(model)[:name]
        isdefined(mod, Symbol(name))
    end
    localnames = map(localmodels) do model
        info(model)[:name]
    end
    return filter(models(args...)) do handle
        handle.name in localnames
    end
end
