## FUNCTIONS TO INSPECT METADATA OF REGISTERED MODELS AND TO
## FACILITATE MODEL SEARCH

# Notes:

# - A "handle" (defined in src/metadata.jl) is a named tuple like
#   `(name="KMeans", pkg="Clustering"). 

# - `Handle` is an alias for such tuples.

# - `Handle(name::String)` returns the appropriate handle, with
#    pkg=missing in the case of duplicate names. See also `model`
#    below, exported for use by user.

#

# The following checks the model/pkg combination is registered and
# throws an error if `pkg` is not specified and `name` is a
# duplicate. Intended for the user, in place of the `Handle`
# constructors which do no checks and throw no errors.
"""
    model(name::String; pkg=nothing)

Converts the specified `name` of a model into a named tuple
`(name=name, pkg=pkg)` where the tuple value `pkg` is the package
containing model with name `name` if the key-word argument `pkg` is 
unspecified. 

"""
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
        haskey(INFO_GIVEN_HANDLE, handle) ||
            throw(ArgumentError("$handle does not exist in the registry. \n"*
                  "Use models() to list all models. "))
    end
    return handle
end

model(handle::Handle; args...) = handle

# convert the dictionary obtained from an MLJBase.info call into a
# named tuple:
function _astuple(i)
    name = i[:name]
    package_name = i[:package_name]
    package_uuid = i[:package_uuid]
    package_url = i[:package_url]
    load_path = i[:load_path]
    is_wrapper = i[:is_wrapper]
    is_pure_julia = i[:is_pure_julia]
    input_scitype = i[:input_scitype]
    if i[:is_supervised]
        package_license = i[:package_license]
        supports_weights = i[:supports_weights]
        is_supervised = true
        is_probabilistic = i[:is_probabilistic]
        target_scitype = i[:target_scitype]
        return NamedTuple{(:name,
                           :package_name,
                           :package_uuid,
                           :package_url,
                           :package_license,
                           :load_path,
                           :is_wrapper,
                           :is_pure_julia,
                           :is_supervised,
                           :supports_weights,
                           :input_scitype,
                           :target_scitype,
                           :is_probabilistic)}((
                               name,
                               package_name,
                               package_uuid,
                               package_url,
                               package_license,
                               load_path,
                               is_wrapper,
                               is_pure_julia,
                               is_supervised,
                               supports_weights,
                               input_scitype,
                               target_scitype,
                               is_probabilistic))
    else
        is_supervised = false
        output_scitype = i[:output_scitype]
        return NamedTuple{(:name,
                           :package_name,
                           :package_uuid,
                           :package_url,
#                           :package_license,
                           :load_path,
                           :is_wrapper,
                           :is_pure_julia,
                           :is_supervised,
                           :input_scitype,
                           :output_scitype)}((
                               name,
                               package_name,
                               package_uuid,
                               package_url,
#                               package_license,
                               load_path,
                               is_wrapper,
                               is_pure_julia,
                               is_supervised,
                               input_scitype,
                               output_scitype))
    end
end    


"""
   traits(model::Model)

Return the traits associated with the specified `model`.

   traits(name::String, pkg=nothing)
   traits((name=name, pkg=pkg))

In the first instance, return model traits, given only the `name` of
the model type, even it is not in scope, but assuming it is
registered. If more than one package implements a model type with that
name, then the keyword argument `pkg::String` is required, or the
second form of the method used.

"""
function traits(handle::Handle)
    haskey(INFO_GIVEN_HANDLE, handle) ||
        throw(ArgumentError("$handle does not exist in the registry. \n"*
                            "Use models() to list all models. "))
    return _astuple(INFO_GIVEN_HANDLE[handle])
end
traits(name::String; pkg=nothing) = traits(model(name, pkg=pkg))
traits(M::Type{<:Model}) = _astuple(MLJBase.info(M))
traits(model::Model) = traits(typeof(model))

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

    task(model) = traits(model).is_supervised && traits(model).is_probabilistic

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
        t = traits(handle)
        return t.is_supervised &&
            t.is_wrapper == false &&
            task.target_scitype <: t.target_scitype &&
            task.input_scitype <: t.input_scitype &&
            task.is_probabilistic == t.is_probabilistic
    end
    return models(condition)
end

function models(task::UnsupervisedTask)
    ret = Dict{String, Any}()
    function condition(handle)
        t = traits(handle)
        return t.is_wrapper == false &&
            task.input_scitype <: t.input_scitype
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
    modeltypes = localmodeltypes(mod)
    names = map(modeltypes) do M
        traits(M).name
    end
    return filter(models(args...)) do handle
        handle.name in names
    end
end
