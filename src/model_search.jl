## FUNCTIONS TO INSPECT METADATA OF REGISTERED MODELS AND TO
## FACILITATE MODEL SEARCH

is_supervised(::Type{<:Supervised}) = true
is_supervised(::Type{<:Unsupervised}) = false

supervised_propertynames =
    sort!(collect(keys(MLJBase.info(ConstantRegressor()))))
alpha = [:name, :package_name, :is_supervised]
omega = [:input_scitype, :target_scitype]
both = vcat(alpha, omega)
filter!(!in(both), supervised_propertynames) 
prepend!(supervised_propertynames, alpha)
append!(supervised_propertynames, omega)
const SUPERVISED_PROPERTYNAMES = Tuple(supervised_propertynames)

unsupervised_propertynames =
    sort!(collect(keys(MLJBase.info(FeatureSelector()))))
alpha = [:name, :package_name, :is_supervised]
omega = [:input_scitype, :output_scitype]
both = vcat(alpha, omega)
filter!(!in(both), unsupervised_propertynames) 
prepend!(unsupervised_propertynames, alpha)
append!(unsupervised_propertynames, omega)
const UNSUPERVISED_PROPERTYNAMES = Tuple(unsupervised_propertynames)

ModelProxy = Union{NamedTuple{SUPERVISED_PROPERTYNAMES},
                   NamedTuple{UNSUPERVISED_PROPERTYNAMES}}

function Base.isless(p1::ModelProxy, p2::ModelProxy)
    if isless(p1.name, p2.name)
        return true
    elseif p1.name == p2.name
        return isless(p1.package_name, p2.package_name)
    else
        return false
    end
end

Base.show(stream::IO, p::ModelProxy) =
    print(stream, "(name = $(p.name), package_name = $(p.package_name), "*
          "... )")

# returns named tuple version of the dictionary i=info(SomeModelType):
function _model(i) 
    propertynames = ifelse(i[:is_supervised], SUPERVISED_PROPERTYNAMES,
                           UNSUPERVISED_PROPERTYNAMES)
    propertyvalues = Tuple(i[property] for property in propertynames)
    return NamedTuple{propertynames}(propertyvalues)
end

model(handle::Handle) = _model(INFO_GIVEN_HANDLE[handle])
    
"""
    model(name::String; pkg=nothing)

Returns the metadata for the registered model type with specified
`name`. The key-word argument `pkg` is required in the case of
duplicate names.

"""
function model(name::String; pkg=nothing)
    name in NAMES ||
        throw(ArgumentError("There is no model named \"$name\" in "*
                            "the registry. \n Run `models()` to view all "*
                            "registered models."))
    # get the handle:
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
    return model(handle)

end


"""
   traits(model::Model)

Return the traits associated with the specified `model`. Equivalent to
`model(name; pkg=pkg)` where `name` is the name of the model type, and
`pkg` the name of the package containing it.
 
"""
traits(M::Type{<:Model}) = _model(MLJBase.info(M))
traits(model::Model) = traits(typeof(model))
model(M::Type{<:Model}) = traits(M)
model(model::Model) = traits(model)

"""
    models()

List all models in the MLJ registry. Here and below *model* means the
registry metadata entry for a genuine model type (a proxy for types
that may not be loaded).

    models(task::MLJTask)

List all models matching the specified `task`. 

    models(conditions...)

List all models satisifying the specified `conditions`. A *condition*
is any `Bool`-valued function on models.

Excluded in the listings are the built-in model-wraps `EnsembleModel`,
`TunedModel`, and `IteratedModel`.

### Example

If

    task(model) = model.is_supervised && model.is_probabilistic

then `models(task)` lists all supervised models making probabilistic
predictions.

See also: [`localmodels`](@ref).

"""
function models(conditions...)
    unsorted = filter(model.(keys(INFO_GIVEN_HANDLE))) do model
        all(c(model) for c in conditions)
    end
    return sort!(unsorted)
end

models() = models(x->true)

function models(task::SupervisedTask)
    ret = Dict{String, Any}()
    function condition(t)
        return t.is_supervised &&
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
        return task.input_scitype <: t.input_scitype
    end
    return models(condition)
end

"""
    localmodels(; mod=Main)
    localmodels(task::MLJTask; mod=Main)
    localmodels(conditions...; mod=Main)
 

List all models whose names are in the namespace of the specified
module `mod`, additionally solving the `task`, or meeting the
`conditions`, if specified. A *condition* is a `Bool`-valued function
on models.

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
