## AN UNIQUE IDENTIFIER FOR REGISTERED MODELS

# struct Handle
#     name::String
#     pkg::Union{String,Missing}
# end
# Base.show(stream::IO,  h::Handle) =
#     print(stream, "\"$(h.name)\"\t (from \"$(h.pkg)\")")

Handle = NamedTuple{(:name, :pkg), Tuple{String,String}}
(::Type{Handle})(name,string) = Handle((name, string))

function Base.isless(h1::Handle, h2::Handle)
    if isless(h1.name, h2.name)
        return true
    elseif h1.name == h2.name
        return isless(h1.pkg, h2.pkg)
    else
        return false
    end
end
 

## FUNCTIONS TO BUILD GLOBAL METADATA CONSTANTS IN MLJ INITIALIZATION

# for use in __init__ to define INFO_GIVEN_HANDLE
function info_given_handle(metadata_file)

    # get the metadata for MLJ models:
    metadata = LittleDict(TOML.parsefile(metadata_file))
    localmodels = MLJBase.finaltypes(Model)
    info_given_model = Dict()
    for M in localmodels
        _info = MLJBase.info(M)
        modelname = _info[:name]
        info_given_model[modelname] = _info
    end

    # merge with the decoded external metadata:
    metadata_given_pkg = decode_dic(metadata)
    metadata_given_pkg["MLJ"] = info_given_model

    # build info_given_handle dictionary:
    ret = Dict{Handle}{Any}()
    packages = keys(metadata_given_pkg)
    for pkg in packages
        info_given_name = metadata_given_pkg[pkg]
        for name in keys(info_given_name)
            handle = Handle(name, pkg)
            ret[handle] = info_given_name[name]
        end
    end
    return ret
    
end

# for use in __init__ to define AMBIGUOUS_NAMES
function ambiguous_names(info_given_handle)
    names_with_duplicates = map(keys(info_given_handle) |> collect) do handle
        handle.name
    end
    frequency_given_name = countmap(names_with_duplicates)
    return filter(keys(frequency_given_name) |> collect) do name
        frequency_given_name[name] > 1
    end
end

# for use in __init__ to define PKGS_GIVEN_NAME
function pkgs_given_name(info_given_handle)
    handles = keys(info_given_handle) |> collect
    ret = Dict{String,Vector{String}}()
    for handle in handles
        if haskey(ret, handle.name)
           push!(ret[handle.name], handle.pkg)
        else
            ret[handle.name] =[handle.pkg, ]
        end
    end
    return ret
end

# for use in __init__ to define NAMES
function model_names(info_given_handle)
    names_with_duplicates = map(keys(info_given_handle) |> collect) do handle
        handle.name
    end
    return unique(names_with_duplicates)
end

function (::Type{Handle})(name::String)
    if name in AMBIGUOUS_NAMES
        return Handle(name, missing)
    else
        return Handle(name, first(PKGS_GIVEN_NAME[name]))
    end
end

function model(name::String; pkg=nothing)
    name in NAMES ||
        throw(ArgumentError("There is no model named \"$name\" in the registry. "))
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


## MACROS TO LOAD MODELS

function load_implementation(handle::Handle; mod=Main, verbosity=1)
    # get name, package and load path:
    info = INFO_GIVEN_HANDLE[handle]
    path = info[:load_path]
    path_components = split(path, '.')
    name = handle.name
    pkg = handle.pkg

    # decide what to print
    toprint = verbosity > 0

    # return if model is already loaded
    localnames = map(handle->handle.name, localmodels(mod=mod))
    if name ∈ localnames
        @info "A model named \"$name\" is already loaded. \n"*
        "Nothing new loaded. "
        return
    end

    toprint && @info "Loading into module \"$mod\": "
    
    # if needed, put MLJModels in the calling module's namespace (it
    # is already loaded into MLJ's namespace):
    if path_components[1] == "MLJModels"
        toprint && print("import MLJModels ")
        mod.eval(:(import MLJModels))
        toprint && println('\u2714')
    end

    # load the package (triggering lazy-load of implementation code if
    # this is in MLJModels):
    pkg_ex = Symbol(pkg)
    toprint && print("import $pkg_ex ")
    mod.eval(:(import $pkg_ex))
    toprint && println('\u2714')

    # load the model:
    load_ex = Meta.parse("import $path")
    toprint && print(string(load_ex, " "))
    mod.eval(load_ex)
    toprint && println('\u2714')

    nothing
end

load_implementation(name::String; pkg=nothing, kwargs...) =
    load_implementation(model(name, pkg=pkg); kwargs...)

macro load(name_ex, kw_exs...)
    name_ = string(name_ex)

    # parse kwargs:
    message = "Invalid @load syntax.\n "*
    "Sample usage: @load PCA pkg=\"MultivariateStats\" verbosity=1"
    for ex in kw_exs
        ex.head == :(=) || throw(ArgumentError(warning))
        variable_ex = ex.args[1]
        value_ex = ex.args[2]
        if variable_ex == :pkg
            pkg = string(value_ex)
        elseif variable_ex == :verbosity
            verbosity = value_ex
        else
            throw(ArgumentError(warning))
        end
    end
    (@isdefined pkg) || (pkg = nothing)
    (@isdefined verbosity) || (verbosity = 0)
                
    # get rid brackets in name_, as in
    # "(MLJModels.Clustering).KMedoids":
    name = filter(name_) do c !(c in ['(',')']) end

    load_implementation(name, mod=__module__, pkg=pkg, verbosity=verbosity)
end

