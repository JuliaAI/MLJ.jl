# to be depreciated:
const FiniteOrderedFactor = OrderedFactor


## LOADING METADATA FOR EXTERNAL PACKAGE MODELS

const path_to_metadata_dot_toml = joinpath(srcdir, "../") # todo: make os independent
const remote_file =
    @RemoteFile "https://raw.githubusercontent.com/alan-turing-institute/MLJRegistry.jl/dev/Metadata.toml" dir=path_to_metadata_dot_toml

const local_metadata_file = joinpath(path_to_metadata_dot_toml, "Metadata.toml")

# update locally archived Metadata.toml:
try 
    download(remote_file, quiet=true, force=true)
catch 
    @info "Unable to update model metadata from github.alan-turing-institute/MLJRegistry. "*
    "Using locally archived metadata. "
end

# metadata for models in external packages (`decode_dic` restores
# symbols from string representations):
const METADATA = TOML.parsefile(local_metadata_file)

"""
   info(model, pkg=nothing)

Return the dictionary of metadata associated with `model::String`. If
more than one package implements `model` then `pkg::String` will need
to be specified.

"""
function MLJBase.info(model::String; pkg=nothing)
    if pkg == nothing
        if model in string.(MLJBase.finaltypes(Model))
            pkg = "MLJ"
        else
            pkg, success = try_to_get_package(model)
            if !success 
                error(pkg*"Use info($model, pkg=...)")
            end
        end
    end
    return metadata()[pkg][model]
end


## FUNCTIONS TO RETRIEVE MODELS AND METADATA

function metadata()
    modeltypes = MLJBase.finaltypes(Model)
    info_given_model = Dict()
    for M in modeltypes
        _info = MLJBase.info(M)
        modelname = string(M) #_info[:name]
        info_given_model[modelname] = _info
    end
    ret = decode_dic(METADATA)
    ret["MLJ"] = info_given_model
    return ret
end

"""
    models()

List all model as a dictionary indexed on package name`. Models
available for immediate use appear under the key "MLJ".

    models(conditional)

Restrict results to package model pairs `(m, p)` satisfying
`conditional(info(m, pkg=p)) == true`.

    models(task::MLJTask)

List all models matching the specified `task`. 

### Example

To retrieve all proababilistic classifiers:

    models(x -> x[:is_supervised] && x[:is_probabilistic]==true)

See also: localmodels

"""
function models(conditional)
    _models_given_pkg = Dict()
    meta = metadata()
    packages = keys(meta) |> collect
    for pkg in packages
        _models = filter(keys(meta[pkg]) |> collect) do model
            conditional(info(model, pkg=pkg)) 
        end
        isempty(_models) || (_models_given_pkg[pkg] = _models)
    end
    return _models_given_pkg
end

models() = models(x->true)

function models(task::SupervisedTask; kwargs...)
    ret = Dict{String, Any}()
    conditional(x) =
        x[:is_supervised] &&
        x[:is_wrapper] == false && 
        task.target_scitype_union <: x[:target_scitype_union] &&
        task.input_scitype_union <: x[:input_scitype_union] &&
        task.is_probabilistic == x[:is_probabilistic] &&
        task.input_is_multivariate == x[:input_is_multivariate]
    return models(conditional, kwargs...)
end

function models(task::UnsupervisedTask; kwargs...)
    ret = Dict{String, Any}()
    conditional(x) =
        x[:is_wrapper] == false && 
        task.input_scitype_union <: x[:input_scitype_union] &&
        task.input_is_multivariate == x[:input_is_multivariate] &&
    return models(conditional, kwargs...)
end

"""
    localmodels()

List all models available for immediate use. Equivalent to
`models()["MLJ"]`. Can also be given a condition function or task as
argument. See `models`.

"""
localmodels(; kwargs...) = models(; kwargs...)["MLJ"]
localmodels(arg; kwargs...) = models(arg; kwargs...)["MLJ"]


## MACROS TO LOAD MODELS

# returns (package, true) if model is in exaclty one package,
# otherwise (message, false):
function try_to_get_package(model::String)
    _models_given_pkg = models()

    # set-valued version (needed to compute inverse dictionary):
    models_given_pkg = Dict{String, Set{String}}()
    for pkg in keys(_models_given_pkg)
        models_given_pkg[pkg] = Set(_models_given_pkg[pkg])
    end

    # get inverse:
    pkgs_given_model = inverse(models_given_pkg)

    pkgs = pkgs_given_model[model]
    if length(pkgs) > 1
        message = "Ambiguous model specification. \n"*
        "The model $model is provided by these packages: $pkgs.\n"
        return (message, false)
    end
    return (pop!(pkgs), true)
end

function _load(model, pkg, mdl)
    # get load path:
    info = decode_dic(METADATA)[pkg][model]
    path = info[:load_path]
    path_components = split(path, '.')

    # if needed, put MLJModels in the calling module's namespace (it
    # is already loaded into MLJ's namespace):
    if path_components[1] == "MLJModels"
        print("import MLJModels ")
        mdl.eval(:(import MLJModels))
        println('\u2714')
    end

    # load the package (triggering lazy-load of implementation code if
    # this is in MLJModels):
    pkg_ex = Symbol(pkg)
    print("import $pkg_ex ")
    mdl.eval(:(import $pkg_ex))
    println('\u2714')

    # load the model:
    load_ex = Meta.parse("import $path")
    print(string(load_ex, " "))
    mdl.eval(load_ex)
    println('\u2714')
end

macro load(model_ex)
    model_ = string(model_ex)

    # get rid brackets, as in "(MLJModels.Clustering).KMedoids":
    model = filter(model_) do c !(c in ['(',')']) end

    if model in string.(MLJBase.finaltypes(Model))
        @info "A model named \"$model\" is already loaded.\n"*
        "Nothing new loaded. "
        return
    end

    pkg, success = try_to_get_package(model)
    if !success # pkg is then a message
        error(pkg*"Use @load $model pkg=\"PackageName\". ")
    end

    _load(model, pkg, __module__)
end

macro load(model_ex, pkg_ex)
    model = string(model_ex)
    pkg = string(pkg_ex.args[2])

    _load(model, pkg, __module__)
end


    
