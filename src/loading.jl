## LOADING METADATA FOR EXTERNAL PACKAGE MODELS

const path_to_metadata_dot_toml = joinpath(srcdir, "../") # todo: make os independent
const remote_file =
    @RemoteFile "https://raw.githubusercontent.com/alan-turing-institute/MLJRegistry.jl/master/Metadata.toml" dir=path_to_metadata_dot_toml

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
const METADATA = decode_dic(TOML.parsefile(local_metadata_file))

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
        modelname = _info[:name]
        info_given_model[modelname] = _info
    end
    ret = deepcopy(METADATA)
    ret["MLJ"] = info_given_model
    return ret
end

"""
    models()

List all models, loaded or registered, as a dictionary indexed on package name.

    models(task)

List, in the same format, all models matching the specified `task`.

"""
function models()
    _models_given_pkg = Dict()
    meta = metadata()
    packages = keys(meta) |> collect
    for pkg in packages
        _models_given_pkg[pkg] = collect(keys(meta[pkg]))
    end
    return _models_given_pkg
end

function models(task::SupervisedTask)
    ret = Dict{String, Any}()
    allmodels = models()
    for pkg in keys(allmodels)
        models_in_pkg = 
            filter(allmodels[pkg]) do model
                info_ = info(model, pkg=pkg)
                info_[:is_supervised] &&
                    info_[:is_wrapper] == false && 
                    task.target_scitype <: info_[:target_scitype] &&
                    task.input_scitypes <: info_[:input_scitypes] &&
                    task.is_probabilistic == info_[:is_probabilistic]
            end
        isempty(models_in_pkg) || (ret[pkg] = models_in_pkg)
    end
    return ret
end


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

macro load(model_ex)
    model = string(model_ex)

    if model in string.(MLJBase.finaltypes(Model))
        @info "A model named \"$model\" is already loaded.\n"*
        "Nothing new loaded. "
        return
    end

    pkg, success = try_to_get_package(model)
    if !success # pkg is then a message
        error(pkg*"Use @load $model pkg=...")
    end

    # get load path:
    info = METADATA[pkg][model]
    path = info[:load_path]
    path_components = split(path, '.')

    # if needed, put MLJModels in the calling module's namespace (it
    # is already loaded into MLJ's namespace):
    if path_components[1] == "MLJModels"
        print("import MLJModels ")
        __module__.eval(:(import MLJModels))
        println('\u2714')
    end

    # load the package (triggering lazy-load of implementation code if
    # this is in MLJModels):
    pkg_ex = Symbol(pkg)
    print("import $pkg_ex ")
    __module__.eval(:(import $pkg_ex))
    println('\u2714')

    # load the model:
    load_ex = Meta.parse("import $path")
    print(string(load_ex, " "))
    __module__.eval(load_ex)
    println('\u2714')

end


# TODO: elminate duplicate code (above and below are same after "# get
# load path:"

macro load(model_ex, pkg_ex)
    model = string(model_ex)
    pkg = string(pkg_ex.args[2])

    # get load path:
    info = METADATA[pkg][model]
    path = info[:load_path]
    path_components = split(path, '.')

    # if needed, put MLJModels in the calling module's namespace (it
    # is already loaded into MLJ's namespace):
    if path_components[1] == "MLJModels"
        print("import MLJModels ")
        __module__.eval(:(import MLJModels))
        println('\u2714')
    end

    # load the package (triggering lazy-load of implementation code if
    # this is in MLJModels):
    pkg_ex = Symbol(pkg)
    print("import $pkg_ex ")
    __module__.eval(:(import $pkg_ex))
    println('\u2714')

    # load the model:
    load_ex = Meta.parse("import $path")
    print(string(load_ex, " "))
    __module__.eval(load_ex)
    println('\u2714')

end


    
