## LOADING METADATA FOR EXTERNAL PACKAGE MODELS

const METADATA = MLJRegistry.metadata()

# merge with the metadata for models defined in MLJ.jl:
modeltypes = MLJRegistry.finaltypes(MLJBase.Model)
for M in modeltypes
    _info = MLJBase.info(M)
    modelname = _info[:name]
    meta_given_package["MLJ"][modelname] = _info
end


## GET LIST OF MODELS IN EACH PACKAGE (INCL MLJ):

_models_given_pkg = Dict()
for pkg in packages
    _models_given_pkg[pkg] = collect(keys(METADATA[pkg]))
end

# set-valued version (needed to compute inverse dictionary):
const models_given_pkg = Dict{String, Set{String}}()
for pkg in keys(_models_given_pkg)
    models_given_pkg[pkg] = Set(_models_given_pkg[pkg])
end

# inverse:
const pkgs_given_model = inverse(models_given_pkg)


## MACROS TO LOAD MODELS

# returns (package, true) if model is unique, otherwise (message, false):
function try_to_get_package(model::String)
    pkgs = deepcopy(pkgs_given_model[model]) # will be popping elements later
    if length(pkgs) > 1
        message = "Ambiguous model specification. \n"*
        "The model $model is provided by these packages: $pkgs.\n"
        return (message, false)
    end
    return (pop!(pkgs), true)
end

macro load(model_ex)
    model = string(model_ex)

    pkg, success = try_to_get_package(model)
    if !success
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


"""
    models()

List the names of all registered MLJ models, as a dictionary indexed on package name.
"""
models() = _models_given_pkg

function MLJBase.info(model::String)
    pkg, success = try_to_get_package(model)
    if !success
        error(pkg*"Use info($model, pkg=...)")
    end
    return METADATA[pkg][model]
end

    
