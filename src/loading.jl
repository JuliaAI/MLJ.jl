## FUNTIONS TO LOAD MODEL IMPLEMENTATION CODE

# Notes:

# - see top of src/model_search.jl for notes on "handles".

"""
    
    load_implementation(name::String; pkg=nothing, mod=Main, verbosity=1)
            
Load the model implementation code for the model with specified `name` into the module `mod`, specifying `pkg` in the case of duplicate names. 

See also [`@load`](@ref)

"""
function load_implementation(proxy::ModelProxy; mod=Main, verbosity=1)
    # get name, package and load path:
    name = proxy.name
    pkg = proxy.package_name
    handle = (name=name, pkg=pkg)
    path = INFO_GIVEN_HANDLE[handle][:load_path]
    path_components = split(path, '.')

    # decide what to print
    toprint = verbosity > 0

    # return if model is already loaded
    localnames = map(p->p.name, localmodels(mod=mod))
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


"""
    @load name pkg=nothing verbosity=0


Load the model implementation code for the model with specified `name`
into the module `mod`, specifying `pkg` in the case of duplicate
names. 

### Examples

    @load DecisionTreeeRegressor
    @load PCA verbosity=1
    @load SVC pkg=LIBSVM 

See also [`load_implementation`](@ref)

"""
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
