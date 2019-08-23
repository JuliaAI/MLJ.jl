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
        if !(_info[:is_wrapper]) &&
            !(M in [DeterministicNetwork, ProbabilisticNetwork])
            modelname = _info[:name]
            info_given_model[modelname] = _info
        end
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
