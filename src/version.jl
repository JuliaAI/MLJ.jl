@static if VERSION >= v"1.4.0"
    function installed()
        @warn "Pkg.installed() is deprecated"
        deps = Pkg.dependencies()
        installs = Dict{String, VersionNumber}()
        for (uuid, dep) in deps
            dep.is_direct_dep || continue
            dep.version === nothing && continue
            installs[dep.name] = dep.version
        end
        return installs
    end
else
    const installed = Pkg.installed
end
const MLJ_VERSION = installed()["MLJ"]
