import Pkg

@static if VERSION >= v"1.4.0"
    function _get_package_version(name)
        for (uuid, dep) in Pkg.dependencies()
            if dep.name == name
                return dep.version
            end
        end
        return nothing
    end
else
    function _get_package_version(name)
        return get(Pkg.installed(), name, nothing)
    end
end

const MLJ_VERSION = _get_package_version("MLJ")
