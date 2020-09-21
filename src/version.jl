import Pkg

using Pkg.TOML 
 const MLJ_VERSION =
     VersionNumber(TOML.parsefile(joinpath(dirname(@__DIR__),
                                           "Project.toml"))["version"])

