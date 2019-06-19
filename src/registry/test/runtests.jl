# using Revise
using Test

# brittle hack b/s of https://github.com/dmlc/XGBoost.jl/issues/58:
# using Pkg
# Pkg.add(PackageSpec(url="https://github.com/dmlc/XGBoost.jl"))

using Registry
using MLJBase

#@test !(isempty(Registry.metadata()))


