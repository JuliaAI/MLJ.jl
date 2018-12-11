# this file defines *and* loads one module

module Constant

export ConstantRegressor, ConstantClassifier

import MLJ
using StatsBase
using Statistics
using CategoricalArrays


## THE CONSTANT REGRESSOR

struct ConstantRegressor{F} <: MLJ.Supervised{F}
    target_type::Type{F}
end
ConstantRegressor(;target_type=Float64) = ConstantRegressor(target_type)

function MLJ.fit(model::ConstantRegressor{F}, verbosity, X, y::Vector{F2}) where {F,F2}
    F == F2 || error("Model specifies target_type=$F but target type is $F2.")
    fitresult = mean(skipmissing(y))
    verbosity < 1 || @info "Mean of target = $fitresult."
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

MLJ.predict(model::ConstantRegressor, fitresult, Xnew) = fill(fitresult, MLJ.nrows(Xnew))

# metadata:
function MLJ.metadata(::Type{ConstantRegressor})
    d = Dict()
    d["package name"] = "MLJ"
    d["package uuid"] = ""
    d["properties"] = []
    d["operations"] = ["predict"]
    d["inputs_can_be"] = ["numeric, nominal, missing"]
    d["outputs_are"] = ["numeric"]
    return d
end

## THE CONSTANT CLASSIFIER

R{T} = Union{CategoricalString{UInt32},CategoricalValue{T,UInt32}}

struct ConstantClassifier{T} <: MLJ.Supervised{R{T}}
    target_type::Type{T}
end
ConstantClassifier(;target_type=Bool) = ConstantClassifier{target_type}(target_type)

function MLJ.fit(model::ConstantClassifier{T},
                 verbosity,
                 X,
                 y::CategoricalVector{T2}) where {T,T2}

    T == T2 || error("Model specifies target_type=$T but target type is $T2.")
    fitresult = StatsBase.mode(collect(skipmissing(y)))
    verbosity < 1 || @info "target mode = $fitresult."
    cache = nothing
    report = nothing

    return fitresult, cache, report

end

function MLJ.predict(model::ConstantClassifier{T}, fitresult, Xnew) where T
    ref = fitresult.level
    refs = fill(ref, MLJ.nrows(Xnew))
    CategoricalArray{T,1}(refs, fitresult.pool)
end

# metadata:
function MLJ.metadata(::Type{ConstantClassifier})
    d = Dict()
    d["package name"] = "MLJ"
    d["package uuid"] = ""
    d["properties"] = []
    d["operations"] = ["predict"]
    d["inputs_can_be"] = ["numeric, nominal, missing"]
    d["outputs_are"] = ["nominal"]
    return d
end

end # module


## EXPOSE THE INTERFACE

using .Constant
