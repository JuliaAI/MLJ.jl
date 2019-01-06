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
    d = Dict{String,String}()
    d["package name"] = "MLJ"
    d["package uuid"] = ""
    d["properties"] = String[]
    d["is_pure_julia"] = "yes"
    d["operations"] = ["predict"]
    d["inputs_can_be"] = ["numeric, nominal, missing"]
    d["outputs_are"] = ["numeric", "deterministic", "univariate"]
    return d
end

## THE CONSTANT CLASSIFIER

# fit-result type:
R{L} = MLJ.UnivariateNominal{L,Float64}

struct ConstantClassifier{L} <: MLJ.Supervised{R{L}}
    target_type::Type{L}
end
ConstantClassifier(;target_type=Bool) = ConstantClassifier{target_type}(target_type)

function MLJ.fit(model::ConstantClassifier{L},
                 verbosity,
                 X,
                 y::CategoricalVector{L2,R,L2_pure}) where {L,R,L2,L2_pure}

    L == L2 || error("Model specifies target_type=$L but target type is $L2.")

    y_pure = skipmissing(y) |> collect
    N = length(y_pure)
    count_given_label = countmap(y_pure)
    prob_given_label = Dict{L2_pure,Float64}()
    for (x, c) in count_given_label
        prob_given_label[x] = c/N
    end
    
    fitresult = MLJ.UnivariateNominal(prob_given_label)

    verbosity < 1 || @info "probabilities: \n$prob_given_label"
    cache = nothing
    report = nothing

    return fitresult, cache, report

end

function MLJ.predict(model::ConstantClassifier{L}, fitresult, Xnew) where L
    return fill(fitresult, MLJ.nrows(Xnew))
end

function MLJ.predict_mode(model::ConstantClassifier{L}, fitresult, Xnew) where L
    m = mode(fitresult)
    labels = fitresult.prob_given_label |> keys |> collect
    N = MLJ.nrows(Xnew)    
    
    # to get a categorical array with all the levels we append the 
    # distribution labels to the prediction vector and truncate afterwards:
    yhat = vcat(fill(m, N), labels) |> categorical
    return yhat[1:N]
end


# metadata:
function MLJ.metadata(::Type{ConstantClassifier})
    d = Dict{String,String}()
    d["package name"] = "MLJ"
    d["package uuid"] = ""
    d["is_pure_julia"] = "yes"
    d["properties"] = String[]
    d["operations"] = ["predict"]
    d["inputs_can_be"] = ["numeric", "nominal", "missing"]
    d["outputs_are"] = ["nominal", "multiclass", "deterministic", "univariate"]
    return d
end

end # module


## EXPOSE THE INTERFACE

using .Constant
