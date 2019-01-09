# this file defines *and* loads one module

module Constant

export ConstantRegressor, ConstantClassifier

import MLJBase
import MLJ # needed for `nrows` 
import Distributions
using StatsBase
using Statistics
using CategoricalArrays


## THE CONSTANT REGRESSOR

"""
    ConstantRegressor(; target_type=Float64, distribution_type=Distributions.Normal)

A regressor that, for any new input pattern, predicts the univariate
probability distribution best fitting the training target data. Use
`predict_mean` to predict the mean value instead.

"""
struct ConstantRegressor{F,D} <: MLJBase.Probabilistic{D}
    target_type::Type{F}
    distribution_type::Type{D}
end
ConstantRegressor(;target_type=Float64, distribution_type=Distributions.Normal) =
    ConstantRegressor(target_type, distribution_type)

function MLJBase.fit(model::ConstantRegressor{F,D}, verbosity, X, y::Vector{F2}) where {F,D,F2}
    F == F2 || error("Model specifies target_type=$F but target type is $F2.")
    fitresult = Distributions.fit(D, collect(skipmissing(y)))
    verbosity < 1 || @info "Fitted a constant probability distribution, $fitresult."
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

MLJBase.predict(model::ConstantRegressor, fitresult, Xnew) = fill(fitresult, MLJ.nrows(Xnew))
MLJBase.predict_mean(model::ConstantRegressor, fitresult, Xnew) = fill(Distributions.mean(fitresult), MLJ.nrows(Xnew))

# metadata:
function MLJBase.info(::Type{ConstantRegressor})
    d = Dict{String,String}()
    d["package name"] = "MLJ"
    d["package uuid"] = ""
    d["properties"] = String[]
    d["is_pure_julia"] = "yes"
    d["operations"] = ["predict","predict_mean"]
    d["inputs_can_be"] = ["numeric, nominal, missing"]
    d["outputs_are"] = ["numeric", "probabilistic", "univariate"]
    return d
end


## THE CONSTANT CLASSIFIER

# fit-result type:
R{L} = MLJBase.UnivariateNominal{L,Float64}

"""
    ConstantClassifier(; target_type=Bool)

A classifier that, for any new input pattern, `predict`s the
`UnivariateNominal` probability distribution `d` best fitting the
training target data. So, `pdf(d, label)` is the proportion of labels
in the training data coinciding with `label`. Use `predict_mode` to
obtain the training target mode instead.

"""
struct ConstantClassifier{L} <: MLJBase.Probabilistic{R{L}}
    target_type::Type{L}
end
ConstantClassifier(;target_type=Bool) = ConstantClassifier{target_type}(target_type)

function MLJBase.fit(model::ConstantClassifier{L},
                 verbosity,
                 X,
                 y::CategoricalVector{L2,R,L2_pure}) where {L,R,L2,L2_pure}

    L == L2 || error("Model specifies target_type=$L but target type is $L2.")

    # dump missing target values and make into a regular array:
    y_pure = Array{L2_pure}(skipmissing(y) |> collect)

    fitresult = Distributions.fit(MLJBase.UnivariateNominal, y_pure)

    verbosity < 1 || @info "probabilities: \n$(fitresult.prob_given_label)"
    cache = nothing
    report = nothing

    return fitresult, cache, report

end

function MLJBase.predict(model::ConstantClassifier{L}, fitresult, Xnew) where L
    return fill(fitresult, MLJ.nrows(Xnew))
end

function MLJBase.predict_mode(model::ConstantClassifier{L}, fitresult, Xnew) where L
    m = mode(fitresult)
    labels = fitresult.prob_given_label |> keys |> collect
    N = MLJ.nrows(Xnew)    
    
    # to get a categorical array with all the original levels we append the 
    # distribution labels to the prediction vector and truncate afterwards:
    yhat = vcat(fill(m, N), labels) |> categorical
    return yhat[1:N]
end


# metadata:
function MLJBase.info(::Type{ConstantClassifier})
    d = Dict{String,String}()
    d["package name"] = "MLJ"
    d["package uuid"] = ""
    d["is_pure_julia"] = "yes"
    d["properties"] = String[]
    d["operations"] = ["predict", "predict_mode"]
    d["inputs_can_be"] = ["numeric", "nominal", "missing"]
    d["outputs_are"] = ["nominal", "multiclass", "deterministic", "univariate"]
    return d
end

end # module


## EXPOSE THE INTERFACE

using .Constant
