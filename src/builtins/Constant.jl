# this file defines *and* loads one module

module Constant

export ConstantRegressor, ConstantClassifier
export DeterministicConstantRegressor, DeterministicConstantClassifier

import MLJBase
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
function ConstantRegressor(;target_type=Float64, distribution_type=Distributions.Normal{Float64})
    model = ConstantRegressor(target_type, distribution_type)
    message = clean!(model)
    isempty(message) || @warn message
    return model
end

function clean!(model::ConstantRegressor)
    message = ""
    MLJBase.isdistribution(model.distribution_type) ||
        error("$model.distribution_type is not a valid distribution_type.")
    return message
end

function MLJBase.fit(model::ConstantRegressor{F,D}, verbosity::Int, X, y::Vector{F2}) where {F,D,F2}
    F == F2 || error("Model specifies target_type=$F but target type is $F2.")
    fitresult = Distributions.fit(D, y)
    verbosity < 1 || @info "Fitted a constant probability distribution, $fitresult."
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

nrows(X) = MLJBase.schema(X).nrows


MLJBase.predict(model::ConstantRegressor, fitresult, Xnew) = fill(fitresult, nrows(Xnew))
MLJBase.predict_mean(model::ConstantRegressor, fitresult, Xnew) = fill(Distributions.mean(fitresult), nrows(Xnew))

# metadata:
MLJBase.load_path(::Type{<:ConstantRegressor}) = "MLJ.ConstantRegressor"
MLJBase.package_name(::Type{<:ConstantRegressor}) = "MLJ"
MLJBase.package_uuid(::Type{<:ConstantRegressor}) = ""
MLJBase.package_url(::Type{<:ConstantRegressor}) = "https://github.com/alan-turing-institute/MLJ.jl"
MLJBase.is_pure_julia(::Type{<:ConstantRegressor}) = :yes
MLJBase.input_kinds(::Type{<:ConstantRegressor}) = [:continuous, :multiclass, :ordered_factor_finite, :ordered_factor_infinite, :missing]
MLJBase.output_kind(::Type{<:ConstantRegressor}) = :continuous
MLJBase.output_quantity(::Type{<:ConstantRegressor}) = :univariate


## THE CONSTANT DETERMINISTIC REGRESSOR (FOR TESTING)

struct DeterministicConstantRegressor{F} <: MLJBase.Deterministic{F}
    target_type::Type{F}
end
DeterministicConstantRegressor(;target_type=Float64) =  DeterministicConstantRegressor(target_type)

function MLJBase.fit(model::DeterministicConstantRegressor{F}, verbosity::Int, X, y::Vector{F2}) where {F,F2}
    F == F2 || error("Model specifies target_type=$F but target type is $F2.")
    fitresult = mean(y)
    verbosity < 1 || @info "mean = $fitresult."
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

MLJBase.predict(model::DeterministicConstantRegressor, fitresult, Xnew) = fill(fitresult, nrows(Xnew))

# metadata:
MLJBase.load_path(::Type{<:DeterministicConstantRegressor}) = load_path(ConstantRegressor)
MLJBase.package_name(::Type{<:DeterministicConstantRegressor}) = MLJBase.package_name_name(ConstantRegressor)
MLJBase.package_uuid(::Type{<:DeterministicConstantRegressor}) = MLJBase.package_name_url(ConstantRegressor)
MLJBase.package_url(::Type{<:DeterministicConstantRegressor}) = MLJBase.package_name_url(ConstantRegressor)
MLJBase.is_pure_julia(::Type{<:DeterministicConstantRegressor}) = :yes
MLJBase.input_kinds(::Type{<:DeterministicConstantRegressor}) = [:continuous, :multiclass, :ordered_factor_finite, :ordered_factor_infinite, :missing]
MLJBase.output_kind(::Type{<:DeterministicConstantRegressor}) = :continuous
MLJBase.output_quantity(::Type{<:DeterministicConstantRegressor}) = :univariate


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
                 verbosity::Int,
                 X,
                 y::CategoricalVector{L2,R}) where {L,R,L2}

    L == L2 || error("Model specifies target_type=$L but target type is $L2.")

    fitresult = Distributions.fit(MLJBase.UnivariateNominal, y)

    verbosity < 1 || @info "probabilities: \n$(fitresult.prob_given_label)"
    cache = nothing
    report = nothing

    return fitresult, cache, report

end

function MLJBase.predict(model::ConstantClassifier{L}, fitresult, Xnew) where L
    return fill(fitresult, nrows(Xnew))
end

function MLJBase.predict_mode(model::ConstantClassifier{L}, fitresult, Xnew) where L
    m = mode(fitresult)
    labels = fitresult.prob_given_label |> keys |> collect
    N = nrows(Xnew)    
    
    # to get a categorical array with all the original levels we append the 
    # distribution labels to the prediction vector and truncate afterwards:
    yhat = vcat(fill(m, N), labels) |> categorical
    return yhat[1:N]
end

# metadata:
MLJBase.load_path(::Type{<:ConstantClassifier}) = "MLJ.ConstantClassifier"
MLJBase.package_name(::Type{<:ConstantClassifier}) = MLJBase.package_name(ConstantRegressor)
MLJBase.package_uuid(::Type{<:ConstantClassifier}) = MLJBase.package_uuid(ConstantRegressor)
MLJBase.package_url(::Type{<:ConstantClassifier}) = MLJBase.package_url(ConstantRegressor)
MLJBase.is_pure_julia(::Type{<:ConstantClassifier}) = :yes
MLJBase.input_kinds(::Type{<:ConstantClassifier}) = [:continuous, :multiclass, :ordered_factor_finite, :ordered_factor_infinite, :missing]
MLJBase.output_kind(::Type{<:ConstantClassifier}) = :multiclass
MLJBase.output_quantity(::Type{<:ConstantClassifier}) = :univariate


## DETERMINISTIC CONSTANT CLASSIFIER (FOR TESTING)

struct DeterministicConstantClassifier{L} <: MLJBase.Deterministic{Tuple{L,Vector{L}}}
    target_type::Type{L}
end
DeterministicConstantClassifier(;target_type=Bool) = DeterministicConstantClassifier{target_type}(target_type)

function MLJBase.fit(model::DeterministicConstantClassifier{L},
                 verbosity::Int,
                 X,
                 y::CategoricalVector{L2,R,L2_pure}) where {L,R,L2,L2_pure}

    L == L2 || error("Model specifies target_type=$L but target type is $L2.")

    # dump missing target values and make into a regular array:
    y_pure = Array{L2_pure}(skipmissing(y) |> collect)

    fitresult = (mode(y_pure), levels(y))

    verbosity < 1 || @info "mode = $fitresult"
    cache = nothing
    report = nothing

    return fitresult, cache, report

end

function MLJBase.predict(model::DeterministicConstantClassifier{L}, fitresult, Xnew) where L
    _mode, _levels = fitresult
    _nrows = nrows(Xnew)
    raw_predictions = fill(_mode, _nrows)
    return categorical(vcat(raw_predictions, _levels))[1:_nrows]
end

# metadata:
MLJBase.load_path(::Type{<:DeterministicConstantClassifier}) = "MLJ.DeterministicConstantClassifier"
MLJBase.package_name(::Type{<:DeterministicConstantClassifier}) = MLJBase.package_name_name(ConstantRegressor)
MLJBase.package_uuid(::Type{<:DeterministicConstantClassifier}) = MLJBase.package_name_uuid(ConstantRegressor)
MLJBase.package_url(::Type{<:DeterministicConstantClassifier}) = MLJBase.package_name_url(ConstantRegressor)
MLJBase.is_pure_julia(::Type{<:DeterministicConstantClassifier}) = :yes
MLJBase.input_kinds(::Type{<:DeterministicConstantClassifier}) = [:continuous, :multiclass, :ordered_factor_finite, :ordered_factor_infinite, :missing]
MLJBase.output_kind(::Type{<:DeterministicConstantClassifier}) = :multiclass
MLJBase.output_quantity(::Type{<:DeterministicConstantClassifier}) = :univariate


end # module


## EXPOSE THE INTERFACE

using .Constant
