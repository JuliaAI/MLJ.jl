## TRAITS FOR MEASURES

# to be extended from MLJBase:
import MLJBase.target_scitype    # fallback value = Unknown
import MLJBase.supports_weights  # fallback value = false

# new:
prediction_type(measure) = :unknown # other options are :probabilistic, :deterministic, or :interval
orientation(measure) = :loss  # other options are :score, :other
reports_each_observation(measure) = false
is_feature_dependent(measure) = false


## EVALUATION 

# yhat - predictions (point or probabilisitic)
# X - features
# y - target observations
# w - per-observation weights

# Note that the following methods are unspecialized to any particular
# measure. They can be overloaded; see eg. src/loss_functions_extension.jl

value(measure, yhat, X, y, w) = value(measure, yhat, X, y, w,
                                      Val(is_feature_dependent(measure)),
                                      Val(supports_weights(measure)))

#  is feature independent, weights not supported:
value(measure, yhat, X, y, w, ::Val{false}, ::Val{false}) = measure(yhat, y)

#  is feature dependent:, weights not supported:
value(measure, yhat, X, y, w, ::Val{true}, ::Val{false}) = measure(yhat, X, y)


#  is feature independent, weights supported:
value(measure, yhat, X, y, w, ::Val{false}, ::Val{true}) = measure(yhat, y, w)
value(measure, yhat, X, y, ::Nothing, ::Val{false}, ::Val{true}) = measure(yhat, y)

#  is feature dependent, weights supported:
value(measure, yhat, X, y, w, ::Val{true}, ::Val{true}) = measure(yhat, X, y, w)
value(measure, yhat, X, y, ::Nothing, ::Val{true}, ::Val{true}) = measure(yhat, X, y)


## HELPERS

"""
$SIGNATURES

Check that two vectors have compatible dimensions
"""
function check_dimensions(ŷ::AbstractVector, y::AbstractVector)
    length(y) == length(ŷ) ||
        throw(DimensionMismatch("Differing numbers of observations and "*
                                "predictions. "))
    return nothing
end

function check_pools(ŷ, y)
              y[1].pool.index == classes(ŷ[1])[1].pool.index ||
              error("Conflicting categorical pools found "*
                    "in observations and predictions. ")
    return nothing
end

              
## FOR BUILT-IN MEASURES

abstract type Measure end

Base.show(stream::IO, ::MIME"text/plain", m::Measure) = print(stream, "$(measurename(m)) (callable Measure)")
Base.show(stream::IO, m::Measure) = print(stream, measurename(m))

"""
    traits(measure::Measure)

Return a named tuple summarizing the traits defined for `measure`. 

"""
traits(measure::Measure) = (target_scitype=target_scitype(measure),
                            prediction_type=prediction_type(measure),
                            orientation=orientation(measure),
                            reports_each_observation=reports_each_observation(measure),
                            is_feature_dependent=is_feature_dependent(measure),
                            supports_weights=supports_weights(measure))


## REGRESSOR METRICS (FOR DETERMINISTIC PREDICTIONS)

mutable struct MAV<: Measure end
"""
    mav(ŷ, y)
    mav(ŷ, y, w)

Mean absolute error (also known as MAE).

``\\text{MAV} =  n^{-1}∑ᵢ|yᵢ-ŷᵢ|`` or ``\\text{MAV} =  ∑ᵢwᵢ|yᵢ-ŷᵢ|/∑ᵢwᵢ``

For more MLJBase.information, run `MLJBase.info(mav)`.

"""
mav = MAV()
measurename(::MAV) = "mav"

target_scitype(::MAV) = Union{AbstractVector{Continuous},AbstractVector{Count}}
prediction_type(::MAV) = :deterministic
orientation(::MAV) = :loss
reports_each_observation(::MAV) = false
is_feature_dependent(::MAV) = false
supports_weights(::MAV) = true

function (::MAV)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = y[i] - ŷ[i]
        ret += abs(dev)
    end
    return ret / length(y)
end

function (::MAV)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    check_dimensions(y, w)
    ret = 0.0
    for i in eachindex(y)
        dev = w[i]*(y[i] - ŷ[i])
        ret += abs(dev)
    end
    return ret / sum(w)
end

# synonym
"""
mae(ŷ, y)

See also [`mav`](@ref).
"""
const mae = mav


struct RMS <: Measure end
"""
    rms(ŷ, y)
    rms(ŷ, y, w)

Root mean squared error:

``\\text{RMS} = \\sqrt{n^{-1}∑ᵢ|yᵢ-ŷᵢ|^2}`` or ``\\text{RMS} = \\sqrt{\\frac{∑ᵢwᵢ|yᵢ-ŷᵢ|^2}{∑ᵢwᵢ}}``

For more MLJBase.information, run `MLJBase.info(rms)`.

"""
rms = RMS()
measurename(::RMS) = "rms"

target_scitype(::RMS) = Union{AbstractVector{Continuous},AbstractVector{Count}}
prediction_type(::RMS) = :deterministic
orientation(::RMS) = :loss
reports_each_observation(::RMS) = false
is_feature_dependent(::RMS) = false
supports_weights(::RMS) = true

function (::RMS)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = y[i] - ŷ[i]
        ret += dev * dev
    end
    return sqrt(ret / length(y))
end

function (::RMS)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = y[i] - ŷ[i]
        ret += w[i]*dev*dev
    end
    return sqrt(ret / sum(w))
end

struct L2 <: Measure end
"""
    l2(ŷ, y)
    l2(ŷ, y, w)

L2 per-observation loss.

For more MLJBase.information, run `MLJBase.info(l2)`.

"""
l2 = L2()
measurename(::L2) = "l2"

target_scitype(::L2) = Union{AbstractVector{Continuous},AbstractVector{Count}}
prediction_type(::L2) = :deterministic
orientation(::L2) = :loss
reports_each_observation(::L2) = true
is_feature_dependent(::L2) = false
supports_weights(::L2) = true

function (::L2)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    (check_dimensions(ŷ, y); (y - ŷ).^2)
end


function (::L2)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    check_dimensions(w, y)
    return (y - ŷ).^2 .* w ./ (sum(w)/length(y)) 
end

struct L1 <: Measure end
"""
    l1(ŷ, y)
    l1(ŷ, y, w)

L1 per-observation loss.

For more MLJBase.information, run `MLJBase.info(l1)`.

"""
l1 = L1()
measurename(::L1) = "l1"

target_scitype(::L1) = Union{AbstractVector{Continuous},AbstractVector{Count}}
prediction_type(::L1) = :deterministic
orientation(::L1) = :loss
reports_each_observation(::L1) = true
is_feature_dependent(::L1) = false
supports_weights(::L1) = true

function (::L1)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    (check_dimensions(ŷ, y); abs.(y - ŷ))
end

function (::L1)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    check_dimensions(w, y)
    return abs.(y - ŷ) .* w ./ (sum(w)/length(y))
end

struct RMSL <: Measure end
"""
    rmsl(ŷ, y)

Root mean squared logarithmic error:

``\\text{RMSL} = n^{-1}∑ᵢ\\log\\left({yᵢ \\over ŷᵢ}\\right)``

For more MLJBase.information, run `MLJBase.info(rmsl)`.

See also [`rmslp1`](@ref).

"""
rmsl = RMSL()
measurename(::RMSL) = "rmsl"

target_scitype(::RMSL) = Union{AbstractVector{Continuous},AbstractVector{Count}}
prediction_type(::RMSL) = :deterministic
orientation(::RMSL) = :loss
reports_each_observation(::RMSL) = false
is_feature_dependent(::RMSL) = false
supports_weights(::RMSL) = false

function (::RMSL)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i]) - log(ŷ[i])
        ret += dev * dev
    end
    return sqrt(ret / length(y))
end


struct RMSLP1 <: Measure end
"""
    rmslp1(ŷ, y)

Root mean squared logarithmic error with an offset of 1:

``\\text{RMSLP1} = n^{-1}∑ᵢ\\log\\left({yᵢ + 1 \\over ŷᵢ + 1}\\right)``

For more MLJBase.information, run `MLJBase.info(rmslp1)`.

See also [`rmsl`](@ref).
"""
rmslp1 = RMSLP1()
measurename(::RMSLP1) = "rmslp1"

target_scitype(::RMSLP1) = Union{AbstractVector{Continuous},AbstractVector{Count}}
prediction_type(::RMSLP1) = :deterministic
orientation(::RMSLP1) = :loss
reports_each_observation(::RMSLP1) = false
is_feature_dependent(::RMSLP1) = false
supports_weights(::RMSLP1) = false

function (::RMSLP1)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i] + 1) - log(ŷ[i] + 1)
        ret += dev * dev
    end
    return sqrt(ret / length(y))
end

struct RMSP <: Measure end
"""
    rmsp(ŷ, y)

Root mean squared percentage loss:

``\\text{RMSP} = m^{-1}∑ᵢ \\left({yᵢ-ŷᵢ \\over yᵢ}\\right)^2``

where the sum is over indices such that `yᵢ≂̸0` and `m` is the number
of such indices.

For more MLJBase.information, run `MLJBase.info(rmsp)`.

"""
rmsp = RMSP()
measurename(::RMSP) = "rmsp"

target_scitype(::RMSP) = Union{AbstractVector{Continuous},AbstractVector{Count}}
prediction_type(::RMSP) = :deterministic
orientation(::RMSP) = :loss
reports_each_observation(::RMSP) = false
is_feature_dependent(::RMSP) = false
supports_weights(::RMSP) = false

function (::RMSP)(ŷ::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    check_dimensions(ŷ, y)
    ret = 0.0
    count = 0
    for i in eachindex(y)
        if y[i] != 0.0
            dev = (y[i] - ŷ[i])/y[i]
            ret += dev * dev
            count += 1
        end
    end
    return sqrt(ret/count)
end


## CLASSIFICATION METRICS (FOR DETERMINISTIC PREDICTIONS)

struct MisclassificationRate <: Measure end

"""
    misclassification_rate(ŷ, y)
    misclassification_rate(ŷ, y, w)

Returns the rate of misclassification of the (point) predictions `ŷ`,
given true observations `y`, optionally weighted by the weights
`w`. All three arguments must be abstract vectors of the same length.

For more MLJBase.information, run `MLJBase.info(misclassification_rate)`.

"""
misclassification_rate = MisclassificationRate()
measurename(::MisclassificationRate) = "misclassification_rate"

target_scitype(::MisclassificationRate) = AbstractVector{<:Finite}
prediction_type(::MisclassificationRate) = :deterministic
orientation(::MisclassificationRate) = :loss
reports_each_observation(::MisclassificationRate) = false
is_feature_dependent(::MisclassificationRate) = false
supports_weights(::MisclassificationRate) = true

(::MisclassificationRate)(ŷ::AbstractVector{<:CategoricalElement},
                          y::AbstractVector{<:CategoricalElement}) =
                              mean(y .!= ŷ)
(::MisclassificationRate)(ŷ::AbstractVector{<:CategoricalElement},
                          y::AbstractVector{<:CategoricalElement},
                          w::AbstractVector{<:Real}) =
                              sum((y .!= ŷ) .*w) / sum(w)


## CLASSIFICATION METRICS (FOR PROBABILISTIC PREDICTIONS)

struct CrossEntropy <: Measure end

"""
    cross_entropy(ŷ, y::AbstractVector{<:Finite})

Given an abstract vector of `UnivariateFinite` distributions `ŷ` (ie, of 
probabilistic predictions) and an abstract vector of true observations
`y`, return the negative log-probability that each observation would
occur, according to the corresponding probabilistic prediction.

For more MLJBase.information, run `MLJBase.info(cross_entropy)`.

"""
cross_entropy = CrossEntropy()
measurename(::CrossEntropy) = "cross_entropy"

target_scitype(::CrossEntropy) = AbstractVector{<:Finite}
prediction_type(::CrossEntropy) = :probabilistic
orientation(::CrossEntropy) = :loss
reports_each_observation(::CrossEntropy) = true
is_feature_dependent(::CrossEntropy) = false
supports_weights(::CrossEntropy) = false

# for single observation:
_cross_entropy(d, y) = -log(pdf(d, y))

function (::CrossEntropy)(ŷ::AbstractVector{<:UnivariateFinite},
                          y::AbstractVector{<:CategoricalElement})
    check_dimensions(ŷ, y)
    check_pools(ŷ, y)          
    return broadcast(_cross_entropy, Any[ŷ...], y)
end


## DEFAULT MEASURES

default_measure(model::M) where M<:Supervised =
    default_measure(model, target_scitype(M))
default_measure(model, ::Any) = nothing
default_measure(model::Deterministic,
                ::Type{<:Union{AbstractVector{Continuous},
                               AbstractVector{Count}}}) = rms
# default_measure(model::Probabilistic,
#                 ::Type{<:Union{AbstractVector{Continuous},
#                                AbstractVector{Count}}}) = rms
default_measure(model::Deterministic,
                ::Type{<:AbstractVector{<:Finite}}) =
                    misclassification_rate
default_measure(model::Probabilistic,
                ::Type{<:AbstractVector{<:Finite}}) =
                    cross_entropy


