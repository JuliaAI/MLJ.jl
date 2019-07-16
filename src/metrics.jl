## DEFAULT MEASURES

default_measure(model::M) where M<:Supervised =
    default_measure(model, target_scitype_union(M))
default_measure(model, ::Any) = nothing
default_measure(model::Deterministic, ::Type{<:Continuous}) = rms
default_measure(model::Probabilistic, ::Type{<:Continuous}) = rms
default_measure(model::Deterministic, ::Type{<:Finite}) =
    misclassification_rate
default_measure(model::Probabilistic, ::Type{<:Finite}) =
    cross_entropy

# TODO: the names to match MLR or MLMetrics?

# Note: If the `ŷ` argument of a deterministic metric does not have
# the expected type, the metric assumes it is a distribution and
# attempts first to compute its mean or mode (according to whether the
# metric is a regression metric or a classification metric). In this
# way each deterministic metric is overloaded as a probabilistic one.

# TODO: Above behaviour not ideal. Should explicitly test if ŷ is a
# vector of distributions (either using isdistribution trait or
# Sampleable type). Throw warning if a deterministic measure is being
# used in a probabilistic context.

"""
$SIGNATURES

Check that two vectors have compatible dimensions
"""
function check_dimensions(ŷ::A, y::A) where A <: AbstractVector{<:Real}
    length(y) == length(ŷ) || throw(DimensionMismatch("Vectors don't have the same length"))
    return nothing
end

## REGRESSOR METRICS (FOR DETERMINISTIC PREDICTIONS)

"""
$SIGNATURES

Mean absolute error (also known as MAE).

``\\text{MAV} =  n^{-1}∑ᵢ|yᵢ-ŷᵢ|``
"""
function mav(ŷ::A, y::A) where A <: AbstractVector{<:Real}
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = y[i] - ŷ[i]
        ret += abs(dev)
    end
    return ret / length(y)
end
mav(ŷ, y) = mav(mean.(ŷ), y)

# synonym
"""
mae(ŷ, y)

See also [`mav`](@ref).
"""
mae = mav


"""
$SIGNATURES

Root mean squared error:

``\\text{RMS} = n^{-1}∑ᵢ|yᵢ-ŷᵢ|^2``
"""
function rms(ŷ::A, y::A) where A <: AbstractVector{<:Real}
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = y[i] - ŷ[i]
        ret += dev * dev
    end
    return sqrt(ret / length(y))
end
rms(ŷ, y) = rms(mean.(ŷ), y)


"""
$SIGNATURES

Root mean squared logarithmic error:

``\\text{RMSL} = n^{-1}∑ᵢ\\log\\left({yᵢ \\over ŷᵢ}\\right)``

See also [`rmslp1`](@ref).
"""
function rmsl(ŷ::A, y::A) where A <: AbstractVector{<:Real}
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i]) - log(ŷ[i])
        ret += dev * dev
    end
    return sqrt(ret / length(y))
end
rmsl(ŷ, y) = rmsl(mean.(ŷ), y)


"""
$SIGNATURES

Root mean squared logarithmic error with an offset of 1:

``\\text{RMSLP1} = n^{-1}∑ᵢ\\log\\left({yᵢ + 1 \\over ŷᵢ + 1}\\right)``

See also [`rmsl`](@ref).
"""
function rmslp1(ŷ::A, y::A) where A <: AbstractVector{<:Real}
    check_dimensions(ŷ, y)
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i] + 1) - log(ŷ[i] + 1)
        ret += dev * dev
    end
    return sqrt(ret / length(y))
end
rmslp1(ŷ, y) = rmslp1(y, mean.(ŷ))


"""
$SIGNATURES

Root mean squared percentage loss:

``\\text{RMSP} = m^{-1}∑ᵢ \left({yᵢ-ŷᵢ \over yᵢ}\right)^2``

where the sum is over indices such that `yᵢ≂̸0` and `m` is the number of such indices.
"""
function rmsp(ŷ::A, y::A) where A <: AbstractVector{<:Real}
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
rmsp(ŷ, y) = rmsp(mean.(ŷ), y)


## CLASSIFICATION METRICS (FOR DETERMINISTIC PREDICTIONS)

misclassification_rate(ŷ::AbstractVector{<:CategoricalElement},
                       y::AbstractVector) = mean(y .!= ŷ)
misclassification_rate(ŷ, y::AbstractVector) =
    misclassification_rate(mode.(ŷ), y)


# TODO: multivariate case


## CLASSIFICATION METRICS (FOR PROBABILISTIC PREDICTIONS)

# for single pattern:
cross_entropy(d::UnivariateFinite, y) = -log(d.prob_given_level[y])

function cross_entropy(ŷ::Vector{<:UnivariateFinite}, y::AbstractVector)
    check_dimensions(ŷ, y)
    return broadcast(cross_entropy, ŷ, y) |> mean
end

# function auc(truelabel::L) where L
#     _auc(y::AbstractVector{L}, ŷ::AbstractVector{T}) where T<:Real =
#         ROC.AUC(ROC.roc(ŷ, y, truelabel))
#     return _auc
# end
