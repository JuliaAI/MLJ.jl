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

# Note: If the `yhat` argument of a deterministic metric does not have
# the expected type, the metric assumes it is a distribution and
# attempts first to compute its mean or mode (according to whether the
# metric is a regression metric or a classification metric). In this
# way each deterministic metric is overloaded as a probabilistic one.

# TODO: Above behaviour not ideal. Should explicitly test if yhat is a
# vector of distributions (either using isdistribution trait or
# Sampleable type). Throw warning if a deterministic measure is being
# used in a probabilistic context.

## REGRESSOR METRICS (FOR DETERMINISTIC PREDICTIONS)


function mav(yhat::AbstractVector{<:Real}, y)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        dev = y[i] - yhat[i]
        ret += abs(dev)
    end
    return ret/length(y)
end
mav(yhat, y) = mav(mean.(yhat), y) 

function rms(yhat::AbstractVector{<:Real}, y)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        dev = y[i] - yhat[i]
        ret += dev*dev
    end
    return sqrt(ret/length(y))
end
rms(yhat, y) = rms(mean.(yhat), y)

function rmsl(yhat::AbstractVector{<:Real}, y)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i]) - log(yhat[i])
        ret += dev*dev
    end
    return sqrt(ret/length(y))
end
rmsl(yhat, y) = rmsl(mean.(yhat), y) 

function rmslp1(yhat::AbstractVector{<:Real}, y)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i] + 1) - log(yhat[i] + 1)
        ret += dev*dev
    end
    return sqrt(ret/length(y))
end
rmslp1(yhat, y) = rmslp1(y, mean.(yhat))


""" Root mean squared percentage loss """
function rmsp(yhat::AbstractVector{<:Real}, y)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    count = 0
    for i in eachindex(y)
        if y[i] != 0.0
            dev = (y[i] - yhat[i])/y[i]
            ret += dev*dev
            count += 1
        end
    end
    return sqrt(ret/count)
end
rmsp(yhat, y) = rmsp(mean.(yhat), y) 


## CLASSIFICATION METRICS (FOR DETERMINISTIC PREDICTIONS)

misclassification_rate(yhat::AbstractVector{<:CategoricalElement},
                       y::AbstractVector) = mean(y .!= yhat)
misclassification_rate(yhat, y::AbstractVector) = 
    misclassification_rate(mode.(yhat), y)


# TODO: multivariate case 


## CLASSIFICATION METRICS (FOR PROBABILISTIC PREDICTIONS)

# for single pattern:
cross_entropy(d::UnivariateFinite, y) = -log(d.prob_given_level[y])

function cross_entropy(yhat::Vector{<:UnivariateFinite}, y::AbstractVector)
    length(y) == length(yhat) || throw(DimensionMismatch())
    return broadcast(cross_entropy, yhat, y) |> mean
end

# function auc(truelabel::L) where L
#     _auc(y::AbstractVector{L}, yhat::AbstractVector{T}) where T<:Real = 
#         ROC.AUC(ROC.roc(yhat, y, truelabel))
#     return _auc
# end

