## DEFAULT MEASURES

default_measure(model::M) where M<:Supervised =
    default_measure(model,
                    Val((MLJBase.output_kind(M),
                         MLJBase.output_quantity(M))))
default_measure(model, ::Any) = nothing
default_measure(model::Deterministic, ::Val{(:continuous, :univariate)}) = rms
default_measure(model::Probabilistic, ::Val{(:continuous, :univariate)}) = rms
default_measure(model::Deterministic, ::Val{(:binary, :univariate)}) = misclassification_rate
default_measure(model::Deterministic, ::Val{(:multiclass, :univariate)}) = misclassification_rate
default_measure(model::Probabilistic, ::Val{(:binary, :univariate)}) = cross_entropy
default_measure(model::Probabilistic, ::Val{(:multiclass, :univariate)}) = cross_entropy


# TODO the names to match MLR or MLMetrics?

# Note: If the `yhat` argument of a deterministic metric does not have
# the expected type, the metric assumes it is a distribution and
# attempts first to compute its mean or mode (according to whether the
# metric is a regression metric or a classification metric). In this
# way each deterministic metric is overloaded as a probabilistic one.

## REGRESSOR METRICS (FOR DETERMINISTIC PREDICTIONS)


function rms(y, yhat::AbstractVector{<:Real})
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        dev = y[i] - yhat[i]
        ret += dev*dev
    end
    return sqrt(ret/length(y))
end
rms(y, yhat) = rms(y, mean.(yhat)) 

function rmsl(y, yhat::AbstractVector{<:Real})
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i]) - log(yhat[i])
        ret += dev*dev
    end
    return sqrt(ret/length(y))
end
rmsl(y, yhat) = rmsl(y, mean.(yhat)) 

function rmslp1(y, yhat::AbstractVector{<:Real})
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i] + 1) - log(yhat[i] + 1)
        ret += dev*dev
    end
    return sqrt(ret/length(y))
end
rmslp1(y, yhat) = rmslp1(y, mean.(yhat)) 


""" Root mean squared percentage loss """
function rmsp(y, yhat::AbstractVector{<:Real})
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
rmsp(y, yhat) = rmsp(y, mean.(yhat)) 


## CLASSIFICATION METRICS (FOR DETERMINISTIC PREDICTIONS)

misclassification_rate(y::CategoricalVector{L}, yhat::CategoricalVector{L}) where L =
    mean(y .!= yhat)
misclassification_rate(y::CategoricalArray, yhat) =
    misclassification_rate(y, mode.(yhat))

# TODO: multivariate case 


## CLASSIFICATION METRICS (FOR PROBABILISTIC PREDICTIONS)

# for single pattern:
cross_entropy(y, d::UnivariateNominal) = -log(d.prob_given_level[y])

cross_entropy(y::CategoricalVector{L}, yhat::Vector{<:UnivariateNominal{L}}) where L =
    broadcast(cross_entropy, y, yhat) |> mean


# function auc(truelabel::L) where L
#     _auc(y::AbstractVector{L}, yhat::AbstractVector{T}) where T<:Real = 
#         ROC.AUC(ROC.roc(yhat, y, truelabel))
#     return _auc
# end

