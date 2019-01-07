# TODO change these names to match MLR or MLMetrics?

## REGRESSOR METRICS

# Note: Each regressor metric is overloaded with a probabilistic
# version which converts `yhat` (an array of distributions) to
# corresponding mean values before applying the usual metric.

function rms(y, yhat::AbstractVector{<:Real})
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        dev = y[i] - yhat[i]
        ret += dev*dev
    end
    return sqrt(ret/length(y))
end
rms(y, yhat) = rms(y, Distributions.mean.(yhat)) 

function rmsl(y, yhat::AbstractVector{<:Real})
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i]) - log(yhat[i])
        ret += dev*dev
    end
    return sqrt(ret/length(y))
end
rmsl(y, yhat) = rmsl(y, Distributions.mean.(yhat)) 

function rmslp1(y, yhat::AbstractVector{<:Real})
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i] + 1) - log(yhat[i] + 1)
        ret += dev*dev
    end
    return sqrt(ret/length(y))
end
rmslp1(y, yhat) = rmslp1(y, Distributions.mean.(yhat)) 


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
rmsp(y, yhat) = rmsp(y, Distributions.mean.(yhat)) 

# function auc(truelabel::L) where L
#     _auc(y::AbstractVector{L}, yhat::AbstractVector{T}) where T<:Real = 
#         ROC.AUC(ROC.roc(yhat, y, truelabel))
#     return _auc
# end
