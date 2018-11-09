# TODO change these names to match MLR or MLMetrics?

function rms(y, yhat, rows)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in rows
        dev = y[i] - yhat[i]
        ret += dev*dev
    end
    return sqrt(ret/length(rows))
end

function rms(y, yhat)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        dev = y[i] - yhat[i]
        ret += dev*dev
    end
    return sqrt(ret/length(y))
end

function rmsl(y, yhat, rows)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in rows
        dev = log(y[i]) - log(yhat[i])
        ret += dev*dev
    end
    return sqrt(ret/length(rows))
end

function rmsl(y, yhat)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i]) - log(yhat[i])
        ret += dev*dev
    end
    return sqrt(ret/length(y))
end

function rmslp1(y, yhat, rows)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in rows
        dev = log(y[i] + 1) - log(yhat[i] + 1)
        ret += dev*dev
    end
    return sqrt(ret/length(rows))
end

function rmslp1(y, yhat)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    for i in eachindex(y)
        dev = log(y[i] + 1) - log(yhat[i] + 1)
        ret += dev*dev
    end
    return sqrt(ret/length(y))
end

""" Root mean squared percentage loss """
function rmsp(y, yhat, rows)
    length(y) == length(yhat) || throw(DimensionMismatch())
    ret = 0.0
    count = 0
    for i in rows
        if y[i] != 0.0
            dev = (y[i] - yhat[i])/y[i]
            ret += dev*dev
            count += 1
        end
    end
    return sqrt(ret/count)
end

function rmsp(y, yhat)
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

# function auc(truelabel::L) where L
#     _auc(y::AbstractVector{L}, yhat::AbstractVector{T}) where T<:Real = 
#         ROC.AUC(ROC.roc(yhat, y, truelabel))
#     return _auc
# end
