# implementation of MLJ measure interface for LossFunctions.jl 

import .LossFunctions: DistanceLoss, MarginLoss, SupervisedLoss
# import LossFunctions: DistanceLoss, MarginLoss, SupervisedLoss

orientation(measure::SupervisedLoss) = :loss 
reports_each_observation(measure::SupervisedLoss) = true
is_feature_dependent(measure::SupervisedLoss) = false
supports_weights(measure::SupervisedLoss) = true

MLJBase.info(measure::SupervisedLoss) =
    (target_scitype=target_scitype(measure),
     prediction_type=prediction_type(measure),
     orientation=orientation(measure),
     reports_each_observation=reports_each_observation(measure),
     is_feature_dependent=is_feature_dependent(measure),
     supports_weights=supports_weights(measure))


## DISTANCE BASED LOSS FUNCTION

prediction_type(measure::DistanceLoss) = :deterministic
target_scitype(measure::DistanceLoss) = AbstractArray{<:Continuous}

value(measure::DistanceLoss, yhat, X, y, ::Nothing, ::Val{false}, ::Val{true}) =
    measure(yhat, y)
value(measure::DistanceLoss, yhat, X, y, w, ::Val{false}, ::Val{true}) =
    w .* measure(yhat, y) ./ (sum(w)/length(y))


## MARGIN BASED LOSS FUNCTIONS

prediction_type(measure::MarginLoss) = :probabilistic
target_scitype(measure::MarginLoss) = AbstractArray{<:Binary}

# convert a Binary vector into vector of +1 or -1 values (for testing
# only):
pm1(y) = Int8(2).*(Int8.(MLJBase.int(y))) .- Int8(3)

# rescale [0, 1] -> [-1, 1]
_scale(p) = 2p - 1


function value(measure::MarginLoss, yhat,
               X, y, ::Nothing, ::Val{false}, ::Val{true})
    check_pools(yhat, y)
    probs_of_observed = broadcast(pdf, yhat, y)
    return broadcast(measure, _scale.(probs_of_observed), 1)
end

value(measure::MarginLoss, yhat, X, y, w, ::Val{false}, ::Val{true}) =
    w .* value(measure, yhat, X, y, nothing) ./ (sum(w)/length(y))









