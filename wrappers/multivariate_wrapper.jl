using MultivariateStats


### TODO: Types need to be re-thought

abstract type  MultivariateModel end

mutable struct MultivariateLlsq <: MultivariateModel
    sol::Matrix
    MultivariateLlsq() = new(zeros(0,0))
end

mutable struct MultivariateRidge <: MultivariateModel
    λ::Number
    sol::Matrix
    label::Union{Dict, Void}
    MultivariateRidge(λ) = new(λ, zeros(0,0), nothing)
end

function getParamsMultivariate()
    possible_parameters = Dict(
        :regType=>[:ridge, :llsq]
    )
    possible_parameters
end

function makeMultivariate(learner::Learner, task::Task)
    prms = learner.parameters
    possible_parameters = getParamsMultivariate()

    if prms[:regType] in possible_parameters[:regType]
        if prms[:regType] == :ridge
            λ = get(prms, :λ, false)
            λ =  (λ==false ? 0.1 : λ)
            MLRModel(MultivariateRidge(λ), copy(prms))
        else
            MLRModel(MultivariateLlsq(), copy(prms))
        end
    else
        throw("regType must be either :ridge or :llsq")
    end
end

"""
    Train a ridge classifier
"""
function learnᵧ!(modelᵧ::MLRModel{<:MultivariateRidge}, learner::Learner, task::ClassificationTask)
    # Convert user labels to +/- 1 "binary" labelled data
    margin_labels = convertlabel( LabelEnc.MarginBased, task.data[:,task.targets])

    # Keep track of transformation for prediction
    original_labels = unique(task.data[:,task.targets])
    keys = Dict(original_labels[1]=>1, original_labels[2]=>2)
    modelᵧ.model.label_keys = keys

    modelᵧ.model.sol = ridge(task.data[:,task.features], margin_labels, modelᵧ.model.λ)
end

"""
    Predicts using a ridge classifier.
    Probabilities are based on distance from predicted label
"""
function predictᵧ(modelᵧ::MLRModel{<:MultivariateModel},
                    data_features::Matrix{<:Real}, task::RegressionTask)

    sol = modelᵧ.model.sol
    A, b = sol[1:end-1,:], sol[end,:][:,:]
    preds = data_features * A .+ b'

    n_points = size(task.data, 1)

    labels, probabilities = zeros(n_points), zeros(n_points)
    for (i,p) in enumerate(preds)
        # Get label by whether predicted positive or negative
        labels[i] = p>0 ? 1 : -1
        # Get "probability" (confidence) by distance to label
        probabilities = 1.0-abs(p - label[i])
    end
    predicted_labels = [ p>0 ? 1 : -1 for p in preds]
    labels, probabilities
end


function learnᵧ!(modelᵧ::MLRModel{<:MultivariateRidge}, learner::Learner, task::RegressionTask)
    modelᵧ.model.sol = ridge(task.data[:,task.features], task.data[:,task.targets], modelᵧ.model.λ)
end


function learnᵧ!(modelᵧ::MLRModel{<:MultivariateLlsq}, learner::Learner, task::RegressionTask)
    modelᵧ.model.sol = llsq(task.data[:,task.features], task.data[:,task.targets])
end

function predictᵧ(modelᵧ::MLRModel{<:MultivariateModel},
                    data_features::Matrix{<:Real}, task::RegressionTask)

    sol = modelᵧ.model.sol
    A, b = sol[1:end-1,:], sol[end,:][:,:]
    preds = data_features * A .+ b'
    preds, nothing
end
