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
    MultivariateRidge(λ) = new(λ, zeros(0,0))
end

function getParamsMultivariate()
    possible_parameters = Dict(
        "regType"=>["ridge", "llsq"]
    )
    possible_parameters
end

function makeMultivariate(learner::Learner, task::Task)
    prms = learner.parameters
    possible_parameters = getParamsMultivariate()

    if prms["regType"] in possible_parameters["regType"]
        if prms["regType"] == "ridge"
            λ = get(prms, "λ", false)
            if λ == false λ=0.1 end
            MLRModel(MultivariateRidge(λ), copy(prms))
        else
            MLRModel(MultivariateLlsq(), copy(prms))
        end
    else
        throw("regType must be either \"ridge\" or \"llsq\"")
    end
end


function learnᵧ!(modelᵧ::MLRModel{<:MultivariateRidge}, learner::Learner, task::Task)

        modelᵧ.model.sol = ridge(task.data[:,task.features], task.data[:,task.targets], modelᵧ.model.λ)
end


function learnᵧ!(modelᵧ::MLRModel{<:MultivariateLlsq}, learner::Learner, task::Task)

        modelᵧ.model.sol = llsq(task.data[:,task.features], task.data[:,task.targets])
end

function predictᵧ(modelᵧ::MLRModel{<:MultivariateModel},
                    data_features::Matrix{<:Real}, task::Task)

    sol = modelᵧ.model.sol
    A, b = sol[1:end-1,:], sol[end,:][:,:]
    preds = data_features * A .+ b'
    preds, nothing
end
