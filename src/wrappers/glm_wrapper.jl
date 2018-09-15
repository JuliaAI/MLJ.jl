using SparseRegression

############# MODEL SPECIFIC CONSTRUCTORS #################

"""
    Functions specifying how model should be constructed given parameters
    These function are completely model dependent and are the only ones
    that need to be written to add a new model to the list.
"""
function get_parameters(model::MLJModel{<:SModel})
    parameters = Dict(
        :λ => Dict(
                    "type"=>Union{Float64, Array{Float64}},
                    "desc"=>"Regularization constant. Small => strong regularization"
                    ),
        :penalty => Dict(
                    "type"=>LearnBase.Penalty,
                    "desc"=>"Penalty to use. Any penalty from LearnBase.Penalty can be used"
                    ),
        :loss => Dict(
                    "type"=>LearnBase.Loss,
                    "desc"=>"Loss to use. Any loss from LearnBase.Loss can be used"
                    )
    )
    parameters
end


function makeRidge(learner::Learner, task::MLTask)
    if isempty(learner.parameters)
        model = SModel(task.data[:, task.features], task.data[:, task.targets])
    else
        parameters = []
        push!(parameters, get_λ(learner.parameters, task))
        model = SModel(task.data[:, task.features], task.data[:, task.targets],
                        L2DistLoss(), L2Penalty(), parameters...)
    end
    MLJModel(model, copy(learner.parameters))
end

function makeLasso(learner::Learner, task::Task)
    if isempty(learner.parameters)
        model = SModel(task.data[:, task.features], task.data[:, task.targets])
    else
        parameters = []
        push!(parameters, get_λ(learner.parameters, task))
        model = SModel(task.data[:, task.features], task.data[:, task.targets],
                        L2DistLoss(), L1Penalty(), parameters...)
    end
    MLJModel(model, copy(learner.parameters))
end

function makeElasticnet(learner::Learner, task::RegressionTask)
    if isempty(learner.parameters)
        model = SModel(task.data[:, task.features], task.data[:, task.targets])
    else
        parameters = []
        push!(parameters, get_λ(learner.parameters, task))
        model = SModel(task.data[:, task.features], task.data[:, task.targets],
                        L2DistLoss(), ElasticNetPenalty(), parameters...)
    end
    MLJModel(model, copy(learner.parameters))
end

function makeGlm(learner::Learner, task::MLTask)
    if isempty(learner.parameters)
        model = SModel(task.data[:, task.features], task.data[:, task.targets])
    else
        parameters = []
        if get(learner.parameters, :λ, false) !== false
            # Add λ
            push!(parameters, get_λ(learner.parameters, task))
        end
        if get(learner.parameters, :penalty, false) !== false
            # Add penalty
            push!(parameters, learner.parameters[:penalty])
        end
        if get(learner.parameters, :loss, false) !== false
            # Add penalty
            push!(parameters, learner.parameters[:loss])
        end
        model = SModel(task.data[:, task.features], task.data[:, task.targets[1]], parameters...)
    end
    MLJModel(model, copy(learner.parameters))
end

# Utiliy function #
function get_λ(parameters, task::RegressionTask)
    if get(parameters, :λ, false) == false
        lambda = fill(0.0, task.features)
    elseif typeof(parameters[:λ]) <: Real
        lambda = fill(parameters[:λ], length(task.features) )
    elseif typeof(parameters[:λ]) <: Vector{AbstractFloat}
        lambda = copy(parameters[:λ])
    end
    lambda
end

################## MODEL SPECIFIC ALGORITHMS ####################

"""
    How to predict using a specific model
"""
function predictᵧ(modelᵧ::MLJModel{<:SModel},
                    data_features::Matrix, task::RegressionTask)
    p = predict(modelᵧ.model, data_features)
    p, nothing
end

"""
    How to learn using a specific model
"""
function learnᵧ!(modelᵧ::MLJModel{<:SModel}, learner::Learner, task::RegressionTask)
    learn!(modelᵧ.model)
end
