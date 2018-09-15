using LIBSVM


function getParamsLibsvm()
    possible_parameters = Dict(
        :svmtype=>LIBSVM.AbstractSVC,
        :kernel=>Kernel.KERNEL,
        :degree=>Integer,
        :cost=>Float64,
        :coef0=>Float64,
        :nu=>Float64,
        :epsilon=>Float64,
        :tolerance=>Float64,
        :probability=>Bool,
        :verbose=>Bool
        # Many more to add but this will do for now
    )
    possible_parameters
end

function makeLibsvm(learner::Learner, task::ClassificationTask)
    parameters = Dict()
    possible_parameters = getParamsLibsvm()

    if get(learner.parameters, :svmtype, false) == false
        throw("Parameter :svmtype must be set")
    end

    for (p_symbol, p_value) in learner.parameters
        if get(possible_parameters, p_symbol, false) != false
            if !(typeof(p_value) <: possible_parameters[p_symbol])
                throw("Parameter $p_symbol is not of the correct type: $p_value
                        ($(typeof(p_value)) instead of $(possible_parameters[p_symbol]))")
            end
            parameters[p_symbol] = p_value
        else
            println("Parameter $p_symbol was not found and is therefore ignored")
        end
    end
    parameters[:svmtype] = typeof(parameters[:svmtype])
    MLJModel(learner.parameters[:svmtype], parameters, inplace=false)
end

function predictᵧ(modelᵧ::MLJModel{<:LIBSVM.SVM{Float64}},
                data_features::Matrix, task::ClassificationTask)
    (labels, decision_values) = svmpredict(modelᵧ.model, data_features')
    labels, decision_values
end

function learnᵧ(modelᵧ::MLJModel{<:LIBSVM.AbstractSVC}, learner::Learner, task::ClassificationTask)
    train = task.data[:, task.features]'
    targets = task.data[:,task.targets[1]]

    model = svmtrain(train,targets; modelᵧ.parameters...)
    modelᵧ = MLJModel(model, modelᵧ.parameters)
end
