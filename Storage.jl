store_results!(no_storage::Void, measure, laraner) = nothing

function store_results!(storage::MLRStorage, measure::Any, learner::Learner)
    push!(storage.models, learner.name)
    push!(storage.measures, measure)
    push!(storage.averageCV, mean(measure))
    push!(storage.parameters, learner.parameters)
end


function get_best(storage::MLRStorage)
    best_index = indmax(storage.averageCV)

    Dict(
        "model"=>storage.models[best_index],
        "CV score"=>storage.averageCV[best_index],
        "parameters"=>storage.paramters[best_index]
    )
end
