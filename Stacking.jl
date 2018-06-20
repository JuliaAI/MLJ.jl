"""
    Tuning for stacking compositve learner
"""
function tune(learner::CompositeLearner{<:Stacking}, task::Task;
                sampler=Resampling()::Resampling, measure=MLMetrics.accuracy::Function,
                storage=MLRStorage()::MLRStorage)


    for (i,lrn) in enumerate(learner.learners)
        new_lrn = tune(lrn, task, lrn.parameters, measure=measure)
        learner.learners[i] = new_lrn
    end
    learner
end

function predictᵧ(stacking::CompositeLearner{Stacking},
                data_features::Matrix, task::Task)

    # TODO: add more stacking options
    predictions_matrix = zeros(size(data_features,1), length(stacking.learners))
    voted_matrix = zeros(Int64, size(data_features,1))

    # Ger predictions from individual learners
    for (i,learner) in enumerate(stacking.learners)
        p = predictᵧ(learner.modelᵧ, data_features, task)
        predictions_matrix[:,i] = p[1]
    end

    # Compose final prediction based on votes
    for row in 1:size(predictions_matrix,1)
        votes = Dict(0=>0, 1=>0)
        for label in predictions_matrix[row,:]
            votes[label]+=1
        end
        if stacking.composite.voting_type == MAJORITY
            voted_matrix[row] = indmax(values(votes))-1
        else
            warn("Only majority vote is currently accepted for the stacking learner.")
        end
    end
    voted_matrix
end
