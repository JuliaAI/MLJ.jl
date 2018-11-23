"""
    Updates the current array of parameters, looping around when out of their
    range. Only modifies array
"""
function update_parameters!(array, range)

    array[1] += 1
    for i in 1:length(array)-1
        if array[i] > range[i][end]
            try
                array[i+1] += 1
            catch e
                println("$i is the culprit")
                println("$(array[i]), $(range[i])")
                println("$range, $array")
            end

            array[i] = range[i][1]
        end
    end
end

"""
    Creates a dictionary of {"Parameter name"=>"Value", .. }
"""
function parameters_dictionary(ps::ParametersSet, array, discrete_dictionary)
    dict = Dict()
    for i in 1:length(array)
        if typeof(ps[i]) <: ContinuousParameter
            dict[ps[i].name] = ps[i].transform( convert(Float64, array[i]) )
        else
            dict[ps[i].name] = discrete_dictionary[ps[i].name][array[i]]
        end
    end
    dict
end

"""
    returns lists of train and test arrays, based on the sampling method
"""
function get_samples(sampler::Resampling, n_obs::Int64)
    trainᵢ = []
    testᵢ = []
    if sampler.method == "KFold"
        kfold = Kfold(n_obs, sampler.iterations)
        for train in kfold
            push!(trainᵢ, collect(train))
            push!(testᵢ, setdiff(1:n_obs, trainᵢ[end]))
        end
    end
    trainᵢ, testᵢ
end

"""
    Tunes the model
"""
function tune(;learner=nothing::Learner, task=nothing::MLTask, data=nothing::Matrix{Real},
                parameters_set=nothing::ParametersSet, sampler=Resampling()::Resampling,
                measure=nothing::Function)

    # TODO: divide and clean up code. Use better goddam variable names.

    n_parameters = length(parameters_set.parameters)
    n_obs        = size(data,1)

    parameters_array = Array{Any}(n_parameters)
    parameters_range = Array{Tuple}(n_parameters)

    # For discrete parameters, the range is set to 1:(number of discrete values)
    # The discrete dictionary variable allows to connect this range to
    # the actual discrete value it represents
    discrete_dictionary = Dict()

    total_parameters = 1

    # Prepare parameters
    for i in 1:n_parameters
        if typeof(parameters_set[i]) <: ContinuousParameter
            lower = parameters_set[i].lower
            upper = parameters_set[i].upper
            parameters_array[i] = lower
            parameters_range[i] = Tuple(lower:upper)
            params = length(lower:upper)
        else
            parameters_array[i] = 1
            parameters_range[i] = Tuple(1:length(parameters_set[i].values))
            discrete_dictionary[parameters_set[i].name] = parameters_set[i].values
            params = length(parameters_set[i].values)
        end
        total_parameters *= params
    end


    # Loop over parameters
    for i in 1:total_parameters
        # Set new parametersparameters_set[i].values
        pd = parameters_dictionary(parameters_set, parameters_array, discrete_dictionary)

        # Update learner with new parameters
        lrn = Learner(learner.name, pd)

        # Get training/testing validation sets
        trainⱼ, testⱼ = get_samples(sampler, n_obs)

        scores = []
        for j in 1:length(trainⱼ)
            modelᵧ = learnᵧ(lrn, task, data[trainⱼ[j], :])
            preds = predictᵧ(modelᵧ, data=data[testⱼ[j],:], task=task)

            score = measure( data[testⱼ[j], task.target], preds)
            push!(scores, score)
        end
        println("Trained:")
        println(lrn)
        println("Average CV accuracy: $(mean(scores))\n")

        update_parameters!(parameters_array, parameters_range)

    end
end
# greedy
# compare with variable selection in MLR https://github.com/mlr-org/mlr/blob/bb32eb8f6e7cbcd3a653440325a28632843de9f6/R/selectFeaturesSequential.R
# backwards is here http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE 

function variable_select_forward(;learner=nothing::Learner, task=nothing::MLTask, data=nothing::Matrix{Real}, sampler=Resampling()::Resampling,
                measure=nothing::Function)

    # TODO: divide and clean up code. Use better goddam variable names.

    p=size(data)[2];
    
    vars=Set([1:p;])
    selvar=Int64[]
    # Loop over parameters
    
    while lengths(selvars)< p
        print("$(length(selvar)+1). Variables")
        
        res=[]
        resv=[]
        for v in vars
            tmpvars= vcat(selvar, [v])
            
        # Set new parametersparameters_set[i].values
          

            # Update learner with new parameters
            lrn = Learner(learner.name)

            # Get training/testing validation sets
            trainⱼ, testⱼ = get_samples(sampler, n_obs)

            scores = []
            for j in 1:length(trainⱼ)
                modelᵧ = learnᵧ(lrn, task, data[trainⱼ[j], tmpvars])
                preds = predictᵧ(modelᵧ, data=data[testⱼ[j],tmpvars], task=task)

                score = measure( data[testⱼ[j], task.target], preds)
                push!(scores, score)
            end
            println("Trained:")
            println(lrn)
            println("Average CV accuracy: $(mean(scores))\n")
            push!(res,mean(scores))
            push!(resv,v)

        end
        i=argmax(res)
        @show selvar=resv[i]
    end
end
