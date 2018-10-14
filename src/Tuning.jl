function get_param_grid(p::ParametersSet)
    """
    Returns a matrix or grid with the cartesian product of the combination 
    of the parameters passed in ParametersSet structure

    Each column of the matrix maps to the values of one parameter, 
    it repeats the values as many times is necessary until complete the number of combinations.
    """
    total_parameters = 1    # total_parameters is used to create the grid base in (number_of_combinations x number_of_parameters)
    for n in 1:length(p.parameters)
        values = get_param_values(p.parameters[n])
        total_parameters *= length(values)
    end

    # matrix_parameters has the size of the total number of combinations (rows) and the total number of parameters (columns)
    matrix_parameters = Array{Any}(total_parameters, length(p.parameters))

    for i in 1:length(p.parameters)
        value_list = []
        values = get_param_values(p.parameters[i])
        for j in 1:length(values)
            push!(value_list, values[j])
        end
        
        # Filling the matrix with the combination of parameters
        # `outer` is the number of times the array needs to be repeated
        outer = Int(total_parameters / length(values))
        matrix_parameters[:,i] = repeat(value_list, outer=outer)
    end
    matrix_parameters
end

function get_grid_size(matrix_parameters)
    return size(matrix_parameters)
end

function get_kth_parameter(matrix_parameters, k)
    return matrix_parameters[k,:]
end

function tune(model::T, parameterset::ParametersSet, X::AbstractArray, y::AbstractArray; measure=MLMetrics.accuracy::Function) where {T <: BaseModel}
    """
    This tune() function has 3 steps
    1) create a grid of parameters based on the types of the parameterset
    2) iterate over the grid of parameters and store the models and performance based on the measure function
    3) return the best model based on the measure function result 
    """
    #load_interface_for(model)

    num_parameters = length(parameterset.parameters)

    # 1) creates grid
    matrix_parameters = get_param_grid(parameterset)
    print(get_grid_size(matrix_parameters))

    # 2) iterate over the parameters grid
    param_names = []
    rows, cols = size(matrix_parameters)

    new_data = rand(1, 10)
    models = []
    measures = []
    for i in 1:rows

        # this part shouldn't be here it gets the parameters from the ParametersSet structure
        # to pass them to the interface.
        penalty = matrix_parameters[i,2]                    # this should be removed
        lambda = fill(matrix_parameters[i,1], size(X,2))    # this should be removed

        # Ideally a tune() function accepting SparseRegressionModel types should get the parameters from the grid and run all the models and return them.
        my_glm_model = SparseRegressionModel(parameterset)
        my_sparse_regression = fit(my_glm_model, X, y, penalty=penalty, λ=lambda)
        push!(models, my_sparse_regression)

        # TODO: implement a function reciving the input data and returing a train and test dataset for evaluation.
        y_pred = predict(my_glm_model, my_sparse_regression, X); #currently predicting same input data
        measure_result = measure(y, y_pred)
        push!(measures, measure_result)
    end

    # 3. Selecting and returning the best model based on the measure function
    print("\nResults from models: ", measures, "\n")
    
    if measure == mean_squared_error
        print("The best model was number $(indmin(measures)) \n. $(measure) $(minimum(measures)) \n")
        return models[indmin(measures)] 
    else
        print("The best model was number $(indmax(measures)) \n. $(measure) $(maximum(measures)) \n")
        return models[indmax(measures)]
    end
end

#beginning of OLD CODE

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
                println("Array out of bound while updating parameters")
            end
            array[i] = range[i][1]
        end
    end
end

"""
    Creates a dictionary of {"Parameter name"=>"Value", .. }
"""
function parameters_dictionary(ps::ParametersSet, array, discrete_prms_map)
    dict = Dict()
    for i in 1:length(array)
        if typeof(ps[i]) <: ContinuousParameter
            dict[Symbol(ps[i].name)] = ps[i].transform( convert(Float64, array[i]) )
        else
            dict[Symbol(ps[i].name)] = discrete_prms_map[ps[i].name][array[i]]
        end
    end
    dict
end


"""
    Sets up the initial parameter value-indeces and ranges of each parameter
    Also sets up the dictionary used for discrete parameters
    @return
        total number of parameters' combinations
"""
function prepare_parameters!(prms_set, prms_value, prms_range, discrete_prms_map,
                             n_parameters)

    total_parameters = 1
    for i in 1:n_parameters
        if typeof(prms_set[i]) <: ContinuousParameter
            # Setup the initial value and range of each parameter
            lower = prms_set[i].lower
            upper = prms_set[i].upper
            prms_value[i] = lower
            prms_range[i] = Tuple(lower:upper)
            params = length(lower:upper)
        else
            # For discrete parameters, we use a dict index=>discrete_value
            prms_value[i] = 1
            prms_range[i] = Tuple(1:length(prms_set[i].values))
            discrete_prms_map[prms_set[i].name] = prms_set[i].values
            params = length(prms_set[i].values)
        end
        total_parameters *= params
    end
    total_parameters
end


"""
    Tunes learner given a task and parameter sets.
    Returns a learner which contains best tuned model
"""
function tune(learner::Learner, task::MLTask, parameters_set::ParametersSet;
                sampler=Resampling()::Resampling, measure=MLMetrics.accuracy::Function,
                storage=MLJStorage()::MLJStorage)

    # TODO: divide and clean up code. Use better goddam variable names.

    n_parameters = length(parameters_set.parameters)
    n_obs        = size(data,1)

    # TODO: remake this iteration hacky thing
    # prms_value: current value-index of each parameter
    # prms_range: range of each parameter
    # For discrete parameters, the range is set to 1:(number of discrete values)
    # The discrete map variable allows to connect this range to
    # the actual discrete value it represents
    prms_value  = Array{Any}(n_parameters)
    prms_range = Array{Tuple}(n_parameters)
    discrete_prms_map = Dict()

    # Prepare parameters
    total_parameters = prepare_parameters!(parameters_set, prms_value, prms_range,
                                            discrete_prms_map, n_parameters)


    # Loop over parameters
    for i in 1:total_parameters
        # Set new parametersparameters_set[i].values
        pd = parameters_dictionary(parameters_set, prms_value, discrete_prms_map)

        # Update learner with new parameters
        lrn = ModelLearner(learner.name, pd)

        # Get training/testing validation sets
        trainⱼ, testⱼ = get_samples(sampler, n_obs)

        measures = []
        for j in 1:length(trainⱼ)
            modelᵧ = learnᵧ(lrn, task)
            preds, prob = predictᵧ(modelᵧ, task.data[testⱼ[j],task.features], task)

            _measure = measure( task.data[testⱼ[j], task.targets[1]], preds)
            push!(measures, _measure)
        end
        # Store and print cross validation results
        store_results!(storage, measures, lrn)
        println("Trained:")
        println(lrn)
        println("Average CV accuracy: $(mean(measures))\n")

        update_parameters!(prms_value, prms_range)
    end

    println("Retraining best model")
    best_index = indmax(storage.averageCV)
    lrn = ModelLearner(storage.models[best_index], storage.parameters[best_index])
    modelᵧ = learnᵧ(lrn, task)
    lrn = ModelLearner(lrn, modelᵧ, parameters_set)

    lrn
end



"""
    Tune for multiplex type
"""
function tune(multiplex::MLJMultiplex, task::MLTask;
    sampler=Resampling()::Resampling, measure=MLMetrics.accuracy::Function,
    storage=nothing::Union{Void,MLJStorage})

    # Tune each model separately
    for i in 1:multiplex.size
        multiplex.learners[i]  = tune(multiplex.learners[i], task, multiplex.parametersSets[i],
                                        sampler=sampler, measure=measure, storage=storage)
    end
end

"""
    Tunes multiple models with multiple different paramters
"""
function GroupTuner(;learners=nothing::Array{<:Learner}, task=nothing::MLTask, data=nothing::Matrix{Real},
                parameters_sets=nothing::Array{<:ParametersSet}, sampler=Resampling()::Resampling,
                measure=nothing::Function, storage=nothing::Union{Void,MLJStorage})

    storage = MLJStorage()
    for (i,lrn) in enumerate(learners)
        tune(learner=lrn, task=task, data=data, parameters_set=parameters_sets[i],
            sampler=sampler, measure=measure, storage=storage)
    end
    storage
end
#End of OLD CODE