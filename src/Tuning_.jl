using MLMetrics

function get_param_grid(p::ParametersSet)
    """
    Returns a matrix or grid with the cartesian product of the combination of the parameters passed in ParametersSet structure

    Each column of the matrix maps to the values of one parameter, it repeats the values as many times is necessary until complete the number of combinations.
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