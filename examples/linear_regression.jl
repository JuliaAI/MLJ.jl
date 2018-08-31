include("ML.jl")

load_interface_for("SparseRegressionModel")

parameters = ParametersSet([
    ContinuousParameter(
        name = :λ,
        lower = -4.,
        upper = -2.,
        transform = x->10^x
    ),
    DiscreteParameter(
        name = :penalty,
        values = [L2Penalty(), L1Penalty()]
    )
    ])

# creating input data
x = randn(1000, 10)
y = x * range(-1, 1, 10) + randn(1_000)

my_glm_model = SparseRegressionModel(parameters)
my_sparse_regression = fit(my_glm_model, x, y)

new_data = rand(1, 10)
predict(my_glm_model, my_sparse_regression, new_data)
y_pred = predict(my_glm_model, my_sparse_regression, x)

mean_squared_error(y, y_pred)

# Tuning
"""
The tune() function returns the best model from a selection of trained models from a grid of parameters.

It takes a model type, a parameter set with its ranges and values to create a grid of parameter combinations, 
the input and target variable, and a function to measure performance of the models.

Create a TunedModel object, from a basic model, ParamterSet and Metric:
tmp = TunedModel(model, paramset, metric)

TODO:
Gergo's suggestion: to create a fit function to wrap up the tune() function. So the user just calls fit().
    fit(tmp, x, y) = tune(tmp.model, tmp.params, x, y, measure = tmp.metric)
"""
best_model = tune(my_glm_model, parameters, x, y, measure=mean_squared_error)