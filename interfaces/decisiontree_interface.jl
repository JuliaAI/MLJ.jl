using DecisionTree
# https://github.com/bensadeghi/DecisionTree.jl
"""
default_parameters = Dict("pruning_purity" => 1.0
    , "max_depth" => -1
    , "min_samples_leaf" => 1
    , "min_samples_split" => 2
    , "min_purity_increase" => 0.0
    , "n_subfeatures" => 0
)
"""

function fit(model::DecisionTreeClassifier, X::AbstractArray, y::AbstractArray)
    print("Fitting in the interface!")
    model_fit = build_tree(y, X)
    ModelFit(model, model_fit)
end

function predict(model::DecisionTreeClassifier, modelFit::BaseModelFit, Xnew)
    print("Predicting using: $(typeof(model))")
    predicted_value = apply_tree(modelFit.fit_result, Xnew);
end

function fit(model::DecisionTreeRegressor, X::AbstractArray, y::AbstractArray)
    print("Fitting regression in the interface!")
    model_fit = build_tree(y, X)
    ModelFit(model, model_fit)
end

function predict(model::DecisionTreeRegressor, modelFit::BaseModelFit, Xnew)
    print("Predicting regression using: $(typeof(model))")
    predicted_value = apply_tree(modelFit.fit_result, Xnew);
end
