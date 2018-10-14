using SparseRegression

"""
# Arguments
- `loss::Loss = .5 * L2DistLoss()`
- `penalty::Penalty = L2Penalty()`
- `λ::Vector{Float64} = fill(size(x, 2), .1)`
- `w::Union{Nothing, AbstractWeights} = nothing`
"""

# fit() function should only have model instance, X, and y. Default values should be used from the model definition.
function fit(model::SparseRegressionModel, X::AbstractArray, y::AbstractArray; penalty::Penalty=L2Penalty(), λ::Vector{Float64} = fill(.1, size(X, 2)))
    #print("Fitting from SparseRegression Interface with parameters...")
    model_fit = SModel(X, y, penalty, λ)
    learn!(model_fit)
    ModelFit(model, model_fit)
end

function predict(model::SparseRegressionModel, model_fit::BaseModelFit, Xnew)
    #print("Predicting using: $(typeof(model))")
    prediction = SparseRegression.predict(model_fit.fit_result, Xnew)
    prediction
end