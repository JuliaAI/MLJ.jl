#module ML

import StatsBase: predict
import Base: getindex, show
import MLBase: Kfold, LOOCV, fit!, predict
import MLMetrics: accuracy, mean_squared_error
import MLLabelUtils: convertlabel, LabelEnc.MarginBased

# Define BaseModel as an abstract type, all models will belong to this category
abstract type BaseModel end
abstract type BaseModelFit{T<:BaseModel} end

# When we fit a model, we get back a ModelFit object, that has the original model, as well as results, and should not change once fitted
struct ModelFit{T} <: BaseModelFit{T}
    model :: T
    fit_result
end
model(modelFit::ModelFit) = modelFit.model # Accessor function for the family of ModelFit types, instead of directly accessing the field. This way the accessor function is already informed by the type of the model, as it infers it from the type of ModelFit it is accessing, and ends up being faster than using modelFit.model arbitrarily?

"""
Every model has 
- a unique name (that is the model type)
- a fit function (that dispatches based on first input of model type and returns a ModelFit type) and a 
- predict function (that dispatches based on the first input of model type, and second input of ModelFit type)
"""
abstract type DecisionTreeModel <: BaseModel end

mutable struct DecisionTreeClassifier <: DecisionTreeModel
    parameters::Dict # a dictionary of names and values 
end

function DecisionTreeClassifier(model::DecisionTreeClassifier, parameters::Dict)
    load_interface_for(model)
    new(model, parameters)
end

mutable struct DecisionTreeRegressor <: DecisionTreeModel
    parameters::Dict # a dictionary of names and values 
end

function DecisionTreeRegressor(model::DecisionTreeRegressor, parameters::Dict)
    load_interface_for(model)
    new(model, parameters)
end

function load_interface_for{T<:BaseModel}(model::T)
    if isa(model, DecisionTreeModel)
        print("Including library for $(typeof(model))")
        include("interfaces/decisiontree_interface.jl")
    end
end

#end # module