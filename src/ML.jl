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
        print("Including library for $(typeof(model)) \n")
        include("src/interfaces/decisiontree_interface.jl")
    elseif isa(model, SparseRegressionModel)
        print("Including library for $(typeof(model)) \n")
        include("src/interfaces/glm_interface.jl")
    end
end

function load_interface_for(model::String)
    if model == "SparseRegressionModel"
        print("Including library for "*model*"\n")
        include("src/interfaces/glm_interface.jl")
    end
end

mutable struct SparseRegressionModel <: BaseModel
    parameters
end

"""
 PARAMETERS 

 A parameter set allows a user to add multiple parameters to tune
 It must include a name. Constructor only accepts key arguments
 TODO: parameters cross-checked by learner to see whether they are valid
"""
abstract type Parameter end

"""
    Discrete parameter requires a name and an array of value to check
    TODO: check whether values are correct for specific learner
"""
struct DiscreteParameter <: Parameter
    name::String
    values::Array{Any}
    DiscreteParameter(;name=nothing,values=nothing) = new(name, values)
end

"""
    Tuning of a parameter. Must provide name, lower & upper bound, and transform
    that iterates through values in lower:upper and gives te actual parameter to test

    e.g.
    ```julia
        # Will check λ={1,4,9,16}
        ContinuousParameter("λ", 1, 4, x->x²)
    ```
"""
struct ContinuousParameter <: Parameter
    name::String
    lower::Real
    upper::Real
    transform::Function
    ContinuousParameter(;name=nothing, lower=nothing, upper=nothing, transform=nothing) = new(name, lower, upper, transform)
end

# util functions to get the actual values from the Parameter types.
function get_param_values(p::ContinuousParameter)
    """
    Util function to get the values from the Parameter.
    """
    values = []
        if p.lower < p.upper
            for x in (p.lower:p.upper)
                push!(values, p.transform(x))
            end
            return values
        else
            error("lower value must be lower than upper value")
        end
end

function get_param_values(p::DiscreteParameter)
    """
    Util function to get the values from the Parameter.
    """
    return p.values
end

"""
    Set of parameters.
    Will be used to implement checks on validity of parameters
"""
struct ParametersSet
   parameters::Array{Parameter}
end
getindex(p::ParametersSet, i::Int64) = p.parameters[i]

include("Tuning_.jl")
#end # module