module MLJ

export  fit, predict, model, tune, load_interface_for,
        DecisionTreeClassifier, DecisionTreeRegressor, SparseRegressionModel, ModelFit, BaseModelFit,
        DiscreteParameter, ContinuousParameter, ParametersSet

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

# Define a generic predict for BaseModelFit, that disambiguates them based on what Model they are the result of
predict(modelFit::BaseModelFit, Xnew) = predict(model(modelFit), modelFit, Xnew)

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

mutable struct SparseRegressionModel <: BaseModel
    parameters
end

"""
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
    Tuning of a parameter. Must provide name, lower&upper bound, and transform
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

# Begin of OLD CODE
"""
    Contains task type (regression,classification,..)
    and the columns to use as target and as features.
    TODO: accept multiple targets
"""
abstract type MLTask end

struct RegressionTask <: MLTask
    targets::Array{<:Integer}
    features::Array{Int}
    data::Matrix{<:Real}
end

struct ClassificationTask <:MLTask
    targets::Array{<:Integer}
    features::Array{Int}
    data::Matrix{<:Real}
end

function MLTask(;task_type=:regression, targets=nothing, data=nothing::Matrix{<:Real})
    if targets == nothing || data == nothing
        throw("Requires target and data to be set")
    end

    # reshapes features without target
    # TODO: optimise this step
    features = size(data,2)
    features = deleteat!( collect(1:features), targets)

    # Transforms target into vector
    if typeof(targets) <: Integer
        targets = [targets]
    end

    # TODO: How to make this general so that new task types can be added easily?
    if task_type == :regression
        task = RegressionTask(targets, features, data)
    elseif task_type == :classification
        task = ClassificationTask(targets, features, data)
    else
        error("Only :regression and :classification tasks are currently available")
    end
    task
end

"""
    Allows resampling for cross validation
    TODO: add more methods (only accepts k-fold)
"""
struct Resampling
    method::String
    iterations::Int
    Resampling() = new("KFold", 3)
    Resampling(name::String) = new(name, 0)
end


"""
    Abstraction layer for model
"""
struct MLJModel{T}
    model::T
    parameters
    inplace::Bool
end
MLJModel(model, parameters; inplace=true::Bool) = MLJModel(model, parameters, inplace)

# Contains the name and the parameters of the model to train.

abstract type Learner end

struct ModelLearner <: Learner
    name::Symbol
    parameters::Union{Void, Dict, ParametersSet}
    modelᵧ::Union{Void, MLJModel}
    ModelLearner(learner::Symbol) = new(learner, nothing)
    ModelLearner(learner::Symbol, parameters) = new(learner, parameters, nothing)
    ModelLearner(learner::Learner, modelᵧ::MLJModel) = new(learner.name, learner.parameters, modelᵧ)
    ModelLearner(learner::Learner, modelᵧ::MLJModel, parameters::ParametersSet) = new(learner.name, parameters, modelᵧ)

end

global const MAJORITY = 1

"""
    Stacking learner. Must be used with CompositeLearner{T}.
    @vars
        vote_type: type of voting, currently only "majority" accepted
"""
struct Stacking
    voting_type::Integer
end

mutable struct CompositeLearner{T} <: Learner
    composite::T
    learners::Array{<:Learner}
end

function show(io::IO,l::ModelLearner)
    println("Learner: $(l.name)")
    if typeof(l.parameters) <: Dict
        for (key, value) in l.parameters
           println(" ▁ ▂ ▃ $key: $value")
        end
    end
end

"""
    Structure used to record results of tuning
"""
mutable struct MLJStorage
    models::Array{<:Any,1}
    measures::Array{<:Any,1}
    averageCV::Array{<:Float64,1}
    parameters::Array{<:Dict,1}
    MLJStorage() = new([],[],Array{Float64}(0),Array{Dict}(0))
end

mutable struct MLJMultiplex
    learners::Array{Learner}
    parametersSets::Array{ParametersSet}
    size::Integer
    MLJMultiplex(lrns::Array{Learner}, ps::Array{ParametersSet}) = new(lrns, ps, size(lrns,1))
end


"""
    Constructor for any model. Will call the function makeModelname, where
    modelname is stored in learner.name
    Function makeModelname should be defined separately for each model
"""
function MLJModel(learner::Learner, task::MLTask)
    # Calls function with name "makeModelname"
    f_name = learner.name
    f_name = "make" * titlecase(String(f_name))

    f = getfield(Main, Symbol(f_name))
    f(learner, task)
end

"""
    Function which sets up model given by learner, and then calls model-based
    learning function, which must be defined separately for each model.
"""
function learnᵧ(learner::Learner, task::MLTask)
    modelᵧ = MLJModel(learner, task)
    if modelᵧ.inplace
        learnᵧ!(modelᵧ, learner, task)
    else
        modelᵧ = learnᵧ(modelᵧ, learner, task)
    end
    modelᵧ
end

"""
    Allows to predict using learner instead of model.
"""
function predictᵧ(learner::ModelLearner,
                data_features::Matrix, task::MLTask)

    predictᵧ(learner.modelᵧ, data_features, task)
end


"""
    Import specific wrapper
"""
function load(wrapper::AbstractString)
    include("src/wrappers/"*wrapper*"_wrapper.jl")
end

"""
    Loads all wrappers in folder "wrapper"
"""
function loadAll()
    for wrapper in readdir("src/wrappers")
        include(wrapper)
    end
end
# End of OLD CODE
    
include("Tuning.jl")
include("Stacking.jl")
include("Resampling.jl")
include("Storage.jl")
include("Utilities.jl")

end # module