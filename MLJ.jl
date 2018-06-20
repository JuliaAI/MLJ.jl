import StatsBase: predict
import Base: getindex, show
import MLBase: Kfold, LOOCV, fit!, predict
import MLMetrics: accuracy, mean_squared_error

"""
    Contains task type (regression,classification,..)
    and the columns to use as target and as features.
    TODO: accept multiple targets
"""
abstract type Task end

immutable RegressionTask <: Task
    targets::Array{<:Integer}
    features::Array{Int}
    data::Matrix{<:Real}
end
immutable ClassificationTask <:Task
    targets::Array{<:Integer}
    features::Array{Int}
    data::Matrix{<:Real}
end

function Task(;task_type=:regression, targets=nothing, data=nothing::Matrix{<:Real})
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
        error("Only regression and classification tasks are currently available")
    end
    task
end


"""
    Allows resampling for cross validation
    TODO: add more methods (only accepts k-fold)
"""
immutable Resampling
    method::String
    iterations::Int
    Resampling() = new("KFold", 3)
    Resampling(name::String) = new(name, 0)
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
immutable DiscreteParameter <: Parameter
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
immutable ContinuousParameter <: Parameter
    name::String
    lower::Real
    upper::Real
    transform::Function
    ContinuousParameter(;name=nothing, lower=nothing, upper=nothing, transform=nothing) = new(name, lower, upper, transform)
end

"""
    Set of parameters.
    Will be used to implement checks on validity of parameters
"""
immutable ParametersSet
   parameters::Array{Parameter}
end
getindex(p::ParametersSet, i::Int64) = p.parameters[i]


"""
    Abstraction layer for model
"""
immutable MLRModel{T}
    model::T
    parameters
    inplace::Bool
end
MLRModel(model, parameters; inplace=true::Bool) = MLRModel(model, parameters, inplace)


"""
    Contains the name and the parameters of the model to train.
"""
abstract type Learner end

immutable ModelLearner <: Learner
    name::String
    parameters::Union{Void, Dict, ParametersSet}
    modelᵧ::Union{Void, MLRModel}
    ModelLearner(learner::String) = new(learner, nothing)
    ModelLearner(learner::String, parameters) = new(learner, parameters, nothing)
    ModelLearner(learner::Learner, modelᵧ::MLRModel) = new(learner.name, learner.parameters, modelᵧ)
    ModelLearner(learner::Learner, modelᵧ::MLRModel, parameters::ParametersSet) = new(learner.name, parameters, modelᵧ)

end


global const MAJORITY = 1

"""
    Stacking learner. Must be used with CompositeLearner{T}.
    @vars
        vote_type: type of voting, currently only "majority" accepted
"""
immutable Stacking
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
mutable struct MLRStorage
    models::Array{<:Any,1}
    measures::Array{<:Any,1}
    averageCV::Array{<:Float64,1}
    parameters::Array{<:Dict,1}
    MLRStorage() = new([],[],Array{Float64}(0),Array{Dict}(0))
end



mutable struct MLRMultiplex
    learners::Array{Learner}
    parametersSets::Array{ParametersSet}
    size::Integer
    MLRMultiplex(lrns::Array{Learner}, ps::Array{ParametersSet}) = new(lrns, ps, size(lrns,1))
end


"""
    Constructor for any model. Will call the function makeModelname, where
    modelname is stored in learner.name
    Function makeModelname should be defined separately for each model
"""
function MLRModel(learner::Learner, task::Task)
    # Calls function with name "makeModelname"
    f_name = learner.name
    f_name = "make" * titlecase(f_name)

    f = getfield(Main, Symbol(f_name))
    f(learner, task)
end

"""
    Function which sets up model given by learner, and then calls model-based
    learning function, which must be defined separately for each model.
"""
function learnᵧ(learner::Learner, task::Task)
    modelᵧ = MLRModel(learner, task)
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
                data_features::Matrix, task::Task)

    predictᵧ(learner.modelᵧ, data_features, task)
end


"""
    Import specific wrapper
"""
function load(wrapper::AbstractString)
    include("wrappers/"*wrapper*"_wrapper.jl")
end

"""
    Loads all wrappers in folder "wrapper"
"""
function loadAll()
    for wrapper in readdir("wrappers")
        include(wrapper)
    end
end

include("Tuning.jl")
include("Stacking.jl")
include("Resampling.jl")
include("Storage.jl")
include("Utilities.jl")
