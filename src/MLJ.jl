module MLJ

export Rows, Cols, Names
export features, X_and_y
export Property, Regressor, TwoClass, MultiClass
export Numeric, Nominal, Weights, NAs
export properties, operations, inputs_can_be, outputs_are
export SupervisedTask, UnsupervisedTask, nrows
export Supervised, Unsupervised
export array

# defined here but extended by files in "interfaces/" (lazily loaded)
export predict, transform, inverse_transform, predict_proba, se, evaluate

# defined in include files:
export partition, @curve, @pcurve                    # "utilities.jl"
export @more, @constant                              # "show.jl"
export rms, rmsl, rmslp1, rmsp                       # "metrics.jl"
export load_boston, load_ames, load_iris, datanow    # "datasets.jl"
export Params, get_params, set_params!, ParamRange   # "parameters.jl"
export KNNRegressor                                  # "builtins/KNN.jl":

# defined in include files "trainable_models.jl" and "networks.jl":
export TrainableModel, NodalTrainableModel, trainable
export source, node, fit!, freeze!, thaw!


# defined in include file "builtins/Transformers.jl":
export FeatureSelector
export ToIntTransformer                     
export UnivariateStandardizer, Standardizer 
# export OneHotEncoder
# export UnivariateBoxCoxTransformer, BoxCoxTransformer
# export DataFrameToArrayTransformer, RegressionTargetTransformer
# export MakeCategoricalsIntTransformer
# export DataFrameToStandardizedArrayTransformer
# export IntegerToInt64Transformer
# export UnivariateDiscretizer, Discretizer

import Requires.@require  # lazy code loading package
import CSV
import DataFrames: DataFrame, AbstractDataFrame, SubDataFrame, eltypes, names
using  Query
import Distributions
import Base.==
import CategoricalArrays

# from Standard Library:
using Statistics
using LinearAlgebra
# using Random


## CONSTANTS

const srcdir = dirname(@__FILE__) # the directory containing this file 
const TREE_INDENT = 2 # indentation for tree-based display of learning networks 
const COLUMN_WIDTH = 24           # for displaying dictionaries with `show`
const DEFAULT_SHOW_DEPTH = 2      # how deep to display fields of `MLJType` objects


## GENERAL PURPOSE UTILITIES

include("utilities.jl")


## ABSTRACT TYPES

# overarching MLJ type:
abstract type MLJType end

# overload `show` method for MLJType (which becomes the fall-back for
# all subtypes):
include("show.jl")

# for storing hyperparameters:
abstract type Model <: MLJType end

abstract type Supervised{R} <: Model end # parameterized by fit-result `R`
abstract type Unsupervised <: Model  end

# tasks:
abstract type Task <: MLJType end 
abstract type Property end # subtypes are the allowable model properties


## LOSS FUNCTIONS

include("metrics.jl")


## UNIVERSAL ADAPTOR FOR DATA CONTAINERS

# Note: By *generic table* we mean any source, supported by Query.jl,
# for which @from iterates over rows. In particular `Matrix` objects
# are not generic tables.

# TODO: Must be a better way to do this?
"""" 
    MLJ.array(X)

Convert a tabular data source `X`, of type supported by Query.jl, into
an `Array`; or, if `X` is an `AbstractArray`, return `X`.

"""
function array(X)
    df= @from row in X begin
        @select row
        @collect DataFrame
    end
    return convert(Array{Float64}, df)
end
array(X::AbstractArray) = X

# For vectors and tabular data containers `df`:
# `df[Rows, r]` gets rows of `df` at `r` (single integer, integer range, or colon)
# `df[Cols, c]` selects features of df at `c` (single integer or symbol, vector of symbols, integer range or colon); not supported for vectors
# `df[Names]` returns names of all features of `df` (or indices if unsupported)

struct Rows end
struct Cols end
struct Names end
struct Eltypes end

Base.getindex(df::AbstractDataFrame, ::Type{Rows}, r) = df[r,:]
Base.getindex(df::AbstractDataFrame, ::Type{Cols}, c) = df[c]
Base.getindex(df::AbstractDataFrame, ::Type{Names}) = names(df)
Base.getindex(df::AbstractDataFrame, ::Type{Eltypes}) = eltypes(df)

# Base.getindex(df::JuliaDB.Table, ::Type{Rows}, r) = df[r]
# Base.getindex(df::JuliaDB.Table, ::Type{Cols}, c) = select(df, c)
# Base.getindex(df::JuliaDB.Table, ::Type{Names}) = getfields(typeof(df.columns.columns))
# Base.getindex(df::JuliaDB.Table, ::Type{Echo}, dg) = dg

Base.getindex(A::AbstractMatrix, ::Type{Rows}, r) = A[r,:]
Base.getindex(A::AbstractMatrix, ::Type{Cols}, c) = A[:,c]
Base.getindex(A::AbstractMatrix, ::Type{Names}) = 1:size(A, 2)
Base.getindex(A::AbstractMatrix{T}, ::Type{Eltypes}) where T = [T for j in 1:size(A, 2)]

Base.getindex(v::AbstractVector, ::Type{Rows}, r) = v[r]
Base.getindex(v::CategoricalArrays.CategoricalArray, ::Type{Rows}, r) = @inbounds v[r]


## MODEL METADATA

# `property(SomeModelType)` is a tuple of instances of:
""" Classfication models with this property allow weighting of the target classes """
struct CanWeightTarget <: Property end
""" Models with this property can provide feature rankings or importance scores """
struct CanRankFeatures <: Property end

# `inputs_can_be(SomeModelType)` and `outputs_are(SomeModelType)` are tuples of
# instances of:
struct Nominal <: Property end
struct Numeric <: Property end
struct NA <: Property end

# additionally, `outputs_are(SomeModelType)` can include:
struct Probababilistic <: Property end
struct Multivariate <: Property end

# for `Model`s with nominal targets (classifiers)
# `outputs_are(SomeModelType)` could also include:
struct Multiclass <: Property end # can handle more than two classes


## CONCRETE TASK SUBTYPES

# TODO: add evaluation metric:
# TODO: add `inputs_can_be` and `outputs_are`
struct UnsupervisedTask <: Task
    data
    ignore::Vector{Symbol}
    operation::Function    # transform, inverse_transform, etc
    properties::Tuple
end

function UnsupervisedTask(
    ; data=nothing
    , ignore=Symbol[]
    , operation=predict
    , properties=())
    
    data != nothing         || throw(error("You must specify data=..."))

    return SupervisedTask(data, ignore, operation, properties)
end

struct SupervisedTask <: Task
    data
    target::Symbol
    ignore::Vector{Symbol}
    operation::Function    # predict, predict_proba, etc
    properties::Tuple
end

function SupervisedTask(
    ; data=nothing
    , target=nothing
    , ignore=Symbol[]
    , operation=predict
    , properties=())
    
    data != nothing         || throw(error("You must specify data=..."))
    target != nothing       || throw(error("You must specify target=..."))
    target in names(data)   || throw(error("Supplied data does not have $target as field."))

    return SupervisedTask(data, target, ignore, operation, properties)
end


## RUDIMENTARY TASK OPERATIONS

Base.length(task::Task) = length(task.data)
Base.size(task::Task) = size(task.data)
nrows(task::Task) = first(size(task))
Base.eachindex(task::Task) = Base.OneTo(nrows(task))

features(task::Task) = filter!(task.data[Names]) do ftr
    !(ftr in task.ignore)
end

features(task::SupervisedTask) = filter(task.data[Names]) do ftr
    ftr != task.target && !(ftr in task.ignore)
end

X_and_y(task::SupervisedTask) = (task.data[Cols, features(task)],
                                 task.data[Cols, task.target])


## SOME LOCALLY ARCHIVED TASKS FOR TESTING AND DEMONSTRATION

include("datasets.jl")


## MODEL INTERFACE 

# Most concrete model types, and their associated low-level methods,
# are defined in package interfaces, located in
# "/src/interfaces.jl". These are are lazily loaded (see the end of
# this file). Built-in model definitions and associated methods (ie,
# ones not dependent on external packages) are contained in
# "/src/builtins.jl"

# every model interface must implement this method, used to generate
# fit-results:
function fit end

# each model interface may optionally overload the following refitting
# method:
update(model::Model, verbosity, fitresult, cache, rows, args...) =
    fit(model, verbosity, rows, args...)

# methods dispatched on a model and fit-result are called *operations*.
# supervised models must implement this operation:
function predict end

# supervised methods may implement this operation:
function predict_proba end

# unsupervised methods must implement this operation:
function transform end

# unsupervised methods may implement this operation:
function inverse_transform end

# operations implemented by some meta-models:
function se end
function evaluate end

# supervised model interfaces buying into introspection should
# implement the following "metadata" methods, dispatched on model
# *type* (see `Properties` below):
function operations end 
function inputs_can_be end
function outputs_are end
function properties end

# a model wishing invalid hyperparameters to be corrected with a
# warning should overload this method (return value is the warning
# message):
clean!(fitresult::Model) = ""

# supervised models may need to overload the following method to
# ensure Tables.jl compliant input data supplied by user is coerced
# into the form required by its `fit` method and operations:
coerce(model::Model, Xtable) = array(Xtable)

# models are `==` if they have the same type and their field values are `==`:
function ==(m1::M, m2::M) where M<:Model
    ret = true
    for fld in fieldnames(M)
        ret = ret && getfield(m1, fld) == getfield(m2, fld)
    end
    return ret
end


## LOAD VARIOUS INTERFACE COMPONENTS

include("trainable_models.jl") 
include("networks.jl")
include("operations.jl")
include("parameters.jl")

## LOAD BUILT-IN MODELS

include("builtins/Transformers.jl")
include("builtins/KNN.jl")


## SETUP LAZY PKG INTERFACE LOADING

# note: presently an MLJ interface to a package, eg `DecisionTree`,
# is not loaded by `using MLJ` alone; one must additionally call
# `import DecisionTree`. 

# files containing a pkg interface must have same name as pkg plus ".jl"

macro load_interface(pkgname, uuid::String, load_instr)
    (load_instr.head == :(=) && load_instr.args[1] == :lazy) ||
        throw(error("Invalid load instruction"))
    lazy = load_instr.args[2]
    filename = joinpath("interfaces", string(pkgname, ".jl"))

    if lazy
        quote
            @require $pkgname=$uuid include($filename)
        end
    else
        quote
            @eval include(joinpath($srcdir, $filename))
        end
    end
end


function __init__()
end

@load_interface DecisionTree "7806a523-6efd-50cb-b5f6-3fa6f1930dbb" lazy=false

end # module

