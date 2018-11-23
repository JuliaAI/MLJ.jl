module MLJ

export Rows, Cols, Names
export features, X_and_y
export Property, Regressor, TwoClass, MultiClass
export Numeric, Nominal, Weights, NAs
export properties, operations, type_of_X, type_of_y
export SupervisedTask, UnsupervisedTask, nrows
export TrainableModel
export array

# defined here but extended by files in "interfaces/" (lazily loaded)
export predict, transform, inverse_transform, predict_proba, se, evaluate

# defined in include files:
export partition, @curve, @pcurve                    # "utilities.jl"
export @more, @constant                              # "show.jl"
export rms, rmsl, rmslp1, rmsp                       # "metrics.jl"
export load_boston, load_ames, load_iris, datanow    # "datasets.jl"
export KNNRegressor                                  # "builtins/KNN.jl":
export node, trainable, fit!, freeze!, thaw!, reload # "networks.jl"

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
import Distributions
import Base.==

# from Standard Library:
using Statistics
using LinearAlgebra


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

abstract type Supervised{E} <: Model end # parameterized by fit-result `E`
abstract type Unsupervised <: Model  end

# tasks:
abstract type Task <: MLJType end 
abstract type Property end # subtypes are the allowable model properties


## LOSS FUNCTIONS

include("metrics.jl")

## UNIVERSAL ADAPTOR FOR DATA CONTAINERS

# TODO: replace with IterationTables interface?

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


## MODEL PROPERTIES

""" Models with this property perform regression """
struct Regression <: Property end    
""" Models with this property perform binary classification """
struct Classification <: Property end
""" Models with this property perform binary and multiclass classification """
struct MultiClass <: Property end
""" Models with this property support nominal (categorical) features """
struct Nominal <: Property end
""" Models with this property support features of numeric type (continuous or ordered factor) """
struct Numeric <: Property end
""" Classfication models with this property allow weighting of the target classes """
struct Weights <: Property end
""" Models with this property support features with missing values """ 
struct NAs <: Property end


## CONCRETE TASK SUBTYPES

# TODO: add evaluation metric:
struct UnsupervisedTask <: Task
    data
    ignore::Vector{Symbol}
    operation::Function    # transform, inverse_transform, etc
    properties::Vector{Property}
end

function UnsupervisedTask(
    ; data=nothing
    , ignore=Symbol[]
    , operation=predict
    , properties=Property[])
    
    data != nothing         || throw(error("You must specify data=..."))
    !isempty(properties)    || @warn "No properties specified for task. "*
                                     "To list properties run `subtypes(Properties)`."
    return SupervisedTask(data, ignore, operation, properties)
end

struct SupervisedTask <: Task
    data
    target::Symbol
    ignore::Vector{Symbol}
    operation::Function    # predict, predict_proba, etc
    properties::Vector{Property}
end

function SupervisedTask(
    ; data=nothing
    , target=nothing
    , ignore=Symbol[]
    , operation=predict
    , properties=Property[])
    
    data != nothing         || throw(error("You must specify data=..."))
    target != nothing       || throw(error("You must specify target=..."))
    target in names(data)   || throw(error("Supplied data does not have $target as field."))
    !isempty(properties)    || @warn "No properties specified for task. "*
                                     "To list properties run `subtypes(Properties)`."
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

# methods to be dispatched on `Model` instances:
function fit end
function update end

# methods to be dispatched on `Model` and a fit-result (*operations*):
function predict end
function predict_proba end
function transform end 
function inverse_transform end
function se end
function evaluate end

# methods to be dispatched on `Model` subtypes:
function operations end
function type_of_nominals end
function type_of_X end
function type_of_y end
function defaults end

# fallback method to correct invalid hyperparameters and return
# a warning (in this case empty):
clean!(fitresult::Model) = ""

# fallback method for refitting:
update(model::Model, verbosity, fitresult, cache, rows, args...) =
    fit(model, verbosity, rows, args...)

# fallback for properties:
properties(model::Model) = Property[]

# models are `==` if they have the same type and their field values are `==`:
function ==(m1::M, m2::M) where M<:Model
    ret = true
    for fld in fieldnames(M)
        ret = ret && getfield(m1, fld) == getfield(m2, fld)
    end
    return ret
end

# TODO: not sure we need this:
"""
    copy(model::Model, fld1=>val1, fld2=>val2, ...)

Return a replica of `model` with the values of fields `fld1`, `fld2`,
... replaced with `val1`, `val2`, ... respectively.

"""
function Base.copy(model::T, field_value_pairs::Vararg{Pair{Symbol}}) where T<:Model
    value_given_field = Dict(field_value_pairs)
    fields = keys(value_given_field)
    constructor_args = map(fieldnames(typeof(model))) do fld
        if fld in fields
            value_given_field[fld]
        else
            getfield(model, fld)
        end
    end
    return T(constructor_args...)
end


## LOAD TRAINABLE MODELS API

include("trainable_models.jl")


## LOAD LEARNING NETWORKS API

include("networks.jl")


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

