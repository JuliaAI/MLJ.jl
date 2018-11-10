module MLJ

export Rows, Cols, Names
export features, X_and_y
export array

# defined here but extended by files in "interfaces/" (lazily loaded)
export predict, transform, inverse_transform

# defined in include files:
export partition, @curve, @pcurve                  # "utilities.jl"
export @more, @constant                            # "show.jl"
export rms, rmsl, rmslp1, rmsp                     # "metrics.jl"
export load_boston, load_ames, load_iris, datanow  # "datasets.jl"
export KNNRegressor                                # "builtins/KNN.jl":

# defined in include file "builtins/Transformers.jl":
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

# from Standard Library:
using Statistics
using LinearAlgebra


## CONSTANTS

const srcdir = dirname(@__FILE__) # the directory containing this file 
const TREE_INDENT = 2 # indentation for tree-based display of dynamic data 
const COLUMN_WIDTH = 24           # for displaying dictionaries with `show`
const DEFAULT_SHOW_DEPTH = 2      # how deep to display fields of `MLJType` objects


## GENERAL PURPOSE UTILITIES

include("Utilities.jl")


## ABSTRACT TYPES

# overarching MLJ type:
abstract type MLJType end

# overload `show` method for MLJType (which becomes the fall-back for
# all subtypes):
include("show.jl")

# for storing hyperparameters:
abstract type Model <: MLJType end 

abstract type Learner <: Model end

# a model type for transformers
abstract type Transformer <: Model end 

# special learners:
abstract type Supervised{E} <: Learner end # parameterized by fit-result `E`
abstract type Unsupervised{E} <: Learner end

# special supervised learners:
abstract type Regressor{E} <: Supervised{E} end
abstract type Classifier{E} <: Supervised{E} end

# tasks:
abstract type Task <: MLJType end 


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
struct Echo end # needed to terminate calls of dynamic data types on unseen source data

Base.getindex(df::AbstractDataFrame, ::Type{Rows}, r) = df[r,:]
Base.getindex(df::AbstractDataFrame, ::Type{Cols}, c) = df[c]
Base.getindex(df::AbstractDataFrame, ::Type{Names}) = names(df)
Base.getindex(df::AbstractDataFrame, ::Type{Eltypes}) = eltypes(df)
Base.getindex(df::AbstractDataFrame, ::Type{Echo}, dg) = dg

# Base.getindex(df::JuliaDB.Table, ::Type{Rows}, r) = df[r]
# Base.getindex(df::JuliaDB.Table, ::Type{Cols}, c) = select(df, c)
# Base.getindex(df::JuliaDB.Table, ::Type{Names}) = getfields(typeof(df.columns.columns))
# Base.getindex(df::JuliaDB.Table, ::Type{Echo}, dg) = dg

Base.getindex(A::AbstractMatrix, ::Type{Rows}, r) = A[r,:]
Base.getindex(A::AbstractMatrix, ::Type{Cols}, c) = A[:,c]
Base.getindex(A::AbstractMatrix, ::Type{Names}) = 1:size(A, 2)
Base.getindex(A::AbstractMatrix{T}, ::Type{Eltypes}) where T = [T for j in 1:size(A, 2)]
Base.getindex(A::AbstractMatrix, ::Type{Echo}, B) = B

Base.getindex(v::AbstractVector, ::Type{Rows}, r) = v[r]
Base.getindex(v::AbstractVector, ::Type{Echo}, w) = w


## CONCRETE TASK TYPES

struct SupervisedTask <: Task
    kind::Symbol
    data
    target::Symbol
    ignore::Vector{Symbol}
end

function SupervisedTask(
    ; kind=nothing
    , data=nothing
    , target=nothing
    , ignore=Symbol[])
    
    kind != nothing         || throw(error("You must specfiy kind=..."))
    data != nothing         || throw(error("You must specify data=..."))
    target != nothing       || throw(error("You must specify target=..."))
    target in names(data)   || throw(error("Supplied data does not have $target as field."))
    return SupervisedTask(kind, data, target, ignore)

end

ClassificationTask(; kwargs...) = SupervisedTask(; kind=:classification, kwargs...)
RegressionTask(; kwargs...)     = SupervisedTask(; kind=:regression, kwargs...)


## RUDIMENTARY TASK OPERATIONS

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

## LOW-LEVEL MODEL METHODS 

# Most concrete model types, and their associated low-level methods,
# are defined in package interfaces, located in
# "/src/interfaces.jl". These are are lazily loaded (see the end of
# this file). Built-in model definitions and associated methods (ie,
# ones not dependent on external packages) are contained in
# "/src/builtins.jl"

# low-level methods to be extended:
function fit end
function fit2 end
function predict end
function predict_proba end
function transform end 
function inverse_transform end

# fallback method to correct invalid hyperparameters and return
# a warning (in this case empty):
clean!(fitresult::Model) = ""

# fallback method for refitting:
fit2(model::Model, verbosity, fitresult, cache, args...) =
    fit(model, verbosity, args...)


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
    @load_interface DecisionTree "7806a523-6efd-50cb-b5f6-3fa6f1930dbb" lazy=false
end

end # module



    
