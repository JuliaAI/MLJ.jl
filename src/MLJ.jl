module MLJ

export Rows, Cols, Names
export features, X_and_y
export SupervisedTask, UnsupervisedTask, nrows
export Supervised, Unsupervised
export matrix

# defined here but extended by files in "interfaces/" (lazily loaded)
export predict, transform, inverse_transform, predict_proba, se, evaluate, best

# defined in include files:
export partition, @curve, @pcurve, readlibsvm        # "utilities.jl"
export @more, @constant                              # "show.jl"
export rms, rmsl, rmslp1, rmsp                       # "metrics.jl"
export load_boston, load_ames, load_iris, datanow    # "datasets.jl"
export SimpleCompositeRegressor                      # "composites.jl"
export Holdout, CV, Resampler                        # "resampling.jl"
export Params, params, set_params!               # "parameters.jl"
export strange, iterator                             # "parameters.jl"
export Grid, TunedModel                              # "tuning.jl"
export ConstantRegressor, ConstantClassifier         # "builtins/Constant.jl
export KNNRegressor                                  # "builtins/KNN.jl":

# defined in include files "machines.jl" and "networks.jl":
export Machine, NodalMachine, machine
export source, node, fit!, freeze!, thaw!


# defined in include file "builtins/Transformers.jl":
export FeatureSelector
export ToIntTransformer
export UnivariateStandardizer, Standardizer
export UnivariateBoxCoxTransformer
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
using CategoricalArrays
import TableTraits

# from Standard Library:
using Statistics
using LinearAlgebra
# using Random


## CONSTANTS

const srcdir = dirname(@__FILE__) # the directory containing this file
# const TREE_INDENT = 2 # indentation for tree-based display of learning networks
const COLUMN_WIDTH = 24           # for displaying dictionaries with `show`
const DEFAULT_SHOW_DEPTH = 1      # how deep to display fields of `MLJType` objects


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


## METRICS

include("metrics.jl")


## DATA INTERFACES

"""
    CategoricalDecoder(C::CategoricalArray; eltype=nothing)

Construct a decoder for transforming a `CategoricalArray{T}` object
into an ordinary array, and for re-encoding similar arrays back into a
`CategoricalArray{T}` object having the same `pool` (and, in
particular, the same levels) as `C`. If `eltype` is not specified then
the element type of the transformed array is `T`. Otherwise, the
element type is `eltype` and the elements are promotions of the
internal (integer) `ref`s of the `CategoricalArray`. One
must have `R <: eltype <: Real` where `R` is the reference type of the
`CategoricalArray` (usually `UInt32`).

    transform(decoder::CategoricalDecoder, C::CategoricalArray)

Transform `C` into an ordinary `Array`.

    inverse_transform(decoder::CategoricalDecoder, A::Array)

Transform an array `A` suitably compatible with `decoder` into a
`CategoricalArray` having the same `pool` as `C`.

### Example

````
julia> using CategoricalArrays
julia> C = categorical(["a" "b"; "a" "c"])
2×2 CategoricalArray{String,2,UInt32}:
 "a"  "b"
 "a"  "c"

julia> decoder = MLJ.CategoricalDecoder(C, eltype=Float64);
julia> A = transform(decoder, C)
2×2 Array{Float64,2}:
 1.0  2.0
 1.0  3.0

julia> inverse_transform(decoder, A[1:1,:])
1×2 CategoricalArray{String,2,UInt32}:
 "a"  "b"

julia> levels(ans)
3-element Array{String,1}:
 "a"
 "b"
 "c"

"""
struct CategoricalDecoder{I<:Real,T,N,R<:Integer}  # I the output eltype
    pool::CategoricalPool{T,R} # abstract type, not optimal
    use_original_type::Bool
    CategoricalDecoder{I,T,N,R}(X::CategoricalArray{T,N,R}, use_original_type) where {I,T,N,R}  =
        new(X.pool, use_original_type)
end

function CategoricalDecoder(X::CategoricalArray{T,N,R}; eltype=nothing) where {T,N,R}
    if eltype ==  nothing
        eltype = R # any integer type will do here; not used
        use_original_type = true
    else
        use_original_type = false
    end
    return CategoricalDecoder{eltype,T,N,R}(X, use_original_type)
end

function transform(decoder::CategoricalDecoder{I,T,N,R}, C::CategoricalArray) where {I,T,N,R}
    if decoder.use_original_type
        return collect(C)
    else
        return broadcast(C.refs) do element
            ref = convert(I, element)
        end
    end
end

function inverse_transform(decoder::CategoricalDecoder{I,T,N,R}, A::Array{J}) where {I,T,N,R,J<:Union{I,T}}
    if decoder.use_original_type
        refs = broadcast(A) do element
            decoder.pool.invindex[element]
        end
    else
        refs = broadcast(A) do element
            round(R, element)
        end
    end
    return CategoricalArray{T,N}(refs, decoder.pool)

end

# Some of what follows is a poor-man's stab at agnostic data
# containers. When the Queryverse columns-view interface becomes
# widely implemented, a better solution, removing specific container
# dependencies, will be possible.

""""
    MLJ.matrix(X)

Convert an iteratable table source `X` into an `Matrix`; or, if `X` is
a `Matrix`, return `X`.

"""
function matrix(X)
    TableTraits.isiterabletable(X) || error("Argument is not an iterable table.")

    df= @from row in X begin
        @select row
        @collect DataFrame
    end
    return convert(Matrix, df)

end
matrix(X::Matrix) = X

struct Rows end
struct Cols end
struct Names end
struct Eltypes end

# select rows of any iterable table `X` with `X[Rows, r]`:
function Base.getindex(X::T, ::Type{Rows}, r) where T

    TableTraits.isiterabletable(X) || error("Argument is not an iterable table.")

    row_iterator = @from row in X begin
        @select row
        @collect
    end
                    
    return @from row in row_iterator[r] begin
        @select row
        @collect T
    end

end

#Base.getindex(df::AbstractDataFrame, ::Type{Rows}, r) = df[r,:]
Base.getindex(df::AbstractDataFrame, ::Type{Cols}, c) = df[c]
Base.getindex(df::AbstractDataFrame, ::Type{Names}) = names(df)
Base.getindex(df::AbstractDataFrame, ::Type{Eltypes}) = eltypes(df)
nrows(df::AbstractDataFrame) = size(df, 1)

#Base.getindex(df::JuliaDB.NextTable, ::Type{Rows}, r) = df[r]
#Base.getindex(df::JuliaDB.NextTable, ::Type{Cols}, c) = select(df, c)
#Base.getindex(df::JuliaDB.NextTable, ::Type{Names}) = getfields(typeof(df.columns.columns))
# nrows(df::JuliaDB.NextTable) = length(df)

Base.getindex(A::AbstractMatrix, ::Type{Rows}, r) = A[r,:]
Base.getindex(A::AbstractMatrix, ::Type{Cols}, c) = A[:,c]
Base.getindex(A::AbstractMatrix, ::Type{Names}) = 1:size(A, 2)
Base.getindex(A::AbstractMatrix{T}, ::Type{Eltypes}) where T = [T for j in 1:size(A, 2)]
nrows(A::AbstractMatrix) = size(A, 1)

Base.getindex(v::AbstractVector, ::Type{Rows}, r) = v[r]
Base.getindex(v::CategoricalArrays.CategoricalArray{T,1,S} where {T,S}, ::Type{Rows}, r) = @inbounds v[r]
nrows(v::AbstractVector) = length(v)


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

# every model interface must implement this method, used to generate
# fit-results:
function fit end

# each model interface may optionally overload the following refitting
# method:
update(model::Model, verbosity, fitresult, cache, args...) =
    fit(model, verbosity, args...)

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
function best end

# models buying into introspection should
# implement the following method, dispatched on model
# *type*:
function metadata end

# a model wishing invalid hyperparameters to be corrected with a
# warning should overload this method (return value is the warning
# message):
clean!(fitresult::Model) = ""

# supervised models may need to overload the following method to
# ensure iterable tables compliant input data supplied by user is coerced
# into the form required by its `fit` method and operations:
coerce(model::Model, Xtable) = Xtable

# models are `==` if they have the same type and their field values are `==`:
function ==(m1::M, m2::M) where M<:Model
    ret = true
    for fld in fieldnames(M)
        ret = ret && getfield(m1, fld) == getfield(m2, fld)
    end
    return ret
end


## LOAD VARIOUS INTERFACE COMPONENTS

include("machines.jl")
include("networks.jl")
include("composites.jl")
include("operations.jl")
include("resampling.jl")
include("parameters.jl")
include("tuning.jl")


## LOAD BUILT-IN MODELS

include("builtins/Transformers.jl")
include("builtins/Constant.jl")
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
    @load_interface DecisionTree "7806a523-6efd-50cb-b5f6-3fa6f1930dbb" lazy=true
    @load_interface  MultivariateStats "6f286f6a-111f-5878-ab1e-185364afe411" lazy=true
end

#@load_interface XGBoost "009559a3-9522-5dbb-924b-0b6ed2b22bb9" lazy=false

end # module
