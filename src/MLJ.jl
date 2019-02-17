module MLJ

export features, X_and_y
export SupervisedTask, UnsupervisedTask
export Supervised, Unsupervised, Deterministic, Probabilistic

# defined in include files:
export partition, @curve, @pcurve, readlibsvm        # "utilities.jl"
export rms, rmsl, rmslp1, rmsp                       # "metrics.jl"
export misclassification_rate, cross_entropy         # "metrics.jl"
export default_measure                               # "metrics.jl"
export load_boston, load_ames, load_iris             # "datasets.jl"
export load_crabs, datanow                           # "datasets.jl"
export Holdout, CV, evaluate!, Resampler             # "resampling.jl"
export Params, params, set_params!                   # "parameters.jl"
export strange, iterator                             # "parameters.jl"
export Grid, TunedModel, learning_curve              # "tuning.jl"
export DeterministicEnsembleModel                    # "ensembles.jl"
export ProbabilisticEnsembleModel                    # "ensembles.jl"
export EnsembleModel                                 # "ensembles.jl"
export ConstantRegressor, ConstantClassifier         # "builtins/Constant.jl
export DeterministicConstantRegressor                # "builtins/Constant.jl
export DeterministicConstantClassifier               # "builtins/Constant.jl
export models, @load                                  # "loading.jl"
export KNNRegressor                                  # "builtins/KNN.jl"
export RidgeRegressor, PCA                           # "builtins/LocalMulitivariateStats.jl

# defined in include files "machines.jl" and "networks.jl":
export Machine, NodalMachine, machine
export source, node, fit!, freeze!, thaw!, Node

# defined in include file "builtins/Transformers.jl":
export FeatureSelector
export ToIntTransformer
export UnivariateStandardizer, Standardizer
export UnivariateBoxCoxTransformer
export OneHotEncoder
# export IntegerToInt64Transformer
# export UnivariateDiscretizer, Discretizer

# rexport from Statistics, Distributions:
export pdf, mode, median, mean, info

export dataframe # utilities.jl

# reexport from MLJBase:
export predict, predict_mean, predict_median, predict_mode
export transform, inverse_transform, se, evaluate
export @constant, @more, HANDLE_GIVEN_ID, UnivariateNominal

# to be extended:
import MLJBase: fit, update, clean!, info
import MLJBase: predict, predict_mean, predict_median, predict_mode
import MLJBase: transform, inverse_transform, se, evaluate

# to be used
import MLJBase: schema, selectrows, selectcols, matrix
import MLJBase: @constant, @more, HANDLE_GIVEN_ID, UnivariateNominal
import MLJBase: average
import MLJBase: fitresult_type

using MLJBase

import MLJRegistry
# import MLJModels

import Requires.@require  # lazy code loading package
using  CategoricalArrays
import CSV
import DataFrames: DataFrame, AbstractDataFrame, SubDataFrame, eltypes, names
import Distributions: pdf, mode
import Distributions
import StatsBase
using ProgressMeter
import Tables
import Pkg
# import TOML

# to be extended:
import Base.==


# from Standard Library:
using Statistics
using LinearAlgebra
using Random
import Distributed: @distributed, nworkers, pmap

const srcdir = dirname(@__FILE__) # the directory containing this file:

include("utilities.jl")     # general purpose utilities
include("metrics.jl")       # loss functions
include("tasks.jl")
include("datasets.jl")      # locally archived tasks for testing and demos
include("machines.jl")      # machine API
include("networks.jl")      # for building learning networks
include("composites.jl")    # composite models, incl. learning networks exported as models
include("operations.jl")    # syntactic sugar for operations (predict, transform, predict_mean, etc.)
include("resampling.jl")    # evaluating models by assorted resampling strategies
include("parameters.jl")    # hyper-parameter range constructors and nested hyper-parameter API
include("tuning.jl")
include("ensembles.jl")     # homogeneous ensembles
include("loading.jl")      # model metadata processing

## LOAD BUILT-IN MODELS

include("builtins/Transformers.jl")
include("builtins/Constant.jl")
include("builtins/KNN.jl")
include("builtins/LocalMultivariateStats.jl")

## SETUP LAZY PKG INTERFACE LOADING (a temporary hack)

# Note: Presently an MLJ interface to a package, eg `DecisionTree`,
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
#    @load_interface DecisionTree "7806a523-6efd-50cb-b5f6-3fa6f1930dbb" lazy=true
#    @load_interface Clustering "aaaa29a8-35af-508c-8bc3-b662a17a0fe5" lazy=true
#    @load_interface GLM "38e38edf-8417-5370-95a0-9cbb8c7f171a" lazy=true
#    @load_interface GaussianProcesses "891a1506-143c-57d2-908e-e1f8e92e6de9" lazy=true
end


end # module
