module MLJ

# defined in include files:
export @curve, @pcurve                               # "utilities.jl"
export rms, rmsl, rmslp1, rmsp                       # "metrics.jl"
export misclassification_rate, cross_entropy         # "metrics.jl"
export default_measure                               # "metrics.jl"
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

# reexport from MLJBase:
export SupervisedTask, UnsupervisedTask
export Deterministic, Probabilistic, Unsupervised, Supervised
export predict, predict_mean, predict_median, predict_mode
export transform, inverse_transform, se, evaluate
export @constant, @more, HANDLE_GIVEN_ID, UnivariateNominal
export partition, X_and_y
export load_boston, load_ames, load_iris, load_reduced_ames
export load_crabs, datanow                
export features, X_and_y

using MLJBase

# to be extended:
import MLJBase: fit, update, clean!, info
import MLJBase: predict, predict_mean, predict_median, predict_mode
import MLJBase: transform, inverse_transform, se, evaluate

using RemoteFiles
import TOML
import Requires.@require  # lazy code loading package
using  CategoricalArrays
import Distributions: pdf, mode
import Distributions
import StatsBase
using ProgressMeter
import Tables

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
include("machines.jl")      # machine API
include("networks.jl")      # for building learning networks
include("composites.jl")    # composite models, incl. learning networks exported as models
include("operations.jl")    # syntactic sugar for operations (predict, transform, etc)
include("resampling.jl")    # evaluating models by assorted resampling strategies
include("parameters.jl")    # hyper-parameter range constructors and nested hyper-parameter API
include("tuning.jl")
include("ensembles.jl")     # homogeneous ensembles


## LOAD BUILT-IN MODELS

include("builtins/Transformers.jl")
include("builtins/Constant.jl")
include("builtins/KNN.jl")
include("builtins/LocalMultivariateStats.jl")


## GET THE EXTERNAL MODEL METADATA AND MERGE WITH MLJ MODEL METADATA

include("loading.jl")      # model metadata processing



end # module
