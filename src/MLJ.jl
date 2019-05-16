module MLJ

# defined in include files:
export @curve, @pcurve                               # utilities.jl
export mav, rms, rmsl, rmslp1, rmsp                  # metrics.jl
export misclassification_rate, cross_entropy         # metrics.jl
export default_measure                               # metrics.jl
export coerce, supervised, unsupervised              # tasks.jl
export report                                        # machines.jl
export Holdout, CV, evaluate!, Resampler             # resampling.jl
export Params, params, set_params!                   # parameters.jl
export strange, iterator                             # parameters.jl
export Grid, TunedModel, learning_curve!             # tuning.jl
export EnsembleModel                                 # ensembles.jl
export ConstantRegressor, ConstantClassifier         # builtins/Constant.jl
export models, localmodels, @load                    # loading.jl
export KNNRegressor                                  # builtins/KNN.jl

# defined in include files "machines.jl and "networks.jl":
export Machine, NodalMachine, machine, AbstractNode
export source, node, fit!, freeze!, thaw!, Node, sources

# defined in include file "builtins/Transformers.jl":
export FeatureSelector
export UnivariateStandardizer, Standardizer
export UnivariateBoxCoxTransformer
export OneHotEncoder
# export IntegerToInt64Transformer
# export UnivariateDiscretizer, Discretizer

# rexport from Random, Statistics, Distributions, CategoricalArrays:
export pdf, mode, median, mean, shuffle!, categorical

# reexport from MLJBase:
export nrows, nfeatures, info
export SupervisedTask, UnsupervisedTask, MLJTask
export Deterministic, Probabilistic, Unsupervised, Supervised
export DeterministicNetwork, ProbabilisticNetwork
export Found, Continuous, Finite, Infinite    
export OrderedFactor, Unknown
export Count, Multiclass, Binary
export scitype, scitype_union, scitypes
export predict, predict_mean, predict_median, predict_mode
export transform, inverse_transform, se, evaluate, fitted_params
export @constant, @more, HANDLE_GIVEN_ID, UnivariateFinite
export partition, X_and_y
export load_boston, load_ames, load_iris, load_reduced_ames
export load_crabs, datanow                
export features, X_and_y

using MLJBase

# to be extended:
import MLJBase: fit, update, clean!
import MLJBase: predict, predict_mean, predict_median, predict_mode
import MLJBase: transform, inverse_transform, se, evaluate, fitted_params
import MLJBase: show_as_constructed, params

using RemoteFiles
import Pkg.TOML
#import Requires.@require  # lazy code loading package
using  CategoricalArrays
import Distributions: pdf, mode
import Distributions
import StatsBase
using ProgressMeter
import Tables

# to be extended:
import Base.==
import StatsBase.fit!

# from Standard Library:
using Statistics
using LinearAlgebra
using Random
import Distributed: @distributed, nworkers, pmap
using RecipesBase # for plotting

const srcdir = dirname(@__FILE__) # the directory containing this file:
const CategoricalElement = Union{CategoricalString,CategoricalValue}

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
include("tasks.jl")         # enhancements to task interface defined in MLJBase


## LOAD BUILT-IN MODELS

include("builtins/Transformers.jl")
include("builtins/Constant.jl")
include("builtins/KNN.jl")
include("builtins/ridge.jl")

## GET THE EXTERNAL MODEL METADATA AND MERGE WITH MLJ MODEL METADATA

include("loading.jl")      # model metadata processing


## SIMPLE PLOTTING RECIPE

include("plotrecipes.jl")

end # module
