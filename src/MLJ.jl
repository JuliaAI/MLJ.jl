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
export @from_network, machines, sources              # composites.jl

# defined in include files "machines.jl and "networks.jl":
export Machine, NodalMachine, machine, AbstractNode
export source, node, fit!, freeze!, thaw!, Node, sources, origins

# defined in include file "builtins/Transformers.jl":
export FeatureSelector
export UnivariateStandardizer, Standardizer
export UnivariateBoxCoxTransformer
export OneHotEncoder
# export IntegerToInt64Transformer
# export UnivariateDiscretizer, Discretizer

# rexport from Random, Statistics, Distributions, CategoricalArrays:
export pdf, mode, median, mean, shuffle!, categorical, shuffle

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

# for DAG multiprocessing:
import Dagger
import Dagger: Thunk, delayed, compute, collect
const DAGGER_DEBUG = Ref(get(ENV, "MLJ_DAGGER_DEBUG", "0") == "1") # FIXME: Remove me
import MemPool
import UUIDs: UUID, uuid4
import Distributed: @everywhere, @spawnat, workers, fetch
export @notest_logs
macro notest_logs(x...)
    # Disables log tests
    return esc(last(x))
end

# submodules of this module:
include("registry/src/Registry.jl") 
import .Registry

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
include("builtins/ridge.jl") # defines a model for testing only


include("loading.jl") # model metadata processing


## GET THE EXTERNAL MODEL METADATA

function __init__()
    @info "Loading model metadata"
    global metadata_file = joinpath(srcdir, "registry", "Metadata.toml")
    global METADATA = TOML.parsefile(metadata_file)
end


## SIMPLE PLOTTING RECIPE

include("plotrecipes.jl")

end # module
