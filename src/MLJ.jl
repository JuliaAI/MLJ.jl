module MLJ

# defined in include files:
export @curve, @pcurve,                               # utilities.jl
        mav, mae, rms, rmsl, rmslp1, rmsp, l1, l2,     # measures.jl
        misclassification_rate, cross_entropy,        # measures.jl
        default_measure,                              # measures.jl
        coerce, supervised, unsupervised,             # tasks.jl
        report,                                       # machines.jl
        Holdout, CV, evaluate!, Resampler,            # resampling.jl
        Params, params, set_params!,                  # parameters.jl
        strange, iterator,                            # parameters.jl
        Grid, TunedModel, learning_curve!,            # tuning.jl
        EnsembleModel,                                # ensembles.jl
        ConstantRegressor, ConstantClassifier,        # builtins/Constant.jl
        models, localmodels, @load,                   # loading.jl
        KNNRegressor,                                 # builtins/KNN.jl
        @from_network, machines, sources, anonymize!, # composites.jl
        fitresults                                    # composites.jl

# defined in include files "machines.jl and "networks.jl":
export Machine, NodalMachine, machine, AbstractNode,
        source, node, fit!, freeze!, thaw!, Node, sources, origins

# defined in include file "builtins/Transformers.jl":
export FeatureSelector,
        UnivariateStandardizer, Standardizer,
        UnivariateBoxCoxTransformer,
        OneHotEncoder
        # IntegerToInt64Transformer,
        # UnivariateDiscretizer, Discretizer

# rexport from Random, Statistics, Distributions, CategoricalArrays:
export pdf, mode, median, mean, shuffle!, categorical, shuffle, levels, levels!

# reexport from MLJBase and ScientificTypes:
export nrows, nfeatures, info,
        SupervisedTask, UnsupervisedTask, MLJTask,
        Deterministic, Probabilistic, Unsupervised, Supervised,
        DeterministicNetwork, ProbabilisticNetwork,
        Found, Continuous, Finite, Infinite,
        OrderedFactor, Unknown,
        Count, Multiclass, Binary, Scientific,
        scitype, scitype_union, schema,
        predict, predict_mean, predict_median, predict_mode,
        transform, inverse_transform, se, evaluate, fitted_params,
        @constant, @more, HANDLE_GIVEN_ID, UnivariateFinite,
        partition, X_and_y,
        load_boston, load_ames, load_iris, load_reduced_ames,
        load_crabs, datanow,
        features, X_and_y

using MLJBase
# to be extended:
import MLJBase: fit, update, clean!,
                predict, predict_mean, predict_median, predict_mode,
                transform, inverse_transform, se, evaluate, fitted_params,
                show_as_constructed, params

using Requires
import Pkg.TOML
using  CategoricalArrays
import Distributions: pdf, mode
import Distributions
import StatsBase
using ProgressMeter
import Tables
import PrettyTables
import Random
using ScientficTypes

# convenience packages
using DocStringExtensions: SIGNATURES, TYPEDEF

# to be extended:
import Base.==
import StatsBase.fit!

# from Standard Library:
using Statistics
using LinearAlgebra
using Random
import Distributed: @distributed, nworkers, pmap
using RecipesBase # for plotting

# submodules of this module:
include("registry/src/Registry.jl")
import .Registry

const srcdir = dirname(@__FILE__) # the directory containing this file:
const CategoricalElement = Union{CategoricalString,CategoricalValue}

include("utilities.jl")     # general purpose utilities
include("measures.jl")       # loss functions
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


## GET THE EXTERNAL MODEL METADATA AND CODE FOR OPTIONAL DEPENDENCIES

function __init__()
    @info "Loading model metadata"
    global metadata_file = joinpath(srcdir, "registry", "Metadata.toml")
    global METADATA = TOML.parsefile(metadata_file)
    @require(LossFunctions="30fc2ffe-d236-52d8-8643-a9d8f7c094a7",
             include("loss_functions_interface.jl"))
end


## SIMPLE PLOTTING RECIPE

include("plotrecipes.jl")

end # module
