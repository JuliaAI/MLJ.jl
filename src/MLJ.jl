module MLJ

# defined in include files:
export @curve, @pcurve, pretty,                       # utilities.jl
        mav, mae, rms, rmsl, rmslp1, rmsp, l1, l2,    # measures.jl
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
        models, localmodels, @load, model,            # loading.jl
        load,                          # loading.jl
        KNNRegressor,                                 # builtins/KNN.jl
        @from_network, machines, sources, anonymize!, # composites.jl
        rebind!, fitresults                           # composites.jl

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
export nrows, nfeatures, traits,
    selectrows, selectcols,
    SupervisedTask, UnsupervisedTask, MLJTask,
    Deterministic, Probabilistic, Unsupervised, Supervised,
    DeterministicNetwork, ProbabilisticNetwork,
    GrayImage, ColorImage, Image,
    Found, Continuous, Finite, Infinite,
    OrderedFactor, Unknown,
    Count, Multiclass, Binary, Scientific,
    scitype, scitype_union, schema, scitypes,
    target_scitype, input_scitype, output_scitype,
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
using OrderedCollections
using  CategoricalArrays
import Distributions: pdf, mode
import Distributions
import StatsBase
using ProgressMeter
import Tables
import PrettyTables
import Random
using ScientificTypes
import ScientificTypes

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

## LOAD BUILT-IN MODELS

include("builtins/Transformers.jl")
include("builtins/Constant.jl")
include("builtins/KNN.jl")
include("builtins/ridge.jl") # defines a model for testing only


## LOAD CORE CODE

include("utilities.jl")     # general purpose utilities
include("measures.jl")      # API for loss functions & defs of built-ins
include("machines.jl")    
include("networks.jl")      # for building learning networks
include("composites.jl")    # composite models & exporting learning networks
include("operations.jl")    # syntactic sugar for operations (predict, etc)
include("resampling.jl")    # resampling strategies and model evaluation
include("parameters.jl")    # hyperparameter ranges and grid generation
include("tuning.jl")
include("ensembles.jl")     # homogeneous ensembles
include("tasks.jl")         # enhancements to MLJBase task interface 
include("metadata.jl")      # tools to initialize metadata resources
include("model_search.jl")  # tools to inspect metadata and find models
include("loading.jl")       # fuctions to load model implementation code
include("scitypes.jl")      # extensions to ScientificTypes.sictype

    

## GET THE EXTERNAL MODEL METADATA AND CODE FOR OPTIONAL DEPENDENCIES

function __init__()
    @info "Loading model metadata from registry. "
    global metadata_file = joinpath(srcdir, "registry", "Metadata.toml")
    global INFO_GIVEN_HANDLE = info_given_handle(metadata_file)
    global AMBIGUOUS_NAMES = ambiguous_names(INFO_GIVEN_HANDLE)
    global PKGS_GIVEN_NAME = pkgs_given_name(INFO_GIVEN_HANDLE)
    global NAMES = model_names(INFO_GIVEN_HANDLE)
    @require(LossFunctions="30fc2ffe-d236-52d8-8643-a9d8f7c094a7",
             include("loss_functions_interface.jl"))
end


## SIMPLE PLOTTING RECIPE

include("plotrecipes.jl")

end # module
