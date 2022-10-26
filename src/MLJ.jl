module MLJ


## METHOD IMPORT

# from the Standard Library:
import Distributed: @distributed, nworkers, pmap
import Pkg
import Pkg.TOML

# from the MLJ universe:
using MLJBase
import MLJBase.save
using MLJEnsembles
using MLJTuning
using MLJModels
using OpenML
using MLJIteration
import MLJIteration.IterationControl

using Tables, CategoricalArrays
import Distributions
import Distributions: pdf, mode
import Statistics, StatsBase, LinearAlgebra, Random
import Random: AbstractRNG, MersenneTwister
using ProgressMeter
using ComputationalResources
using ComputationalResources: CPUProcesses

# to be extended:
import MLJBase: fit, update, clean!, fit!, predict, fitted_params,
                show_as_constructed, ==
import MLJModels: models
import ScientificTypes


## CONSTANTS

const srcdir = dirname(@__FILE__)
const TRAITS_NOT_EXPORTED = [
    :supports_online, # no models support this
    :name,            # likely conflict with other uses
    :abstract_type,   # for advanced development only
    :hyperparameter_ranges # not implemented and will probably be deprecated
                           # in favour of hyperparameter priors
]


## INCLUDE FILES

include("version.jl")       # defines MLJ_VERSION constant
include("scitypes.jl")      # extensions to ScientificTypesBase.scitype


## METHOD EXPORT

export MLJ_VERSION


## METHOD RE-EXPORT

# traits for models and measures:
using MLJBase.StatisticalTraits
for trait in setdiff(StatisticalTraits.TRAITS, TRAITS_NOT_EXPORTED)
    :(export $trait) |> eval
end

# re-export from Random, Statistics, Distributions, CategoricalArrays:
export pdf, logpdf, mode, median, mean, shuffle!, categorical, shuffle,
    levels, levels!, std, support, sampler

# re-exports from (MLJ)ScientificTypesBase via MLJBase
export Scientific, Found, Unknown, Known, Finite, Infinite,
    OrderedFactor, Multiclass, Count, Continuous, Textual,
    Binary, ColorImage, GrayImage, Image, Table
export scitype, scitype_union, elscitype, nonmissing, trait
export coerce, coerce!, autotype, schema, info

# re-export from MLJBase:
import MLJBase: serializable, restore!
export nrows, color_off, color_on,
    selectrows, selectcols, restrict, corestrict, complement,
    training_losses, feature_importances,
    predict, predict_mean, predict_median, predict_mode, predict_joint,
    transform, inverse_transform, evaluate, fitted_params, params,
    @constant, @more, HANDLE_GIVEN_ID, UnivariateFinite,
    classes, table, report, rebind!,
    partition, unpack,
    @load_boston, @load_ames, @load_iris, @load_reduced_ames, @load_crabs,
    load_boston, load_ames, load_iris, load_reduced_ames, load_crabs,
    Machine, machine, AbstractNode, @node,
    source, node, fit!, freeze!, thaw!, Node, sources, origins,
    machines, sources, anonymize!, @from_network, fitresults,
    @pipeline, Stack, Pipeline, TransformedTargetModel,
    ResamplingStrategy, Holdout, CV, TimeSeriesCV,
    StratifiedCV, evaluate!, Resampler, iterator, PerformanceEvaluation,
    default_resource, pretty,
    make_blobs, make_moons, make_circles, make_regression,
    fit_only!, return!, int, decoder,
    default_scitype_check_level,
    serializable, restore!

# MLJBase/composition/abstract_types.jl:
for T in vcat(MLJBase.MLJModelInterface.ABSTRACT_MODEL_SUBTYPES,
    MLJBase.COMPOSITE_TYPES,
    MLJBase.SURROGATE_TYPES)
    @eval(export $T)
end
export Surrogate, Composite
export DeterministicNetwork, ProbabilisticNetwork, UnsupervisedNetwork


# MLJBase/measures:
# measure names:
for m in MLJBase.MEASURE_TYPES_ALIASES_AND_INSTANCES
    :(export $m) |> eval
end
export measures,
    aggregate, default_measure, value, skipinvalid,
    roc_curve, roc,
    no_avg, macro_avg, micro_avg

# re-export from MLJEnsembles:
export EnsembleModel

# re-export from MLJTuning:
export Grid, RandomSearch, Explicit, TunedModel, LatinHypercube,
    learning_curve!, learning_curve

# re-export from MLJModels:
export models, localmodels, @load, @iload, load, info, doc,
    ConstantRegressor, ConstantClassifier,     # builtins/Constant.jl
    FeatureSelector, UnivariateStandardizer,   # builtins/Transformers.jl
    Standardizer, UnivariateBoxCoxTransformer,
    OneHotEncoder, ContinuousEncoder, UnivariateDiscretizer,
    FillImputer, matching, BinaryThresholdPredictor,
    UnivariateTimeTypeToContinuous, InteractionTransformer

# re-export from MLJIteration:
export MLJIteration
export IteratedModel
for control in MLJIteration.CONTROLS
    eval(:(export $control))
end
export IterationControl

# re-export from MLJOpenML
const MLJOpenML = OpenML

export OpenML, MLJOpenML

# re-export from ComputaionalResources:
export CPU1, CPUProcesses, CPUThreads, CUDALibs


end # module
