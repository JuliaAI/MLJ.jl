"""
   MLJ

[`MLJ`](https://juliaai.github.io/MLJ.jl//dev/) is a Machine Learning toolbox for
Julia. It collects together functionality from separate components listed below, which can
be loaded individually.

Actual model code (e.g., code for instantiating a `DecisionTreeClassifier`) must generally
be loaded explicitly from the model-providing package, using `@load`, for example. However
core model wrappers and some common transformers are immediately available; do
`localmodels(wrappers=true)` at startup to list these.

# Components

- MLJBase.jl: The `machine` interface, tools to `partition` and `unpack` datasets,
  `evaluate`/`evaluate!` for model performance, `|>` pipeline syntax,
  `TransformedTargetModel` wrapper, general model composition syntax (learning networks),
  synthetic data generators, `scitype` and `schema` methods (from ScientificTypes.jl) for
  checking how MLJ interprets your data. Generally required for any MLJ workflow.

- StatisticalMeasures.jl: MLJ-compatible measures (metrics) for machine learning,
  confusion matrices, ROC curves.

- MLJModels.jl: Hosts the MLJ Model Registry and the associated model search tools,
  `models(...)` and `matching(...)`. Provides the `@load` macro for loading model
  code. Provides dummy models, such as `ConstantClassifier`, and the
  `BinaryThresholdPredictor` wrapper.

- MLJTransforms: Makes immediately available common data preprocessing operations, such as
  categorical feature encoding.

- MLJTuning.jl: Hyperparameter optimization via `TunedModel` wrapper

- MLJIteration.jl: `IteratedModel` Wrapper for controlling iterative models

- MLJEnsembles.jl: Homogeneous model ensembling, via the `EnsembleModel` wrapper

- MLJBalancing.jl: Incorporation of oversampling/undersampling methods in pipelines, via
  the `BalancedModel` wrapper

- FeatureSelection.jl: Transformers for feature selection, and the supervised model wrapper
  `RecursiveFeatureSelection`.

- MLJFlow.jl: Integration with MLflow workflow tracking

- OpenML.jl: Tool for grabbing datasets from OpenML.org



"""
module MLJ


## METHOD IMPORT

# from the Standard Library:
import Distributed: @distributed, nworkers, pmap
import Pkg
import Pkg.TOML

using Reexport

# from the MLJ universe:
using MLJBase
import MLJBase.save
using MLJEnsembles
using MLJTuning
using MLJModels
@reexport using FeatureSelection
using OpenML
@reexport using MLJFlow
@reexport using StatisticalMeasures
import MLJBalancing
@reexport using MLJBalancing: BalancedModel
using MLJIteration
import MLJIteration.IterationControl
@reexport using MLJTransforms

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
    ResamplingStrategy, Holdout, CV, TimeSeriesCV, InSample,
    StratifiedCV, evaluate!, Resampler, iterator, PerformanceEvaluation,
    CompactPerformanceEvaluation,
    default_resource, default_logger, pretty,
    make_blobs, make_moons, make_circles, make_regression,
    fit_only!, return!, int, decoder,
    default_scitype_check_level,
    serializable, restore!

# abstract model types from MLJBase:
for T in MLJBase.EXTENDED_ABSTRACT_MODEL_TYPES
    @eval(export $T)
end

# re-export from MLJEnsembles:
export EnsembleModel

# re-export from MLJTuning:
export Grid, RandomSearch, Explicit, TunedModel, LatinHypercube,
    learning_curve!, learning_curve

# re-export from MLJModels:
export models, localmodels, @load, @iload, load, info, doc,
    ConstantRegressor, ConstantClassifier,     # builtins/Constant.jl
    UnivariateStandardizer,
    Standardizer, UnivariateBoxCoxTransformer,
    OneHotEncoder, ContinuousEncoder, UnivariateDiscretizer,
    FillImputer, matching, BinaryThresholdPredictor,
    UnivariateTimeTypeToContinuous, InteractionTransformer  # builtins/Transformers.jl

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
