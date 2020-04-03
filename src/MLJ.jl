module MLJ


## METHOD EXPORT

export MLJ_VERSION

# utilities.jl:
export @curve, @pcurve

# ensembles.jl:
export EnsembleModel

# model_matching.jl:
export matching


## METHOD RE-EXPORT

# re-export from Random, Statistics, Distributions, CategoricalArrays:
export pdf, mode, median, mean, shuffle!, categorical, shuffle,
    levels, levels!, std, support, sampler

# re-exports from (MLJ)ScientificTypes via MLJBase
export Scientific, Found, Unknown, Known, Finite, Infinite,
       OrderedFactor, Multiclass, Count, Continuous, Textual,
       Binary, ColorImage, GrayImage, Image, Table
export scitype, scitype_union, elscitype, nonmissing, trait
export coerce, coerce!, autotype, schema, info

# re-export from MLJBase:
export nrows, nfeatures, color_off, color_on,
    selectrows, selectcols, restrict, corestrict, complement,
    SupervisedTask, UnsupervisedTask, MLJTask,
    Deterministic, Probabilistic, Unsupervised, Supervised, Static,
    DeterministicNetwork, ProbabilisticNetwork,
    target_scitype, input_scitype, output_scitype,
    predict, predict_mean, predict_median, predict_mode,
    transform, inverse_transform, evaluate, fitted_params, params,
    @constant, @more, HANDLE_GIVEN_ID, UnivariateFinite,
    classes, table, report, rebind!,
    partition, unpack,
    default_measure, measures,
    @load_boston, @load_ames, @load_iris, @load_reduced_ames, @load_crabs,
    load_boston, load_ames, load_iris, load_reduced_ames, load_crabs,
    Machine, NodalMachine, machine, AbstractNode,
    source, node, fit!, freeze!, thaw!, Node, sources, origins,
    machines, sources, anonymize!, @from_network, fitresults,
    @pipeline,
    ResamplingStrategy, Holdout, CV,
    StratifiedCV, evaluate!, Resampler, iterator,
    default_resource, pretty,
    OpenML

export measures,
    orientation, reports_each_observation,
    is_feature_dependent, aggregation,
    aggregate,
    default_measure, value,
    mav, mae, rms, rmsl, rmslp1, rmsp, l1, l2,
    confusion_matrix, confmat,
    cross_entropy, BrierScore,
    misclassification_rate, mcr, accuracy,
    balanced_accuracy, bacc, bac,
    matthews_correlation, mcc,
    auc, area_under_curve, roc_curve, roc,
    TruePositive, TrueNegative, FalsePositive, FalseNegative,
    TruePositiveRate, TrueNegativeRate, FalsePositiveRate, FalseNegativeRate,
    FalseDiscoveryRate, Precision, NPV, FScore,
    TPR, TNR, FPR, FNR,
    FDR, PPV,
    Recall, Specificity, BACC,
    truepositive, truenegative, falsepositive, falsenegative,
    true_positive, true_negative, false_positive, false_negative,
    truepositive_rate, truenegative_rate, falsepositive_rate,
    true_positive_rate, true_negative_rate, false_positive_rate,
    falsenegative_rate, negativepredictive_value,
    false_negative_rate, negative_predictive_value,
    positivepredictive_value, positive_predictive_value,
    tpr, tnr, fpr, fnr,
    falsediscovery_rate, false_discovery_rate, fdr, npv, ppv,
    recall, sensitivity, hit_rate, miss_rate,
    specificity, selectivity, f1score, f1, fallout

# re-export from MLJTuning:
export Grid, RandomSearch, Explicit, TunedModel,
    learning_curve!, learning_curve

# re-export from MLJModels:
export models, localmodels, @load, load, info,
    ConstantRegressor, ConstantClassifier,     # builtins/Constant.jl
    StaticTransformer, FeatureSelector,        # builtins/Transformers.jl
    UnivariateStandardizer, Standardizer,
    UnivariateBoxCoxTransformer,
    OneHotEncoder, UnivariateDiscretizer,
    FillImputer

# re-export from ComputaionalResources:
export CPU1, CPUProcesses, CPUThreads


## METHOD IMPORT

# from the Standard Library:
import Distributed: @distributed, nworkers, pmap
import Pkg
import Pkg.TOML

# from the MLJ universe:
using MLJBase
import MLJBase.save
using MLJTuning
using MLJModels

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
import MLJScientificTypes

## CONSTANTS

const srcdir = dirname(@__FILE__)
const CategoricalElement = Union{CategoricalString,CategoricalValue}

# FIXME replace with either Pkg.installed()["MLJ"] or uuid =
# Pkg.project().dependencies["MLJ"] version =
# Pkg.dependencies()[uuid].version --- this is currently messy because
# it's been enacted then reverted see
# https://github.com/JuliaLang/julia/pull/33410 and
# https://github.com/JuliaLang/Pkg.jl/pull/1086/commits/996c6b9b69ef0c058e0105427983622b7cc8cb1d
toml = Pkg.TOML.parsefile(joinpath(dirname(dirname(pathof(MLJ))),
                                  "Project.toml"))
const MLJ_VERSION = toml["version"]


## INCLUDE FILES

include("ensembles.jl")     # homogeneous ensembles
include("model_matching.jl")# inferring model search criterion from data
include("scitypes.jl")      # extensions to ScientificTypes.scitype

end # module
