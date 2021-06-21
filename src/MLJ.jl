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
using MLJOpenML
using MLJSerialization
import MLJSerialization.save # but not re-exported
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
import MLJScientificTypes


## METHOD EXPORT

export MLJ_VERSION


## METHOD RE-EXPORT

# re-export from Random, Statistics, Distributions, CategoricalArrays:
export pdf, logpdf, mode, median, mean, shuffle!, categorical, shuffle,
    levels, levels!, std, support, sampler

# re-exports from (MLJ)ScientificTypes via MLJBase
export Scientific, Found, Unknown, Known, Finite, Infinite,
    OrderedFactor, Multiclass, Count, Continuous, Textual,
    Binary, ColorImage, GrayImage, Image, Table
export scitype, scitype_union, elscitype, nonmissing, trait
export coerce, coerce!, autotype, schema, info

# re-export from MLJBase:
export nrows, color_off, color_on,
    selectrows, selectcols, restrict, corestrict, complement,
    Deterministic, Probabilistic, JointProbabilistic, Unsupervised, Supervised, Static,
    DeterministicNetwork, ProbabilisticNetwork, UnsupervisedNetwork,
    ProbabilisticComposite, JointProbabilisticComposite, DeterministicComposite,
    IntervalComposite, UnsupervisedComposite, StaticComposite,
    ProbabilisticSurrogate, JointProbabilisticSurrogate, DeterministicSurrogate,
    IntervalSurrogate, UnsupervisedSurrogate, StaticSurrogate,
    Surrogate, Composite,
    target_scitype, input_scitype, output_scitype, load_path, training_losses,
    predict, predict_mean, predict_median, predict_mode, predict_joint,
    transform, inverse_transform, evaluate, fitted_params, params,
    @constant, @more, HANDLE_GIVEN_ID, UnivariateFinite,
    classes, table, report, rebind!,
    partition, unpack,
    default_measure, measures,
    @load_boston, @load_ames, @load_iris, @load_reduced_ames, @load_crabs,
    load_boston, load_ames, load_iris, load_reduced_ames, load_crabs,
    Machine, machine, AbstractNode, @node,
    source, node, fit!, freeze!, thaw!, Node, sources, origins,
    machines, sources, anonymize!, @from_network, fitresults,
    @pipeline, Stack,
    ResamplingStrategy, Holdout, CV,
    StratifiedCV, evaluate!, Resampler, iterator,
    default_resource, pretty,
    make_blobs, make_moons, make_circles, make_regression,
    fit_only!, return!, int, decoder

# MLJBase/measure/measures.jl:
export orientation, reports_each_observation,
    is_feature_dependent, aggregation,
    aggregate, default_measure, value,
    supports_class_weights, prediction_type, human_name

# MLJBase/measures/continuous.jl:
export mav, mae, mape, rms, rmsl, rmslp1, rmsp, l1, l2, log_cosh,
    MAV, MAE, MeanAbsoluteError, mean_absolute_error, mean_absolute_value,
    LPLoss, RootMeanSquaredProportionalError, RMSP,
    RMS, rmse, RootMeanSquaredError, root_mean_squared_error,
    RootMeanSquaredLogError, RMSL, root_mean_squared_log_error, rmsl, rmsle,
    RootMeanSquaredLogProportionalError, rmsl1, RMSLP,
    MAPE, MeanAbsoluteProportionalError, log_cosh_loss, LogCosh, LogCoshLoss

# MLJBase/measures/confusion_matrix.jl:
export confusion_matrix, confmat, ConfusionMatrix

# MLJBase/measures/finite.jl:
export cross_entropy, BrierScore, brier_score,
    BrierLoss, brier_loss,
    LogLoss, log_loss,
    misclassification_rate, mcr, accuracy,
    balanced_accuracy, bacc, bac, BalancedAccuracy,
    matthews_correlation, mcc, MCC, AUC, AreaUnderCurve,
    MisclassificationRate, Accuracy, MCR, BACC, BAC,
    MatthewsCorrelation

# MLJBase/measures/finite.jl -- Multiclass{2} (order independent):
export auc, area_under_curve, roc_curve, roc

# MLJBase/measures/finite.jl -- OrderedFactor{2} (order dependent):
export TruePositive, TrueNegative, FalsePositive, FalseNegative,
    TruePositiveRate, TrueNegativeRate, FalsePositiveRate,
    FalseNegativeRate, FalseDiscoveryRate, Precision, NPV, FScore,
    NegativePredictiveValue,
    # standard synonyms
    TPR, TNR, FPR, FNR, FDR, PPV,
    Recall, Specificity, BACC,
    # instances and their synonyms
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
    specificity, selectivity, f1score, fallout

# MLJBase/measures/finite.jl -- Finite{N} - multiclass generalizations of
# above OrderedFactor{2} measures (but order independent):
export MulticlassTruePositive, MulticlassTrueNegative, MulticlassFalsePositive,
      MulticlassFalseNegative, MulticlassTruePositiveRate,
      MulticlassTrueNegativeRate, MulticlassFalsePositiveRate,
      MulticlassFalseNegativeRate, MulticlassFalseDiscoveryRate,
      MulticlassPrecision, MulticlassNegativePredictiveValue, MulticlassFScore,
      # standard synonyms
      MTPR, MTNR, MFPR, MFNR, MFDR, MPPV,
      MulticlassRecall, MulticlassSpecificity,
      # instances and their synonyms
      multiclass_truepositive, multiclass_truenegative,
      multiclass_falsepositive,
      multiclass_falsenegative, multiclass_true_positive,
      multiclass_true_negative, multiclass_false_positive,
      multiclass_false_negative, multiclass_truepositive_rate,
      multiclass_truenegative_rate, multiclass_falsepositive_rate,
      multiclass_true_positive_rate, multiclass_true_negative_rate,
      multiclass_false_positive_rate, multiclass_falsenegative_rate,
      multiclass_negativepredictive_value, multiclass_false_negative_rate,
      multiclass_negative_predictive_value, multiclass_positivepredictive_value,
      multiclass_positive_predictive_value, multiclass_tpr, multiclass_tnr,
      multiclass_fpr, multiclass_fnr, multiclass_falsediscovery_rate,
      multiclass_false_discovery_rate, multiclass_fdr, multiclass_npv,
      multiclass_ppv, multiclass_recall, multiclass_sensitivity,
      multiclass_hit_rate, multiclass_miss_rate, multiclass_specificity,
      multiclass_selectivity, macro_f1score, micro_f1score,
      multiclass_f1score, multiclass_fallout, multiclass_precision,
      # averaging modes
      no_avg, macro_avg, micro_avg

# MLJBase/measures/loss_functions_interface.jl
export dwd_margin_loss, exp_loss, l1_hinge_loss, l2_hinge_loss, l2_margin_loss,
    logit_margin_loss, modified_huber_loss, perceptron_loss, sigmoid_loss,
    smoothed_l1_hinge_loss, zero_one_loss, huber_loss, l1_epsilon_ins_loss,
    l2_epsilon_ins_loss, lp_dist_loss, logit_dist_loss, periodic_loss,
    quantile_loss

# MLJBase/measures/loss_functions_interface.jl
export DWDMarginLoss, ExpLoss, L1HingeLoss, L2HingeLoss, L2MarginLoss,
    LogitMarginLoss, ModifiedHuberLoss, PerceptronLoss, SigmoidLoss,
    SmoothedL1HingeLoss, ZeroOneLoss, HuberLoss, L1EpsilonInsLoss,
    L2EpsilonInsLoss, LPDistLoss, LogitDistLoss, PeriodicLoss,
    QuantileLoss

# re-export from MLJEnsembles:
export EnsembleModel

# re-export from MLJTuning:
export Grid, RandomSearch, Explicit, TunedModel, LatinHypercube,
    learning_curve!, learning_curve

# re-export from MLJModels:
export models, localmodels, @load, @iload, load, info,
    ConstantRegressor, ConstantClassifier,     # builtins/Constant.jl
    FeatureSelector, UnivariateStandardizer,   # builtins/Transformers.jl
    Standardizer, UnivariateBoxCoxTransformer,
    OneHotEncoder, ContinuousEncoder, UnivariateDiscretizer,
    FillImputer, matching, BinaryThresholdPredictor

# re-export from MLJIteration:
export IteratedModel
for control in MLJIteration.CONTROLS
    eval(:(export $control))
end
export IterationControl

# re-export from MLJOpenML
const OpenML = MLJOpenML # for backwards compatibility
export OpenML

# re-export from MLJSerialization:
export Save # control, not method

# re-export from ComputaionalResources:
export CPU1, CPUProcesses, CPUThreads



## CONSTANTS

const srcdir = dirname(@__FILE__)


## INCLUDE FILES

include("version.jl")       # defines MLJ_VERSION constant
include("scitypes.jl")      # extensions to ScientificTypes.scitype

end # module
