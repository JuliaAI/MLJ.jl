if Base.HOME_PROJECT[] !== nothing
    Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[])
end

using Pkg
using Documenter
using MLJ
using MLJBase
import MLJ.MLJBase.MLJModelInterface
import MLJ.MLJIteration
import MLJ.MLJIteration.IterationControl
import MLJ.MLJIteration.IterationControl.EarlyStopping
import MLJ.MLJTuning
import MLJ.MLJModels
import MLJ.MLJEnsembles
import MLJ.ScientificTypes
import MLJ.MLJBalancing
import MLJ.FeatureSelection
import ScientificTypesBase
import Distributions
using CategoricalArrays
import CategoricalDistributions
import StatisticalMeasures
import StatisticalMeasuresBase

const MMI = MLJModelInterface

include("model_docstring_tools.jl")

# checking every model has a descriptor, for determining categories under which it appears
# in the Model Browser section of manual:
@info "Checking ModelDescriptors.toml to see all models have descriptors assigned. "
problems = models_missing_descriptors()
isempty(problems) || error(
    "The following keys are missing from /docs/ModelDescriptors.toml: "*
        "$problems. ")

# compose the individual model docstring pages:
@info "Getting individual model docstrings from the registry and generating "*
    "pages for them, written at /docs/src/models/ ."
for model in models(wrappers=true)
    write_page(model)
end

# compose the model browser page:
@info "Composing the Model Browser page, /docs/src/model_browser.md"
write_page()

# using Literate
# Literate.markdown("common_mlj_workflows.jl", ".",
#                   codefence = "```@example workflows" => "```")

pages = [
    "Home" => "index.md",
    "Model Browser" => "model_browser.md",
    "About MLJ" => "about_mlj.md",
    "Learning MLJ" => "learning_mlj.md",
    "Basics" => [
        "Getting Started" => "getting_started.md",
        "Common MLJ Workflows" => "common_mlj_workflows.md",
        "Machines" => "machines.md",
        "MLJ Cheatsheet" => "mlj_cheatsheet.md",
    ],
    "Data" => [
        "Working with Categorical Data" => "working_with_categorical_data.md",
        "Preparing Data" => "preparing_data.md",
        "Generating Synthetic Data" => "generating_synthetic_data.md",
        "OpenML Integration" => "openml_integration.md",
    ],
    "Models" => [
        "Model Search" => "model_search.md",
        "Loading Model Code" => "loading_model_code.md",
        "Transformers and Other Unsupervised models" => "transformers.md",
        "List of Supported Models" => "list_of_supported_models.md",
    ],
    "Meta-algorithms" => [
        "Evaluating Model Performance" => "evaluating_model_performance.md",
        "Tuning Models" => "tuning_models.md",
        "Learning Curves" => "learning_curves.md",
        "Controlling Iterative Models" => "controlling_iterative_models.md",
        "Correcting Class Imbalance" => "correcting_class_imbalance.md",
        "Thresholding Probabilistic Predictors" =>
            "thresholding_probabilistic_predictors.md",
        "Target Transformations" => "target_transformations.md",
        "Homogeneous Ensembles" => "homogeneous_ensembles.md",
    ],
    "Model Composition" => [
        "Composing Models" => "composing_models.md",
        "Linear Pipelines" => "linear_pipelines.md",
        "Model Stacking" => "model_stacking.md",
        "Learning Networks" => "learning_networks.md",
    ],
    "Third Party Tools" => [
        "Logging Workflows using MLflow" => "logging_workflows.md",
        "Third Party Packages" => "third_party_packages.md",
    ],
    "Customization and Extension" => [
        "Simple User Defined Models" => "simple_user_defined_models.md",
        "Quick-Start Guide to Adding Models" =>
            "quick_start_guide_to_adding_models.md",
        "Adding Models for General Use" => "adding_models_for_general_use.md",
        "Modifying Behavior" => "modifying_behavior.md",
        "Internals" => "internals.md",
    ],
    "Miscellaneous" => [
        "Performance Measures" => "performance_measures.md",
        "Weights" => "weights.md",
        "Acceleration and Parallelism" => "acceleration_and_parallelism.md",
        "Glossary" => "glossary.md",
        "FAQ" => "frequently_asked_questions.md",
    ],
    "Index of Methods" => "api.md",
]

const ASSET_URL1 =
    "https://fonts.googleapis.com/css2?"*
    "family=Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;1,100;1,300;1,400;1,700;1,900"*
    "&family=Montserrat:ital,wght@0,100..900;1,100..900&display=swap"

const ASSET_URL2 =
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"

makedocs(
    doctest  = true,
    sitename = "MLJ",
    format   = Documenter.HTML(
        collapselevel = 1,
        assets = [
            "assets/favicon.ico",
            asset(ASSET_URL1, class = :css),
            asset(ASSET_URL2, class = :css),
        ],
        repolink="https://github.com/JuliaAI/MLJ.jl"
    ),
    modules  = [
        MLJ,
        MLJBase,
        MLJTuning,
        MLJModels,
        MLJEnsembles,
        MLJBalancing,
        MLJIteration,
        ScientificTypes,
        MLJModelInterface,
        ScientificTypesBase,
        StatisticalMeasures,
        EarlyStopping,
        IterationControl,
        CategoricalDistributions,
        StatisticalMeasures,
        FeatureSelection,
    ],
    pages    = pages,
    warnonly = [:cross_references, :missing_docs],
)

@info "`makedocs` has finished running. "

deploydocs(
    repo = "github.com/JuliaAI/MLJ.jl",
    devbranch="master",
    push_preview=false,
)

@info "`deploydocs` has finished running. "
