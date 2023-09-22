if Base.HOME_PROJECT[] !== nothing
    Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[])
end

using Pkg
using Documenter
using MLJ
import MLJIteration
import IterationControl
import EarlyStopping
import MLJBase
import MLJTuning
import MLJModels
import MLJEnsembles
import ScientificTypes
import MLJModelInterface
import ScientificTypesBase
import Distributions
using CategoricalArrays
using LossFunctions
import CategoricalDistributions

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
for model in models()
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
    "Getting Started" => "getting_started.md",
    "Common MLJ Workflows" => "common_mlj_workflows.md",
    "Working with Categorical Data" => "working_with_categorical_data.md",
    "Model Search" => "model_search.md",
    "Loading Model Code" => "loading_model_code.md",
    "Machines" => "machines.md",
    "Evaluating Model Performance" => "evaluating_model_performance.md",
    "Performance Measures" => "performance_measures.md",
    "Weights" => "weights.md",
    "Tuning Models" => "tuning_models.md",
    "Learning Curves" => "learning_curves.md",
    "Preparing Data" => "preparing_data.md",
    "Transformers and Other Unsupervised models" => "transformers.md",
    "More on Probabilistic Predictors" => "more_on_probabilistic_predictors.md",
    "Composing Models" => "composing_models.md",
    "Linear Pipelines" => "linear_pipelines.md",
    "Target Transformations" => "target_transformations.md",
    "Homogeneous Ensembles" => "homogeneous_ensembles.md",
    "Model Stacking" => "model_stacking.md",
    "Learning Networks" => "learning_networks.md",
    "Controlling Iterative Models" => "controlling_iterative_models.md",
    "Generating Synthetic Data" => "generating_synthetic_data.md",
    "Logging Workflows" => "logging_workflows.md",
    "OpenML Integration" => "openml_integration.md",
    "Acceleration and Parallelism" => "acceleration_and_parallelism.md",
    "Simple User Defined Models" => "simple_user_defined_models.md",
    "Quick-Start Guide to Adding Models" =>
               "quick_start_guide_to_adding_models.md",
    "Adding Models for General Use" => "adding_models_for_general_use.md",
    "Modifying Behavior" => "modifying_behavior.md",
    "Internals" => "internals.md",
    "List of Supported Models" => "list_of_supported_models.md",
    "Third Party Packages" => "third_party_packages.md",
    "Glossary" => "glossary.md",
    "MLJ Cheatsheet" => "mlj_cheatsheet.md",
    "FAQ" => "frequently_asked_questions.md",
    "Index of Methods" => "api.md",
    ]

for (k, v) in pages
    println("$k\t=>$v")
end

makedocs(
    doctest  = true,
    sitename = "MLJ",
    format   = Documenter.HTML(),
    modules  = [MLJ,
                MLJBase,
                MLJTuning,
                MLJModels,
                MLJEnsembles,
                ScientificTypes,
                MLJModelInterface,
                ScientificTypesBase,
                MLJIteration,
                EarlyStopping,
                IterationControl,
                CategoricalDistributions],
    pages    = pages,
    strict   = Documenter.except(:cross_references, :missing_docs),
)

@info "`makedocs` has finished running. "

deploydocs(
    repo = "github.com/alan-turing-institute/MLJ.jl",
    devbranch="master",
    push_preview=false,
)

@info "`deploydocs` has finished running. "
