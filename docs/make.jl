if Base.HOME_PROJECT[] !== nothing
    Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[])
end

using Pkg
using Documenter
using MLJ
import MLJBase
import MLJTuning
import MLJModels
import MLJScientificTypes
import MLJModelInterface
import ScientificTypes
import Distributions
using CategoricalArrays # avoid types like CategoricalArrays.Categorica
using LossFunctions

const MMI = MLJModelInterface

# using Literate
# Literate.markdown("common_mlj_workflows.jl", ".",
#                   codefence = "```@example workflows" => "```")

pages = [
    "Introduction" => "index.md",
    "Getting Started" => "getting_started.md",
    "Common MLJ Workflows" => "common_mlj_workflows.md",
    "Working with Categorical Data" => "working_with_categorical_data.md",
    "Model Search" => "model_search.md",
    "Machines" => "machines.md",
    "Evaluating Model Performance" => "evaluating_model_performance.md",
    "Performance Measures" => "performance_measures.md",
    "Tuning Models" => "tuning_models.md",
    "Learning Curves" => "learning_curves.md",
    "Transformers and other unsupervised models" => "transformers.md",
    "Composing Models" => "composing_models.md",
    "Homogeneous Ensembles" => "homogeneous_ensembles.md",
    "Generating Synthetic Data" => "generating_synthetic_data.md",
    "OpenML Integration" => "openml_integration.md",
    "Simple User Defined Models" => "simple_user_defined_models.md",
    "Quick-Start Guide to Adding Models" => "quick_start_guide_to_adding_models.md",
    "Adding Models for General Use" => "adding_models_for_general_use.md",
    "Benchmarking" => "benchmarking.md",
    "Internals" => "internals.md",
    "List of Supported Models" => "list_of_supported_models.md",
    "Glossary" => "glossary.md",
    "MLJ Cheatsheet" => "mlj_cheatsheet.md",
    "FAQ" => "frequently_asked_questions.md",
    "Julia BlogPost" => "julia_blogpost.md",
    "Acceleration and Parallelism" => "acceleration_and_parallelism.md",
    "Index of Methods" => "api.md",
    ]

for (k, v) in pages
    println("$k\t=>$v")
end

makedocs(
    sitename = "MLJ",
    format   = Documenter.HTML(),
    modules  = [MLJ, MLJBase, MLJTuning, MLJModels, MLJScientificTypes, MLJModelInterface, ScientificTypes],
    pages    = pages)

# By default Documenter does not deploy docs just for PR
# this causes issues with how we're doing things and ends
# up choking the deployment of the docs, so  here we
# force the environment to ignore this so that Documenter
# does indeed deploy the docs
ENV["TRAVIS_PULL_REQUEST"] = "false"

deploydocs(
    repo = "github.com/alan-turing-institute/MLJ.jl.git"
)
