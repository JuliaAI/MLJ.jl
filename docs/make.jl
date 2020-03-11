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
import Distributions
using CategoricalArrays # avoid types like CategoricalArrays.Categorica
using LossFunctions

const MMI = MLJModelInterface

# using Literate
# Literate.markdown("common_mlj_workflows.jl", ".",
#                   codefence = "```@example workflows" => "```")

pages = [
    "Getting Started" => "index.md",
    "Common MLJ Workflows" => "common_mlj_workflows.md",
    "Model Search" => "model_search.md",
    "Machines" => "machines.md",
    "Evaluating Model Performance" => "evaluating_model_performance.md",
    "Performance Measures" => "performance_measures.md",
    "Tuning Models" => "tuning_models.md",
    "Learning Curves" => "learning_curves.md",
    "Built-in Transformers" => "built_in_transformers.md",
    "Composing Models" => "composing_models.md",
    "Homogeneous Ensembles" => "homogeneous_ensembles.md",
    "OpenML Integration" => "openml_integration.md",
    "Simple User Defined Models" => "simple_user_defined_models.md",
    "Quick-Start Guide to Adding Models" => "quick_start_guide_to_adding_models.md",
    "Adding Models for General Use" => "adding_models_for_general_use.md",
    "Benchmarking" => "benchmarking.md",
    "Internals" => "internals.md",
    "Glossary" => "glossary.md",
    # "API" => "api.md", # NOTE: commented as currently empty
    "MLJ Cheatsheet" => "mlj_cheatsheet.md",
    "MLJ News" => "NEWS.md",
    "FAQ" => "frequently_asked_questions.md",
    "Julia BlogPost" => "julia_blogpost.md",
    "Acceleration and Parallelism" => "acceleration_and_parallelism.md"
    ]

for (k, v) in pages
    println("$k\t=>$v")
end

makedocs(
    sitename = "MLJ",
    format   = Documenter.HTML(),
    modules  = [MLJ, MLJBase, MLJTuning, MLJModels, MLJScientificTypes, MLJModelInterface],
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
