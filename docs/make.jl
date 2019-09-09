if Base.HOME_PROJECT[] !== nothing
    Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[])
end
using Pkg
using Documenter
using MLJ
using MLJBase
using MLJModels.Transformers
using MLJModels.Constant
using MLJModels.KNN
using MLJModels
using ScientificTypes

#prettyurls to be changed
makedocs(
    sitename = "MLJ",
    format = Documenter.HTML(),
    modules = [MLJ, MLJBase, MLJModels, MLJModels.Transformers, MLJModels.Constant,
               MLJModels.KNN, ScientificTypes],
    pages = Any["Getting Started"=>"index.md",
                "Evaluating model performance"=>"evaluating_model_performance.md",
                "Performance Measures"=> "performance_measures.md",
                "Tuning models"=>"tuning_models.md",
                "Built-in Transformers" => "built_in_transformers.md",
                "Composing Models" => "composing_models.md",
                "Homogeneous Ensembles" => "homogeneous_ensembles.md",
                "Simple User Defined Models" => "simple_user_defined_models.md",
                "Adding Models for General Use" => "adding_models_for_general_use.md",
#                "Working with Tasks" => "working_with_tasks.md",
                "Benchmarking" => "benchmarking.md",
                "Internals"=>"internals.md",
                "Glossary"=>"glossary.md",
                "API"=>"api.md",
                "MLJ Cheatsheet" => "mlj_cheatsheet.md",
                "MLJ News"=>"NEWS.md",
                "FAQ" => "frequently_asked_questions.md",
                 "Julia BlogPost"=>"julia_blogpost.md"]
)

deploydocs(
    repo = "github.com/alan-turing-institute/MLJ.jl.git"
)

#    modules = [MLJ]
# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more MLJBase.information.
