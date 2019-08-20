if Base.HOME_PROJECT[] !== nothing
    Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[])
end
using Pkg
#Pkg.add("Documenter")
#Pkg.clone("https://github.com/alan-turing-institute/MLJBase.jl")
#Pkg.clone("https://github.com/alan-turing-institute/MLJModels.jl")
#Pkg.clone("https://github.com/alan-turing-institute/MLJ.jl")
using Documenter
using MLJ
using MLJBase
using MLJ.Transformers
using MLJ.Constant
using MLJModels
using ScientificTypes

#prettyurls to be changed
makedocs(
    sitename = "MLJ",
    format = Documenter.HTML(),
    modules = [MLJ, MLJBase, MLJModels, MLJ.Transformers, ScientificTypes],
    pages = Any["Getting Started"=>"index.md",
                "Evaluating model performance"=>"evaluating_model_performance.md",
                "Measures"=> "measures.md",
                "Tuning models"=>"tuning_models.md",
                "Built-in Transformers" => "built_in_transformers.md",
                "Learning Networks" => "learning_networks.md",
                "Simple User Defined Models" => "simple_user_defined_models.md",
                "Adding Models for General Use" => "adding_models_for_general_use.md",
                "Working with Tasks" => "working_with_tasks.md",
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
# for more information.
