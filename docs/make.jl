if Base.HOME_PROJECT[] !== nothing
    Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[])
end
using Pkg
#Pkg.add("Documenter")
#Pkg.clone("https://github.com/wildart/TOML.jl")
#Pkg.clone("https://github.com/alan-turing-institute/MLJBase.jl")
#Pkg.clone("https://github.com/alan-turing-institute/MLJModels.jl")
#Pkg.clone("https://github.com/alan-turing-institute/MLJ.jl")
using Documenter
using MLJ
using MLJBase
using MLJModels

#prettyurls to be changed
makedocs(
    sitename = "MLJ",
    format = Documenter.HTML(),
    modules = [MLJ, MLJBase, MLJModels, MLJ.Transformers],
    pages = Any["Getting Started"=>"index.md",
                "Scientific Data Types"=>"scientific_data_types.md",
                "Learning Networks" => "learning_networks.md",
                "Adding New Models"=> ["Adding New Models"=>"adding_new_models.md",
                                       "The Simplified Model API"=>"the_simplified_model_api.md"],
                "Internals"=>"internals.md",
                "Glossary"=>"glossary.md",
                "API"=>"api.md",
                "FAQ" => "frequently_asked_questions.md",
                "MLJ News"=>"NEWS.md"]
)

deploydocs(
    repo = "github.com/alan-turing-institute/MLJ.jl.git"
)

#    modules = [MLJ]
# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
