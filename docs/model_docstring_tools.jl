# # METHODS TO WRITE A MODEL DOCSTRING TO A FILE

const PATH_TO_MODEL_DOCS = joinpath(@__DIR__, "src", "models")

"""
    remove_doc_refs(str::AbstractString)

Removes `@ref` references from `str`. For example, a substring of the form
"[`some.thing_like_this123!`](@ref)" is replaced with "`some.thing_like_this123!`".

"""
function remove_doc_refs(page)
    regex = r"\[([\?'\.\d`\!\_a-zA-Z]*)\]\(\@ref\)"
    while contains(page, regex)
        # replace the first matched regex with the captured string
        page = replace(page, regex => s"\1")
    end
    page
end

demote_headings(str) = replace(str, "# "=>"## ")
remove_example_fencing(str) = replace(str, "```@example"=>"```julia")
handle(model) = model.name*"_"*model.package_name

"""
    write_page(model; path=PATH_TO_MODEL_DOCS)

**Private method.**

Compose and write to file the documentation page for `model`. Here `model` is an entry in
the MLJ Model Registry, i.e., an element of `MLJModels.models(; wrappers=true)`. The file
name has the form `"ModelName_PackageName.md"`, for example,
`"DecisionTreeClassifier_DecisionTree.md"`. Such a page can be referenced from any other
markdown page in /docs/src/ like this: `[DecisionTreeClassifier](@ref
DecisionTreeClassifier_DecisionTree)`.

"""
function write_page(model; path=PATH_TO_MODEL_DOCS)
    id  = handle(model)
    pagename = id*".md"
    pagepath = joinpath(path, pagename)
    open(pagepath, "w") do stream
        header = "# [$(model.name)](@id $id)\n\n"
        md_page = doc(model.name, pkg=model.package_name)
        page = header*demote_headings(string(md_page)) |> remove_doc_refs |>
            remove_example_fencing
        write(stream, page)
        nothing
    end
end

# # METHODS TO GENERATE THE MODEL BROWSER PAGE

# read in dictionary of model descriptors:
const DESCRIPTORS_GIVEN_HANDLE =
    Pkg.TOML.parsefile(joinpath(@__DIR__, "ModelDescriptors.toml"))

# determined the list of all descriptors, ranked by frequency:
const descriptors = vcat(values(DESCRIPTORS_GIVEN_HANDLE)...)
const ranking = MLJBase.countmap(descriptors)
ranking["meta algorithms"] = 1e10
const DESCRIPTORS = sort(unique(descriptors), by=d -> ranking[d], rev=true)
const HANDLES = keys(DESCRIPTORS_GIVEN_HANDLE)

"""
    models_missing_descriptors()

Return a list of handles for those models in the registry not having the corresponding
handle as key in /docs/src/ModelDescriptors.toml.

"""
function models_missing_descriptors()
    handles = handle.(MLJ.models(wrappers=true))
    filter(handles) do h
        !(h in HANDLES)
    end
end

"""
    modelswith(descriptor)

**Private method.**

Return the list of  models with a given `descriptor`, such as "regressor", as
these appear in /src/docs/ModelDescriptors.toml.

"""
modelswith(descriptor) = filter(models(wrappers=true)) do model
    descriptor in DESCRIPTORS_GIVEN_HANDLE[handle(model)]
end

function title(camel_case_string)
    words = uppercasefirst.(split(camel_case_string, "_"))
    join(words, " ")
end

function doc_entry(descriptor)
    header = "##  $(title(descriptor))\n"
    parts = map(modelswith(descriptor)) do model
        interface_pkg = split(model.load_path, ".") |> first
        pkg = model.package_name
        pkgs = "$pkg.jl"
        if interface_pkg != pkg
            pkgs *= "/$interface_pkg.jl"
        end
        "- [$(model.name) ($pkgs)](@ref $(handle(model)))"
    end
    header*join(parts, "\n\n")
end

"""
    write_page()

Write the "Model Browser" page at /docs/model_browser.md"

"""
function write_page()
    header = """
           # [Model Browser](@id model_browser)

           Models may appear under multiple categories.

           Below an *encoder* is any transformer that does not fall under
           another category, such as "Missing Value Imputation" or "Dimension Reduction".

           """
    entries_for_contents = map(DESCRIPTORS) do d
        "[$(title(d))](@ref)"
    end
    contents = "### Categories\n"*join(entries_for_contents, "  |  ")
    body = join([doc_entry(d) for d in DESCRIPTORS], "\n\n")
    page = header*contents*"\n\n"*body
    open(joinpath(@__DIR__, "src", "model_browser.md"), "w") do stream
        write(stream, page)
    end
    nothing
end
