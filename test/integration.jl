using MLJTestIntegration, MLJModels, MLJ, Test, Markdown
import MLJTestIntegration as MTI
import Pkg.TOML as TOML
using Suppressor

const JULIA_TEST_LEVEL = 4
const OTHER_TEST_LEVEL = 3

# # IMPORTANT

# There are two main ways to flag a problem model for integration test purposes.

# - Adding to `FILTER_GIVEN_ISSUE` means the model is allowed to fail silently, unless
#  tests pass, a fact that will be reported in the log.

# - Adding to `PATHOLOGIES` completely excludes the model from testing.

# Obviously the first method is strongly preferred.


# # RECORD OF OUTSTANDING ISSUES

FILTER_GIVEN_ISSUE = Dict(
    "https://github.com/JuliaAI/MLJ.jl/issues/1085" =>
        model ->
        (model.name == "AdaBoostStumpClassifier" &&
        model.package_name == "DecisionTree") ||
        (model.name == "COFDetector" &&
        model.package_name == "OutlierDetectionNeighbors") ||
        (model.name == "TSVDTransformer" &&
        model.package_name == "TSVD"),
    "https://github.com/sylvaticus/BetaML.jl/issues/65" =>
        model -> model.name in ["KMeans", "KMedoids"] &&
        model.package_name == "BetaML",
    "https://github.com/JuliaAI/MLJ.jl/issues/1074" =>
        model -> model.name == "AutoEncoderMLJ",
     "https://github.com/rikhuijzer/SIRUS.jl/issues/78" =>
        model -> model.package_name == "SIRUS",
    "MLJScikitLearnInterface - multiple issues, WIP" =>
        model -> model.package_name == "MLJScikitLearnInterface" &&
        model.name in [
            "MultiTaskElasticNetCVRegressor",
            "MultiTaskElasticNetRegressor",
            "MultiTaskLassoCVRegressor",
            "MultiTaskLassoRegressor",
        ],
    "https://github.com/JuliaAI/FeatureSelection.jl/issues/15" =>
        model -> model.package_name == "FeatureSelection" &&
        model.name == "RecursiveFeatureElimination",
    "https://github.com/sylvaticus/BetaML.jl/issues/75" =>
        model -> model.package_name == "BetaML" &&
        model.name == "NeuralNetworkClassifier",
    # "https://github.com/JuliaAI/Imbalance.jl/issues/103" =>
    #     model -> model.package_name == "Imbalance",
)


# # LOG OUTSTANDING ISSUES TO STDOUT

const MODELS = models();
const JULIA_MODELS = filter(m->m.is_pure_julia, MODELS);
const OTHER_MODELS = setdiff(MODELS, JULIA_MODELS);

const EXCLUDED_BY_ISSUE = filter(MODELS) do model
    any([p(model) for p in values(FILTER_GIVEN_ISSUE)])
end

affected_packages = unique([m.package_name for m in EXCLUDED_BY_ISSUE])
n_excluded = length(EXCLUDED_BY_ISSUE)
report = """

# Integration Tests

Currently, $n_excluded models are excluded from integration tests because of outstanding
issues. When fixed, update `FILTER_GIVEN_ISSUE` in /test/integration.jl.

If an issue is related to model traits (aka metadata), then the MLJ Model Registry may
need to be updated to resolve the integration test failures. See the `MLJModels.@update`
document string for how to do that.

## Oustanding issues

""";
for issue in keys(FILTER_GIVEN_ISSUE)
    global report *= "\n- $issue\n"
end;
report *= "\n## Affected packages\n"
for pkg in affected_packages
    global report *= "\n- $pkg"
end;
report_md = Markdown.parse(report);

n_excluded > 0 && begin
    show(stdout, MIME("text/plain"), report_md)
    println()
    println()
    sleep(1)
end


# # FLAG MODELS THAT DON'T HAVE COMPATIBLE DATASETS FOR TESTING

# We use the version of `MLJTestIntegration.test` that infers appropriate datasets. The
# datasets provided by MLJTestIntegration.jl are not yet comprehensive, so we exclude
# models from testing when no compatible dataset can be found.
WITHOUT_DATASETS = filter(MODELS) do model
    # multi-target datasets:
    model.target_scitype <: Union{Table, AbstractMatrix} ||
        # https://github.com/JuliaAI/MLJTestInterface.jl/issues/19
        model.package_name == "MLJText" ||
        # univariate transformers:
        model.input_scitype <: AbstractVector ||
        # image data:
        model.input_scitype <: AbstractVector{<:Image} ||
        # other data:
        (model.name == "BernoulliNBClassifier" &&
        model.package_name == "MLJScikitLearnInterface") ||
        (model.name == "MultinomialNBClassifier" &&
        model.package_name == "NaiveBayes") ||
        (model.name == "OneRuleClassifier" &&
        model.package_name == "OneRule") ||
        (model.name == "ComplementNBClassifier" &&
        model.package_name == "MLJScikitLearnInterface") ||
        (model.name == "MultinomialNBClassifier" &&
        model.package_name == "MLJScikitLearnInterface") ||
        (model.name == "SMOTEN" &&
        model.package_name == "Imbalance")
end;

# To remove any warning issued below, update `WITHOUT_DATASETS` defined above:
for model in WITHOUT_DATASETS
    !isempty(MLJTestIntegration.datasets(model)) &&
        @warn "The model `$(model.name)` from `$(model.package_name)` "*
        "is currently excluded "*
        "from integration tests even though a compatible dataset appears "*
        "to be available now. "
end

# Additionally exclude some models for which the inferred datasets have a model-specific
# pathology that prevents valid generic test, or for some other reason requiring complete
# exclusion from testing.

PATHOLOGIES = filter(MODELS) do model
    # in the subsampling occuring in stacking, we get a Cholesky
    # factorization fail (`PosDefException`):
    (model.name=="GaussianNBClassifier" && model.package_name=="NaiveBayes") ||
        # https://github.com/JuliaStats/MultivariateStats.jl/issues/224
        (model.name =="ICA" && model.package_name=="MultivariateStats") ||
        # in tuned_pipe_evaluation C library gives "Incorrect parameter: specified nu is
        # infeasible":
        (model.name in ["NuSVC", "ProbabilisticNuSVC"] &&
        model.package_name == "LIBSVM") ||
        # too slow to train!
        (model.name == "LOCIDetector" && model.package_name == "OutlierDetectionPython") ||
        # TO REDUCE TESTING TIME
        model.package_name == "MLJScikitLearnInterface" ||
        # "https://github.com/MilesCranmer/SymbolicRegression.jl/issues/390" =>
        model.package_name == "SymbolicRegression" ||
        # can be removed after resolution of
        # https://github.com/JuliaAI/FeatureSelection.jl/issues/15
        # and a Model Registry update
        model.name == "RecursiveFeatureElimination"
end

WITHOUT_DATASETS = vcat(WITHOUT_DATASETS, PATHOLOGIES)


# # CHECK PROJECT FILE INCLUDES ALL MODEL-PROVIDING PACKAGES

# helper; `project_lines` are lines from a Project.toml file:
function pkgs(project_lines)
    project = TOML.parse(join(project_lines, "\n"))
    headings = Set(keys(project)) ∩ ["deps", "extras"]
    return vcat(collect.(keys.([project[h] for h in headings]))...)
end

# identify missing pkgs:
project_path = joinpath(@__DIR__, "..", "Project.toml")
project_lines = open(project_path) do io
    readlines(io)
end
pkgs_in_project = pkgs(project_lines)
registry_project_lines = MLJModels.Registry.registry_project()
pkgs_in_registry = pkgs(registry_project_lines)
missing_pkgs = setdiff(pkgs_in_registry, pkgs_in_project)

# throw error if there are any:
isempty(missing_pkgs) || error(
    "Integration tests cannot proceed because the following packages are "*
        "missing from the [extras] section of the MLJ Project.toml file: "*
        join(missing_pkgs, ", ")
)

# # LOAD ALL MODEL CODE

# Load all the model providing packages with a broad level=1 test:
MLJTestIntegration.test(MODELS, (nothing, ), level=1, throw=true, verbosity=0);


# # JULIA TESTS

const INFO_TEST_NOW_PASSING =
    "The model above now passes tests.\nConsider removing from "*
    "`FILTER_GIVEN_ISSUE` in test/integration.jl."

problems = []

const nmodels = length(JULIA_MODELS) + length(OTHER_MODELS)
i = 0
println()
for (model_set, level) in [
    (:JULIA_MODELS, JULIA_TEST_LEVEL),
    (:OTHER_MODELS, OTHER_TEST_LEVEL),
    ]
    set = eval(model_set)
    options = (
        ; level,
        verbosity = 0, # bump to 2 to debug
        throw = false,
    )
    @testset "$model_set tests" begin
        for model in set
            global i += 1
            progress = string("(", round(i/nmodels*100, digits=1), "%) Testing: ")

            # exclusions:
            model in WITHOUT_DATASETS && continue

            notice = "$(model.name) ($(model.package_name))"
            print("\r", progress, notice, "                       ")

            okay = @suppress isempty(MLJTestIntegration.test(
                model;
                mod=@__MODULE__,
                options...,
            ))
            if model in EXCLUDED_BY_ISSUE
                okay && (println(); @info INFO_TEST_NOW_PASSING)
            else
                okay || push!(problems, notice)
            end
        end
    end
end

okay = isempty(problems)
okay || print("Integration tests failed for these models: \n $problems")
println()

@test okay

true
