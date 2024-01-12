using MLJTestIntegration, MLJModels, MLJ, Test, Markdown
import MLJTestIntegration as MTI
import Pkg.TOML as TOML

const JULIA_TEST_LEVEL = 1
const OTHER_TEST_LEVEL = 1


# # RECORD OF OUTSTANDING ISSUES

FILTER_GIVEN_ISSUE = Dict(
    "https://github.com/JuliaAI/CatBoost.jl/pull/28 (waiting for 0.3.3 release)" =>
        model -> model.name == "CatBoostRegressor",
    "LOCIDetector too slow to train!" =>
        model -> model.name == "LOCIDetector",
    "https://github.com/JuliaML/LIBSVM.jl/issues/98" =>
        model -> model.name == "LinearSVC" &&
        model.package_name == "LIBSVM",
    "https://github.com/OutlierDetectionJL/OutlierDetectionPython.jl/issues/4" =>
        model -> model.name == "CDDetector" &&
        model.package_name == "OutlierDetectionPython",
    "https://github.com/JuliaAI/CatBoost.jl/issues/22" =>
        model -> model.name == "CatBoostClassifier",
    "https://github.com/sylvaticus/BetaML.jl/issues/65" =>
        model -> model.name in ["KMeans", "KMedoids"] &&
        model.package_name == "BetaML",
    "https://github.com/JuliaAI/MLJTSVDInterface.jl/pull/17" =>
        model -> model.name == "TSVDTransformer",
    "https://github.com/alan-turing-institute/MLJ.jl/issues/1074" =>
        model -> model.name == "AutoEncoderMLJ",
    "https://github.com/sylvaticus/BetaML.jl/issues/64" =>
        model -> model.name =="GaussianMixtureClusterer" && model.package_name=="BetaML",
     "https://github.com/rikhuijzer/SIRUS.jl/issues/78" =>
        model -> model.package_name == "SIRUS",
    "https://github.com/lalvim/PartialLeastSquaresRegressor.jl/issues/29 "*
        "(still need release > 2.2.0)" =>
        model -> model.package_name == "PartialLeastSquaresRegressor",
    "MLJScikitLearnInterface - multiple issues, hangs tests, WIP" =>
        model -> model.package_name == "MLJScikitLearnInterface",
)

# # LOG OUTSTANDING ISSUES TO STDOUT

const MODELS= models();
const JULIA_MODELS = filter(m->m.is_pure_julia, MODELS);
const OTHER_MODELS = setdiff(MODELS, JULIA_MODELS);

const EXCLUDED_BY_ISSUE = filter(MODELS) do model
    any([p(model) for p in values(FILTER_GIVEN_ISSUE)])
end;

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
# pathololgy that prevents a valid test:

PATHOLOGIES = filter(MODELS) do model
    # in the subsampling occuring in stacking, we get a Cholesky
    # factorization fail (`PosDefException`):
    (model.name=="GaussianNBClassifier" && model.package_name=="NaiveBayes") ||
        # https://github.com/JuliaStats/MultivariateStats.jl/issues/224
        (model.name =="ICA" && model.package_name=="MultivariateStats") ||
        # in tuned_pipe_evaluation C library gives "Incorrect parameter: specified nu is
        # infeasible":
        (model.name in ["NuSVC", "ProbabilisticNuSVC"] &&
        model.package_name == "LIBSVM")
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

options = (
    level = JULIA_TEST_LEVEL,
    verbosity = 0, # bump to 2 to debug
    throw = true,
)
@testset "level 4 tests" begin
    println()
    for model in JULIA_MODELS

        # exclusions:
        model in WITHOUT_DATASETS && continue
        model in EXCLUDED_BY_ISSUE && continue

        print("\rTesting $(model.name) ($(model.package_name))                       ")
        @test isempty(MLJTestIntegration.test(model; mod=@__MODULE__, options...))
    end
end


# # NON-JULIA TESTS

options = (
    level = OTHER_TEST_LEVEL,
    verbosity = 0, # bump to 2 to debug
    throw = true,
)
@testset "level 3 tests" begin
    println()
    for model in OTHER_MODELS

        # exclusions:
        model in WITHOUT_DATASETS && continue
        model in EXCLUDED_BY_ISSUE && continue

        print("\rTesting $(model.name) ($(model.package_name))                       ")
        @test isempty(MLJTestIntegration.test(model; mod=@__MODULE__, options...))
    end
end

true
