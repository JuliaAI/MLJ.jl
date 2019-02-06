# Implementing the MLJ interface for a learning algorithm

This guide outlines the specification of the MLJ model interface. The
machine learning tools provided by MLJ can be applied to the models
in any package that imports the module
[MLJBase](https://github.com/alan-turing-institute/MLJBase.jl)
and implements the API defined there as outlined below.

To implement the API described here, some familiarity with the
following packages is helpful:
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
(for probabilistic predictions),
[CategoricalArrays.jl](https://github.com/JuliaData/CategoricalArrays.jl)
(essential if you are implementing a classifier, or a learner that
handles categorical inputs),
[Tables.jl](https://github.com/JuliaData/Tables.jl) (if you're
algorithm needs input data in a novel format).

For a quick and dirty implementation of user-defined models see [here]().

<!-- As a temporary measure, -->
<!-- the MLJ package also implements the MLJ interface for some -->
<!-- non-compliant packages, using lazily loaded modules ("glue code") -->
<!-- residing in -->
<!-- [src/interfaces](https://github.com/alan-turing-institute/MLJ.jl/tree/master/src/interfaces) -->
<!-- of the MLJ.jl repository. A checklist for adding models in this latter -->
<!-- way is given at the end; a template is given here: -->
<!-- ["src/interfaces/DecisionTree.jl"](https://github.com/alan-turing-institute/MLJ.jl/tree/master/src/interfaces/DecisionTree.jl). -->

In MLJ, the basic interface exposed to the user, built atop the model
interface described here, is the *machine interface*. After a first
reading of this document, the reader may wish to refer to the
simplified description of the machine interface appearing under ["MLJ
Internals"](internals.md).

<!-- ### MLJ types -->

<!-- Every type introduced the core MLJ package should be a subtype of: -->

<!-- ``` -->
<!-- abstract type MLJType end -->
<!-- ``` -->

<!-- The Julia `show` method is informatively overloaded for this -->
<!-- type. Variable bindings declared with `@constant` "register" the -->
<!-- binding, which is reflected in the output of `show`. -->


## Overview

A *model* is an object storing hyper-parameters associated with some
machine learning algorithm, where "learning algorithm" is broadly
interpreted.  In MLJ, hyper-parameters include configuration
parameters, like the number of threads, and special instructions, such
as "compute feature rankings", which may or may not affect the final
learning outcome.  However, the logging level (`verbosity` below) is
excluded.

The name of the Julia type associated with a model indicates the
associated algorithm (e.g., `DecisionTreeClassifier`). The outcome of
training a learning algorithm is here called a *fit-result*. For
ordinary multilinear regression, for example, this would be the
coefficients and intercept. For a general supervised model, it is the
(generally minimal) information needed to make new predictions.

The ultimate supertype of all models is `MLJBase.Model`, which
has two abstract subtypes; quoting MLJBase.jl:

````julia
abstract type MLJType end
abstract type Supervised{R} <: Model end
abstract type Unsupervised <: Model end
````

Here the parameter `R` refers to a fit-result type. By declaring a
model to be a subtype of `MLJBase.Supervised{R}` you guarantee
the fit-result to be of type `R` and, if `R` is concrete, one may
improve the performance of homogeneous ensembles of the model (as
defined by the built-in MLJ `EnsembleModel` wrapper). There is no abstract
type for fit-results because these types are generally declared
outside of MLJBase.

`Supervised` models are further divided according to whether they are
able to furnish probabilistic predictions of the target(s) (which they
will do so by default) or directly predict "point" estimates, for each
new input pattern:

````julia
abstract type Probabilistic{R} <: Supervised{R} end
abstract type Deterministic{R} <: Supervised{R} end
````

Associated with every concrete subtype of `Model` there must be a
`fit` method, which implements the associated algorithm to produce the
fit-result. Additionally, every `Supervised` model has a `predict`
method, while `Unsupervised` models must have a `transform`
method. More generally, methods such as these, that are dispatched on
a model instance and a fit-result (plus other data), are called
*operations*. `Probabilistic` supervised models optionally implement
a `predict_mode` operation (in the case of classifiers) or a
`predict_mean` and/or `predict_median` operations (in the case of
regressors) overriding obvious fallbacks provided by
`MLJBase`. `Unsupervised` models may implement an
`inverse_transform` operation.


## The Model API

<!-- Every package interface should live inside a submodule for namespace -->
<!-- hygiene (see the template at -->
<!-- "src/interfaces/DecisionTree.jl"). Ideally, package interfaces should -->
<!-- export no `struct` outside of the new model types they define, and -->
<!-- import only abstract types. All "structural" design should be -->
<!-- restricted to the MLJ core to prevent rewriting glue code when there -->
<!-- are design changes. -->

### New model type declarations

Here is an example of a concrete supervised model type declaration:

````julia
import MLJ

# R{S, T} is the parametric type of the `fitresult` object
# that will be generated when `fit(KNNRegressor...)` is called
const R{S,T} = Tuple{Matrix{S},Vector{T}} where {S<:AbstractFloat,T<:AbstractFloat}

mutable struct KNNRegressor{S,T,M,K} <: MLJBase.Deterministic{R{S,T}}
    source_type::S
    target_type::T
    K::Int          
    metric::M
    kernel::K
end
````

Models (which are mutable) should not be given internal
constructors. It is recommended that they be given an external lazy
keyword constructor of the same name that defines default values (for
every field) and checks their validity, by calling a `clean!` method
(see further below).

```julia
# another, simple example, demonstrating with keyword with default value
# assuming the only hyperparameter in KMeans is "k" the number of clusters.
# the clean! method here should check that a given `k` is valid (>0).
KMeans(; k=3)
```


### Supervised models

Below we describe the compulsory and optional methods to be specified
for each concrete type `SomeSupervisedModel{R} <: MLJBase.Supervised{R}`. We
restrict attention to algorithms handling a *single* (univariate)
target. Differences in the multivariate case are described later.


#### The form of data for fitting and prediction

The argument `X` passed to the `fit` method described below, and the
argument `Xnew` of the `predict` (or `transform`) method, are
arbitrary tables. Here *table* means an object supporting
the[Tables.jl](https://github.com/JuliaData/Tables.jl) interface. If
the core algorithm requires data in a different or more specific form,
then `fit` will need to coerce the table into the form desired. To
this end, MLJ provides the convenience method `MLJBase.matrix`;
`MLJBase.matrix(Xtable)` is a two-dimensional `Array{T}` where `T` is
the tightest common type of elements of `Xtable`, and `Xtable` is any
table. 

Other convenience methods provided by MLJBase for handling tabular
data are: `selectrows`, `selectcols`, `schema` (for extracting the
size, names and eltypes of a table) and `table` (for materializing an
abstract matrix, or named tuple of vectors, as a table matching a
given prototype). Query the doc-strings for details.

Note that generally the same type coercions applied to `X` by `fit` will need to
be applied by `predict` to `Xnew`. 


**Important convention** It is to be understood that the columns of the
table `X` correspond to features and the rows to patterns.

The target data `y` passed to `fit` will always be an
`Vector{F}` for some `F<:AbstractFloat` - in the case of regressors -
or a `CategoricalVector` - in the case of classifiers. (At present
only target `CategoricalVector`s of the default reference type
`UInt32` are supported.) If the target to be predicted is an *ordered*
factor, then an ordered `CategoricalVector` is to be expected; if the
ordered factor is infinite (unbounded), then a `Vector{<:Integer}` is
to be expected.


#### The fit method

A compulsory `fit` method returns three objects:

````julia
MLJBase.fit(model::SomeSupervisedModelType, verbosity::Int, X, y) -> fitresult, cache, report
````

Note: The `Int` typing of `verbosity` cannot be omitted.

1. `fitresult::R` is the fit-result in the sense above (which becomes an
    argument for `predict` discussed below). Any training-related statistics,
    such as internal estimates of the generalization error, feature rankings,
    and coefficients in linear models, should be returned in the `report`
    object. How, or if, these are generated should be controlled by hyper-
    parameters (the fields of `model`).
2.  `report` is either a `Dict{Symbol,Any}` object, or `nothing` if there is
    nothing to report. So for example, `fit` might declare
    `report[:feature_importances] = ...`.  Reports get merged with those
    generated by previous calls to `fit` by MLJ. The value of `cache` can be
    `nothing` unless one is also defining an `update` method (see below). The Julia type of `cache` is not presently restricted.
3. `cache` is either `nothing` or an object allowing to update a `fitresult`
    more efficiently in the case where a model can be updated.

It is not necessary for `fit` to provide dimension checks or to call
`clean!` on the model; MLJ will carry out such checks.

The method `fit` should never alter hyper-parameter values. If the
package is able to suggest better hyper-parameters, as a byproduct of
training, return these in the report field.

One should test that actual fit-results have the type declared in the model
`mutable struct` declaration. To help with this,
`MLJBase.fitresult_type(m)` returns the declared type, for any
supervised model (or model type) `m`.

The `verbosity` level (0 for silent) is for passing to learning
algorithm itself. A `fit` method wrapping such an algorithm should
generally avoid doing any of its own logging.

#### The predict method

The compulsory predict method has the form
````julia
MLJBase.predict(model::SomeSupervisedModelType, fitresult, Xnew) -> yhat
````

Here `Xnew` is an arbitrary table (see above).

**Prediction types for deterministic responses.** In the case of
`Deterministic` models, `yhat` must have the same type as the target
`y` passed to the `fit` method (see above discussion on the form of
data for fitting), with one exception: If predicting an infinite
ordered factor (where `fit` receives a `Vector{<:Integer}` object) the
prediction may be continuous, i.e., of type
`Vector{<:AbstractFloat}`. For all other classifiers, the categorical
vector returned by `predict` **must have the same levels of the target
data presented in training**, even if not all levels appear in the
prediction itself. That is, we require `levels(yhat) == levels(y)`.

For code not written with the preservation of categorical
levels in mind, MLJ provides a utility `CategoricalDecoder` which can
decode a `CategoricalArray` into a plain array, and re-encode a
prediction with the original levels intact. The `CategoricalDecoder`
object created during `fit` will need to be bundled with `fitresult`
to make it available to `predict` during re-encoding.

So, for example, if the core algorithm being wrapped by `fit` expects
a nominal target `yint` of type `Vector{Int64}` then the `fit` method
may look something like this:

````julia
function MLJBase.fit(model::SomeSupervisedModelType, verbosity, X, y)
    decoder = MLJBase.CategoricalDecoder(y, Int64)
	yint = transform(decoder, y)
	core_fitresult = SomePackage.fit(X, yint, verbosity=verbosity)
	fitresult = (decoder, core_fitresult)
	cache = nothing
	report = nothing
	return fitresult, cache, report
end
````
while the corresponding `predict` operation might look like this:

````julia
function MLJBase.predict(model::SomeSupervisedModelType, fitresult, Xnew)
    decoder, core_fitresult = fitresult
    yhat = SomePackage.predict(core_fitresult, Xnew)
	return inverse_transform(decoder, yhat)
end
````
Query `?MLJBase.DecodeCategorical` for more information.

**Prediction types for probabilistic responses.** In the case of
`Probabilistic` models, `yhat` must be a `Vector` whose elements are
distributions (one distribution per row of `Xnew`).

A *distribution* is any instance of a subtype of
`Distributions.Distribution` from the package Distributions.jl, or any
instance of the additional types `UnivariateNominal` and
`MultivariateNominal` defined in MLJBase.jl (or any other type
`D` for which `MLJBase.isdistribution(::D) = true`, meaning `Base.rand`
and `Distributions.pdf` are implemented, as well
`Distributions.mean`/`Distribution.median` or `Distributions.mode`).

Use `UnivariateNominal` for `Probabilistic` classifiers with a single
nominal target, whether binary or multiclass. For example, suppose
`levels(y)=["yes", "no", "maybe"]` and set `L=levels(y)`. Then, if the
predicted probabilities for some input pattern are `[0.1, 0.7, 0.2]`,
respectively, then the prediction returned for that pattern will be
`UnivariateNominal(L, [0.1, 0.7, 0.2])`. Query `?UnivariateNominal`
for more information.

#### Trait declarations

There are a number of recommended trait declarations for each concrete
subtype `SomeSupervisedModel <: Supervised`. Basic fitting, resampling
and tuning in MLJ does not require these traits but some advanced MLJ
meta-algorithms may require them now, or in the future. In particular,
MLJ's `models(::Task)` method (matching models to user-specified
tasks) can only identify models having a complete set of trait
declarations. A full set of declarations are shown below for the
`RidgeRegressor` type:

````julia
MLJBase.output_kind(::Type{<:RidgeRegressor}) = :continuous
MLJBase.output_quantity(::Type{<:RidgeRegressor}) = :univariate
MLJBase.input_kinds(::Type{<:RidgeRegressor}) = [:continuous, ]
MLJBase.input_quantity(::Type{<:RidgeRegressor}) = :multivariate
MLJBase.is_pure_julia(::Type{<:RidgeRegressor}) = :yes
MLJBase.load_path(::Type{<:RidgeRegressor}) = "MLJ.RidgeRegressor"
MLJBase.package_name(::Type{<:RidgeRegressor}) = "MultivariateStats"
MLJBase.package_uuid(::Type{<:RidgeRegressor}) = "6f286f6a-111f-5878-ab1e-185364afe411"
MLJBase.package_url((::Type{<:RidgeRegressor}) = "https://github.com/JuliaStats/MultivariateStats.jl"
````

method                   | return type       | declarable return values | default value
-------------------------|-------------------|---------------------------|-------------------
`output_kind`            | `Symbol`          |`:continuous`, `:binary`, `:multiclass`, `:ordered_factor_finite`, `:ordered_factor_infinite`, `:same_as_inputs` | `:unknown`
`output_quantity`        | `Symbol`          |`:univariate`, `:multivariate`| `:univariate`
`input_kinds`          | `Vector{Symbol}`  | one or more of: `:continuous`, `:multiclass`, `:ordered_factor_finite`, `:ordered_factor_infinite`, `:missing` | `Symbol[]`
`input_quantity`        | `Symbol`          | `:univariate`, `:multivariate` | `:multivariate`
`is_pure_julia`          | `Symbol`          | `:yes`, `:no`             | `:unknown`
`load_path`              | `String`          | unrestricted              | "unknown"
`package_name`           | `String`          | unrestricted              | "unknown"
`package_uuid`           | `String`          | unrestricted              | "unknown"
`package_url`            | `String`          | unrestricted              | "unknown"

Note that `:binary` does not mean *boolean*. Rather, it
means the model is a classifier but is unable to classify targets with more than two
classes. As explained above, all classifiers are passed training targets
as `CategoricalVector`s, whose element types are
arbitrary.

The option `:same_as_inputs` for `output_kind` is intended primarily
for transformers, such as MLJ's built-in `FeatureSelector`.

You can test declarations of traits by calling `info(SomeModelType)`.


#### The clean! method

A `clean!` method may optionally be overloaded (the default returns an
empty message without changing model fields):

````julia
MLJBase.clean!(model::Supervised) -> message::String
````

This method is for checking and correcting invalid fields
(hyper-parameters) of the model, returning a warning `message`
explaining what has been changed. It should only throw an exception as
a last resort. This method should be called by the model keyword
constructor, as shown in the example below, and is called by MLJ
before each call to `fit`.

````julia
mutable struct RidgeRegressor <: MLJBase.Deterministic{Tuple{Vector{Float64}, Float64}}
    lambda::Float64
end

function MLJBase.clean!(model::RidgeRegressor)
    warning = ""
    if  model.lambda < 1
	warning *= "Need lambda ≥ 0. Resetting lamda = 0.\n"
        model.pruning_purity = 1.0
    end
    return warning
end

function RidgeRegressor(; lambda=0.0)
    model =  RidgeRegressor(lambda)
    message = MLJBase.clean!(model)     
    isempty(message) || @warn message
    return model
end
````

#### The update! method

An `update` method may be overloaded to enable a call by MLJ to
retrain a model (on the same training data) to avoid repeating
computations unnecessarily.

````julia
MLJBase.update(model::SomeSupervisedModelType, verbosity, old_fitresult, old_cache, X, y) -> fitresult, cache, report
````

If an MLJ `Machine` is being `fit!` and it is not the first time, then
`update` is called instead of `fit` unless `fit!` has been called with
new rows. However, `MLJBase` defines a fallback for `update` which
just calls `fit`. For context, see ["MLJ
Internals"](internals.md). Learning networks wrapped as models
constitute one use-case: One would like each component model to be
retrained only when hyper-parameter changes "upstream" make this
necessary. In this case MLJ provides a fallback (specifically, the
fallback is for any subtype of `Supervised{Node}`). A second important
use-case is iterative models, where calls to increase the number of
iterations only restarts the iterative procedure if other
hyper-parameters have also changed. For an example see
`builtins/Ensembles.jl`.

In the event that the argument `fitresult` (returned by a preceding
call to `fit`) is not sufficient for performing an update, the author
can arrange for `fit` to output in its `cache` return value any
additional information required, as this is also passed as an argument
to the `update` method.


#### Multivariate models

TODO

<!-- ##  Checklist for new adding models  -->

<!-- At present the following checklist is just for supervised models in -->
<!-- lazily loaded interface implementations. -->

<!-- - Copy and edit file -->
<!-- ["src/interfaces/DecisionTree.jl"](../src/interfaces/DecisionTree.jl) -->
<!-- which is annotated for use as a template. Give your new file a name -->
<!-- identical to the package name, including ".jl" extension, such as -->
<!-- "DecisionTree.jl". Put this file in "src/interfaces/". -->

<!-- - Register your package for lazy loading with MLJ by finding out the -->
<!-- UUID of the package and adding an appropriate line to the `__init__` -->
<!-- method at the end of "src/MLJ.jl". It will look something like this: -->

<!-- ````julia -->
<!-- function __init__() -->
<!--    @load_interface DecisionTree "7806a523-6efd-50cb-b5f6-3fa6f1930dbb" lazy=true -->
<!--    @load_interface NewExternalPackage "893749-98374-9234-91324-1324-9134-98" lazy=true -->
<!-- end -->
<!-- ```` -->

<!-- With `lazy=true`, your glue code only gets loaded by the MLJ user -->
<!-- after they run `import NewExternalPackage`. For testing in your local -->
<!-- MLJ fork, you may want to set `lazy=false` but to use `Revise` you -->
<!-- will also need to move the `@load_interface` line out outside of the -->
<!-- `__init__` function.  -->

<!-- - Write self-contained test-code for the methods defined in your glue -->
<!-- code, in a file with name like "TestExternalPackage.jl", but -->
<!-- placed in "test/". This code should be wrapped in a module to prevent -->
<!-- namespace conflicts with other test code. For a module name, just -->
<!-- prepend "Test", as in "TestDecisionTree". See "test/TestDecisionTree.jl" -->
<!-- for an example. -->

<!-- - Do not add the external package to the `Project.toml` file in the -->
<!--   usual way. Rather, add its UUID to the `[extras]` section of -->
<!--   `Project.toml` and add the package name to `test = [Test", "DecisionTree", -->
<!--   ...]`. -->

<!-- - Add suitable lines to ["test/runtests.jl"](../test/runtests.jl) to -->
<!-- `include` your test file, for the purpose of testing MLJ core and all -->
<!-- currently supported packages, including yours. You can Test your code -->
<!-- by running `test MLJ` from the Julia interactive package manager. You -->
<!-- will need to `Pkg.dev` your local MLJ fork first. To test your code in -->
<!-- isolation, locally edit "test/runtest.jl" appropriately. -->

### Unsupervised models

<!--
TODO:
* specify that the output of `predict`/`transform` should be a table
-->
