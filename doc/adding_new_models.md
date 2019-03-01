# Adding New Models

This guide outlines the specification of the MLJ model interface and
provides guidelines for implementing the interface for models defined
in external packages. For sample implementations, see
[MLJModels/src](https://github.com/alan-turing-institute/MLJModels.jl/tree/master/src).

The machine learning tools provided by MLJ can be applied to the
models in any package that imports the package
[MLJBase](https://github.com/alan-turing-institute/MLJBase.jl) and
implements the API defined there, as outlined below. For a quick and
dirty implementation of user-defined models see [here]().  To make new
models available to all MLJ users, see [Where to place code
implementing new models](#Where-to-place-code-implementing-new-models)
below.

It is assumed the reader has read [Getting Started](getting_started.md).
To implement the API described here, some familiarity with the
following packages is also helpful:

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
(for probabilistic predictions)

- [CategoricalArrays.jl](https://github.com/JuliaData/CategoricalArrays.jl)
(essential if you are implementing a model handling data of
`Multiclass` or `FiniteOrderedFactor` scitype)

- [Tables.jl](https://github.com/JuliaData/Tables.jl) (if you're
algorithm needs input data in a novel format).

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
reading of this document, the reader may wish to refer to [MLJ
Internals](internals.md) for context.


### Overview

A *model* is an object storing hyperparameters associated with some
machine learning algorithm, where "learning algorithm" is broadly
interpreted.  In MLJ, hyperparameters include configuration
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
has two abstract subtypes:

````julia
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

> The necessity to declare the fitresult type `R` may disappear in the future (issue #93).

`Supervised` models are further divided according to whether they are
able to furnish probabilistic predictions of the target(s) (which they
will do so by default) or directly predict "point" estimates, for each
new input pattern:

````julia
abstract type Probabilistic{R} <: Supervised{R} end
abstract type Deterministic{R} <: Supervised{R} end
````

Further division of model types is realized through [trait declarations](#Trait-Declarations).

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


### New model type declarations and optional clean! method

Here is an example of a concrete supervised model type declaration, made after defining an appropriate fit-result type (an optional step):

````julia
import MLJ

struct LinearFitResult{F<:AbstractFloat} <: MLJBase.MLJType
    coefficients::Vector{F}
    bias::F
end

mutable struct RidgeRegressor{F} <: MLJBase.Deterministic{LinearFitResult{F}}
    target_type::Type{F}
    lambda::Float64
end
````

Models (which are mutable) should not be given internal
constructors. It is recommended that they be given an external lazy
keyword constructor of the same name. This constructor defines default values for
every field, and optionally corrects invalid field values by calling a `clean!` method
(whose fallback returns an empty message string):


```julia
function MLJ.clean!(model::RidgeRegressor)
    warning = ""
    if model.lambda < 0
        warning *= "Need lambda ≥ 0. Resetting lambda=0. "
        model.lambda = 0
    end
    return warning
end

# keyword constructor
function RidgeRegressor(; target_type=Float64, lambda=0.0)

    model = RidgeRegressor(target_type, lambda)

    message = MLJBase.clean!(model)
    isempty(message) || @warn message

    return model
    
end
```


### The model API for supervised models

Below we describe the compulsory and optional methods to be specified
for each concrete type `SomeSupervisedModel{R} <: MLJBase.Supervised{R}`. 


#### The form of data for fitting and predicting

In every circumstance, the argument `X` passed to the `fit` method
described below, and the argument `Xnew` of the `predict` method, will
be some table supporting the
[Tables.jl](https://github.com/JuliaData/Tables.jl) API. The interface
implementer can control the scientific type of data appearing in `X`
with an appropriate `input_scitype` declaration (see [Trait
Declarations](#Trait-Declarations) below). If the core algorithm
requires data in a different or more specific form, then `fit` will
need to coerce the table into the form desired. To this end, MLJ
provides the convenience method `MLJBase.matrix`;
`MLJBase.matrix(Xtable)` is a two-dimensional `Array{T}` where `T` is
the tightest common type of elements of `Xtable`, and `Xtable` is any
table.

> Tables.jl has recently added a `matrix` method as well.

Other convenience methods provided by MLJBase for handling tabular
data are: `selectrows`, `selectcols`, `schema` (for extracting the
size, names and eltypes of a table) and `table` (for materializing an
abstract matrix, or named tuple of vectors, as a table matching a
given prototype). Query the doc-strings for details.

Note that generally the same type coercions applied to `X` by `fit` will need to
be applied by `predict` to `Xnew`. 

**Important convention** It is to be understood that the columns of the
table `X` correspond to features and the rows to patterns.

The form of the target data `y` passed to `fit` is constrained by the
`target_scitype` trait declaration. All elements of `y` will satisfy
`scitype(y) <: target_scitype(SomeSupervisedModelType)`. Furthermore,
for univariate targets, `y` is always a `Vector` or
`CategoricalVector`, according to the value of the trait:

`target_scitype(SomeSupervisedModelType)`  | type of `y`  | tightest known supertype of `eltype(y)`
------------------------------|---------------------------|--------------------------------------------
`Continuous`                  | `Vector`                  | `Real`
`<: Multiclass`               | `CategoricalVector`       | `Union{CategoricalString, CategoricalValue}`
`<: FiniteOrderedFactor`      | `CategoricalVector`       | `Union{CategoricalString, CategoricalValue}`
`Count`                       | `Vector`                  | `Integer`

So, for example, if your model is a binary classifier, you declare

````julia
target_scitype(SomeSupervisedModelType)=Multiclass{2}
````

If it can predict any number of classes, you might instead declare

````julia
target_scitype(SomeSupervisedModelType)=Union{Multiclass, FiniteOrderedFactor}
````

See also the table in [Getting Started](getting_started.md).

For multivariate targets, `y` will be a table whose columns have the
scitypes indicated in the `Tuple` type returned by `target_scitype`;
for example, if you declare `target_scitype(SomeSupervisedModelType) = Tuple{Continuous,Count}`,
then `y` will have two columns, the first with `Real` elements, the
second with `Integer` elements.


#### The fit method

A compulsory `fit` method returns three objects:

````julia
MLJBase.fit(model::SomeSupervisedModelType, verbosity::Int, X, y) -> fitresult, cache, report
````

Note: The `Int` typing of `verbosity` cannot be omitted.

1. `fitresult::R` is the fit-result in the sense above (which becomes an
    argument for `predict` discussed below).

2.  `report` is either a `Dict{Symbol,Any}` object, or `nothing` if
    there is nothing to report. So for example, `fit` might declare
    `report[:feature_importances] = ...`.  Any training-related
    statistics, such as internal estimates of the generalization
    error, feature rankings, and coefficients in linear models, should
    be returned in the `report` dictionary. How, or if, these are
    generated should be controlled by hyperparameters (the fields of
    `model`). Reports get merged with those generated by previous
    calls to `fit` by MLJ.

3.	The value of `cache` can be `nothing`, unless one is also defining an 
   `update` method (see below). The Julia type of `cache` is not presently restricted.

It is not necessary for `fit` to provide dimension checks or to call
`clean!` on the model; MLJ will carry out such checks.

The method `fit` should never alter hyperparameter values. If the
package is able to suggest better hyperparameters, as a byproduct of
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
`Deterministic` models, `yhat` must have the same form as the target
`y` passed to the `fit` method (see above discussion on the form of
data for fitting), with one exception: If predicting a `Count`, the
prediction may be `Continuous`. For all models predicting a
`Multiclass` or `FiniteOrderedFactor`, the categorical vectors
returned by `predict` **must have the levels in the categorical pool
of the target data presented in training**, even if not all levels
appear in the training data or prediction itself. That is, we must
have `levels(yhat) == levels(y)`.

Unfortunately, code not written with the preservation
of categorical levels in mind poses special problems. To help with
this, MLJ provides a utility `CategoricalDecoder` which can decode a
`CategoricalArray` into a plain array, and re-encode a prediction with
the original levels intact. The `CategoricalDecoder` object created
during `fit` will need to be bundled with `fitresult` to make it
available to `predict` during re-encoding. (If you are coding a learning algorithm 
from scratch, rather than 
wrapping an existing one, conversions may be unnecessary. It may suffice 
to record the pool of `y` and bundle that with the fitresult for `predict` to append 
to the levels of its categorical output, or to add to the support of the predicted 
probability distributions.)

So, for example, if the core algorithm being wrapped by `fit` expects
a nominal target `yint` of type `Vector{Int64}` then a `fit` method
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
while a corresponding deterministic `predict` operation might look like this:

````julia
function MLJBase.predict(model::SomeSupervisedModelType, fitresult, Xnew)
    decoder, core_fitresult = fitresult
    yhat = SomePackage.predict(core_fitresult, Xnew)
    return inverse_transform(decoder, yhat)
end
````
Query `?MLJBase.DecodeCategorical` for more information.

**Prediction types for probabilistic responses.** In the case of
`Probabilistic` models with univariate targets, `yhat` must be a
`Vector` whose elements are distributions (one distribution per row of
`Xnew`).

A *distribution* is any instance of a subtype of
`Distributions.Distribution` from the package Distributions.jl, or any
instance of the additional types `UnivariateNominal` and
`MultivariateNominal` defined in MLJBase.jl (or any other type
`D` you define for which `MLJBase.isdistribution(::D) = true`, meaning `Base.rand`
and `Distributions.pdf` are implemented, as well
`Distributions.mean`/`Distribution.median` or `Distributions.mode`).

Use `UnivariateNominal` for `Probabilistic` models predicting
`Multiclass` or `FiniteOrderedFactor` targets. For example, suppose
`levels(y)=["yes", "no", "maybe"]` and set `L=levels(y)`. Then, if the
predicted probabilities for some input pattern are `[0.1, 0.7, 0.2]`,
respectively, then the prediction returned for that pattern will be
`UnivariateNominal(L, [0.1, 0.7, 0.2])`. Query `?UnivariateNominal`
for more information.

The `predict` method will need access to all levels in the pool of the target
variable presented `y` presented for training, which consequently need
to be encoded in the `fitresult` returned by `fit`. If a
`CategoricalDecoder` object, `decoder`, has been bundled in
`fitresult`, as in the deterministic example above, then the levels
are given by `levels(decoder)`. Levels not observed in the training data 
(i.e., only in its pool) should be assigned probability zero.


#### Trait declarations

There are a number of recommended trait declarations for each model
mutable structure `SomeSupervisedModel <: Supervised` you define. Basic
fitting, resampling and tuning in MLJ does not require these traits
but some advanced MLJ meta-algorithms may require them now, or in the
future. In particular, MLJ's `models(::Task)` method (matching models
to user-specified tasks) can only identify models having a complete
set of trait declarations. A full set of declarations is shown below
for the `DecisionTreeClassifier` type (defined in the submodule DecisionTree_ of MLJModels):

````julia
MLJBase.load_path(::Type{<:DecisionTreeClassifier}) = "MLJModels.DecisionTree_.DecisionTreeClassifier" 
MLJBase.package_name(::Type{<:DecisionTreeClassifier}) = "DecisionTree"
MLJBase.package_uuid(::Type{<:DecisionTreeClassifier}) = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
MLJBase.package_url(::Type{<:DecisionTreeClassifier}) = "https://github.com/bensadeghi/DecisionTree.jl"
MLJBase.is_pure_julia(::Type{<:DecisionTreeClassifier}) = true
MLJBase.input_is_multivariate(::Type{<:DecisionTreeClassifier}) = true
MLJBase.input_scitypes(::Type{<:DecisionTreeClassifier}) = MLJBase.Continuous
MLJBase.target_scitype(::Type{<:DecisionTreeClassifier}) = MLJBase.Multiclass
````

Note that models predicting multivariate targets will need to need to have
`target_scitype` return an appropriate `Tuple` type. 

For an explanation of `Found` and `Other` in the table below, see [Scientific
Types](scientific_data_types.md).

method                   | return type       | declarable return values           | default value
-------------------------|-------------------|------------------------------------|---------------
`target_scitype`         | `DataType`        | subtype of `Found` or tuple of such types | `Other`
`input_scitypes`         | `DataType`        | subtype of `Union{Missing,Found}`  |  `Other`
`input_is_multivariate`  | `Bool`            | `true` or `false`                  | `true`
`is_pure_julia`          | `Bool`            | `true` or `false`                  | `false`
`load_path`              | `String`          | unrestricted                       | "unknown"
`package_name`           | `String`          | unrestricted                       | "unknown"
`package_uuid`           | `String`          | unrestricted                       | "unknown"
`package_url`            | `String`          | unrestricted                       | "unknown"


You can test declarations of traits by calling `info(SomeModelType)`.


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
just calls `fit`. For context, see [MLJ
Internals](internals.md). Learning networks wrapped as models
constitute one use-case: One would like each component model to be
retrained only when hyperparameter changes "upstream" make this
necessary. In this case MLJ provides a fallback (specifically, the
fallback is for any subtype of `Supervised{Node}`). A second important
use-case is iterative models, where calls to increase the number of
iterations only restarts the iterative procedure if other
hyperparameters have also changed. For an example, see
`builtins/Ensembles.jl`.

In the event that the argument `fitresult` (returned by a preceding
call to `fit`) is not sufficient for performing an update, the author
can arrange for `fit` to output in its `cache` return value any
additional information required, as this is also passed as an argument
to the `update` method.


#### Multivariate models

TODO


### Unsupervised models

TODO


### Where to place code implementing new models

Note that different packages can implement models having the same name
without causing conflicts, although an MLJ user cannot simultaneously
*load* two such models.

There are two options for making a new model available to all MLJ
users:

1. **Native implementations** (preferred option). The implementation
   code lives in the same package that contains the learning
   algorithms implementing the interface. In this case, it is
   sufficient to open an issue at
   [MLJRegistry](https://github.com/alan-turing-institute/MLJRegistry.jl)
   requesting the package to be registered with MLJ. Registering a package allows
   the MLJ user to access its models' metadata and to selectively load them.

2. **External implementations** (short-term alternative). The model
   implementation code is necessarily separate from the package
   `SomePkg` defining the learning algorithm being wrapped. In this
   case, the recommended procedure is to include the implementation
   code at
   [MLJModels/src](https://github.com/alan-turing-institute/MLJModels.jl/tree/master/src)
   via a pull-request, and test code at
   [MLJModels/test](https://github.com/alan-turing-institute/MLJModels.jl/tree/master/test).
   Assuming `SomePkg` is the only package imported by the
   implementation code, one needs to: (i) register `SomePkg` at
   MLJRegistry as explained above; and (ii) add a corresponding
   `@require` line in the PR to
   [MLJModels/src/MLJModels.jl](https://github.com/alan-turing-institute/MLJModels.jl/tree/master/src/MLJModels.jl)
   to enable lazy-loading of that package by MLJ (following the
   pattern of existing additions). If other packages must be imported,
   add them to the MLJModels project file after checking they are not
   already there. If it is really necessary, packages can be also
   added to Project.toml for testing purposes.
   
Additionally, one needs to ensure that the implementation code defines
the `package_name` and `load_path` model traits appropriately, so that
`MLJ`'s `@load` macro can find the necessary code (see
[MLJModels/src](https://github.com/alan-turing-institute/MLJModels.jl/tree/master/src)
for examples). The `@load` command can only be tested after
registration. If changes are made, lodge an issue at
[MLJRegistry](https://github.com/alan-turing-institute/MLJRegistry.jl)
to make the changes available to MLJ.  

