# Adding Models for General Use

!!! note

	Models implementing the MLJ model interface according to the instructions given here should import MLJModelInterface version 1.0.0 or higher. This is enforced with a statement such as `MLJModelInterface = "^1" ` under `[compat]` in the Project.toml file of the package containing the implementation.

This guide outlines the specification of the MLJ model interface
and provides detailed guidelines for implementing the interface for
models intended for general use. See also the more condensed
[Quick-Start Guide to Adding Models](@ref).

For sample implementations, see
[MLJModels/src](https://github.com/JuliaAI/MLJModels.jl/tree/master/src/builtins).

Interface code can be hosted by the package providing the core machine
learning algorithm, or by a stand-alone "interface-only" package, using
the template
[MLJExampleInterface.jl](https://github.com/JuliaAI/MLJExampleInterface.jl)
(see [Where to place code implementing new models](@ref) below).

The machine learning tools provided by MLJ can be applied to the
models in any package that imports the package
[MLJModelInterface](https://github.com/JuliaAI/MLJModelInterface.jl) and
implements the API defined there, as outlined below. For a
quick-and-dirty implementation of user-defined models see [Simple User
Defined Models](simple_user_defined_models.md).  To make new models
available to all MLJ users, see [Where to place code implementing new
models](@ref).


#### Important

[MLJModelInterface](https://github.com/JuliaAI/MLJModelInterface.jl)
is a very light-weight interface allowing you to *define* your
interface, but does not provide the functionality required to use or
test your interface; this requires
[MLJBase](https://github.com/JuliaAI/MLJBase.jl).  So,
while you only need to add `MLJModelInterface` to your project's
[deps], for testing purposes you need to add
[MLJBase](https://github.com/JuliaAI/MLJBase.jl) to your
project's [extras] and [targets]. In testing, simply use `MLJBase` in
place of `MLJModelInterface`.

It is assumed the reader has read [Getting Started](index.md).
To implement the API described here, some familiarity with the
following packages is also helpful:

- [ScientificTypes.jl](https://github.com/JuliaAI/ScientificTypes.jl)
  (for specifying model requirements of data)

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
  (for probabilistic predictions)

- [CategoricalArrays.jl](https://github.com/JuliaData/CategoricalArrays.jl)
  (essential if you are implementing a model handling data of
  `Multiclass` or `OrderedFactor` scitype; familiarity with
  `CategoricalPool` objects required)

- [Tables.jl](https://github.com/JuliaData/Tables.jl) (if your
  algorithm needs input data in a novel format).

In MLJ, the basic interface exposed to the user, built atop the model
interface described here, is the *machine interface*. After the first
reading of this document, the reader may wish to refer to [MLJ
Internals](internals.md) for context.


## Overview

A *model* is an object storing hyperparameters associated with some
machine learning algorithm, and that is all. In MLJ, hyperparameters
include configuration parameters, like the number of threads, and
special instructions, such as "compute feature rankings", which may or
may not affect the final learning outcome.  However, the logging level
(`verbosity` below) is excluded. *Learned parameters* (such as the
coefficients in a linear model) have no place in the model struct.

The name of the Julia type associated with a model indicates the
associated algorithm (e.g., `DecisionTreeClassifier`). The outcome of
training a learning algorithm is called a *fitresult*. For
ordinary multivariate regression, for example, this would be the
coefficients and intercept. For a general supervised model, it is the
(generally minimal) information needed to make new predictions.

The ultimate supertype of all models is `MLJModelInterface.Model`, which
has two abstract subtypes:

```julia
abstract type Supervised <: Model end
abstract type Unsupervised <: Model end
```

`Supervised` models are further divided according to whether they are
able to furnish probabilistic predictions of the target (which they
will then do by default) or directly predict "point" estimates, for each
new input pattern:

```julia
abstract type Probabilistic <: Supervised end
abstract type Deterministic <: Supervised end
```

Further division of model types is realized through [Trait declarations](@ref).

Associated with every concrete subtype of `Model` there must be a
`fit` method, which implements the associated algorithm to produce the
fitresult. Additionally, every `Supervised` model has a `predict`
method, while `Unsupervised` models must have a `transform`
method. More generally, methods such as these, that are dispatched on
a model instance and a fitresult (plus other data), are called
*operations*. `Probabilistic` supervised models optionally implement a
`predict_mode` operation (in the case of classifiers) or a
`predict_mean` and/or `predict_median` operations (in the case of
regressors) although MLJModelInterface also provides fallbacks that will suffice
in most cases. `Unsupervised` models may implement an
`inverse_transform` operation.


## New model type declarations and optional clean! method

Here is an example of a concrete supervised model type declaration,
for a model with a single hyper-parameter:

```julia
import MLJModelInterface
const MMI = MLJModelInterface

mutable struct RidgeRegressor <: MMI.Deterministic
	lambda::Float64
end
```

Models (which are mutable) should not be given internal
constructors. It is recommended that they be given an external lazy
keyword constructor of the same name. This constructor defines default values
for every field, and optionally corrects invalid field values by calling a
`clean!` method (whose fallback returns an empty message string):

```julia
function MMI.clean!(model::RidgeRegressor)
	warning = ""
	if model.lambda < 0
		warning *= "Need lambda ≥ 0. Resetting lambda=0. "
		model.lambda = 0
	end
	return warning
end

# keyword constructor
function RidgeRegressor(; lambda=0.0)
	model = RidgeRegressor(lambda)
	message = MMI.clean!(model)
	isempty(message) || @warn message
	return model
end
```

*Important.* The clean method must have the property that
`clean!(clean!(model)) == clean!(model)` for any instance `model`.

Although not essential, try to avoid `Union` types for model
fields. For example, a field declaration `features::Vector{Symbol}`
with a default of `Symbol[]` (detected with `isempty` method) is
preferred to `features::Union{Vector{Symbol}, Nothing}` with a default
of `nothing`.

### Hyper-parameters for parellizatioin options

The section [Acceleration and Parallelism](@ref) indicates how MLJ
models specify an option to run an algorithm using distributed
processing or multithreading. A hyper-parameter specifying such an
option should be called `acceleration`. Its value `a` should satisfy
`a isa AbstractResource` where `AbstractResource` is defined in the
ComputationalResources.jl package. An option to run on a GPU is
ordinarily indicated with the `CUDALibs()` resource.

### Hyper-parameter access and mutation

To support hyper-parameter optimization (see [Tuning Models](@ref))
any hyper-parameter to be individually controlled must be:

- property-accessible; nested property access allowed, as in
  `model.detector.K`

- mutable

For an un-nested hyper-parameter, the requirement is that
`getproperty(model, :param_name)` and `setproperty!(model,
:param_name, value)` have the expected behavior. (In hyper-parameter
tuning, recursive access is implemented using
[`MLJBase.recursive_getproperty`](@ref)` and
[`MLJBase.recursively_setproperty!`](@ref).)

Combining hyper-parameters in a named tuple does not generally
work, because, although property-accessible (with nesting), an
individual value cannot be mutated.

For a suggested way to deal with hyper-parameters varying in number,
see the
[implementation](https://github.com/JuliaAI/MLJBase.jl/blob/dev/src/composition/models/stacking.jl)
of `Stack`, where the model struct stores a varying number of base
models internally as a vector, but components are named at
construction and accessed by overloading `getproperty/setproperty!`
appropriately.

### Macro shortcut

An alternative to declaring the model struct, clean! method and
keyword constructor, is to use the `@mlj_model` macro, as in the
following example:

```julia
@mlj_model mutable struct YourModel <: MMI.Deterministic
	a::Float64 = 0.5::(_ > 0)
	b::String  = "svd"::(_ in ("svd","qr"))
end
```

This declaration specifies:

* A keyword constructor (here `YourModel(; a=..., b=...)`),
* Default values for the hyperparameters,
* Constraints on the hyperparameters where `_` refers to a value
  passed.

For example, `a::Float64 = 0.5::(_ > 0)` indicates that
the field `a` is a `Float64`, takes `0.5` as default value, and
expects its value to be positive.

You cannot use the `@mlj_model` macro if your model struct has type
parameters.

#### Known issue with @mlj_macro

Defaults with negative values can trip up the `@mlj_macro` (see [this
issue](https://github.com/JuliaAI/MLJBase.jl/issues/68)). So,
for example, this does not work:

```julia
@mlj_model mutable struct Bar
	a::Int = -1::(_ > -2)
end
```

But this does:

```julia
@mlj_model mutable struct Bar
	a::Int = (-)(1)::(_ > -2)
end
```


## Supervised models

### Mathematical assumptions

At present, MLJ's performance estimate functionality (resampling using
`evaluate`/`evaluate!`) tacitly assumes that feature-label pairs of
observations `(X1, y1), (X2, y2), (X2, y2), ...` are being modelled as
identically independent random variables (i.i.d.), and constructs some
kind of representation of an estimate of the conditional probability
`p(y | X)` (`y` and `X` *single* observations). It may be that a model
implementing the MLJ interface has the potential to make predictions
under weaker assumptions (e.g., time series forecasting
models). However the output of the compulsory `predict` method
described below should be the output of the model under the i.i.d
assumption.

In the future newer methods may be introduced to handle weaker
assumptions (see, e.g., [The predict_joint method](@ref) below).


### Summary of methods

The compulsory and optional methods to be implemented for each
concrete type `SomeSupervisedModel <: MMI.Supervised` are
summarized below.

An `=` indicates the return value for a fallback version of the
method.

Compulsory:

```julia
MMI.fit(model::SomeSupervisedModel, verbosity, X, y) -> fitresult, cache, report
MMI.predict(model::SomeSupervisedModel, fitresult, Xnew) -> yhat
```

Optional, to check and correct invalid hyperparameter values:

```julia
MMI.clean!(model::SomeSupervisedModel) = ""
```

Optional, to return user-friendly form of fitted parameters:

```julia
MMI.fitted_params(model::SomeSupervisedModel, fitresult) = fitresult
```

Optional, to avoid redundant calculations when re-fitting machines
associated with a model:

```julia
MMI.update(model::SomeSupervisedModel, verbosity, old_fitresult, old_cache, X, y) =
   MMI.fit(model, verbosity, X, y)
```

Optional, to specify default hyperparameter ranges (for use in tuning):

```julia
MMI.hyperparameter_ranges(T::Type) = Tuple(fill(nothing, length(fieldnames(T))))
```

Optional, if `SomeSupervisedModel <: Probabilistic`:

```julia
MMI.predict_mode(model::SomeSupervisedModel, fitresult, Xnew) =
	mode.(predict(model, fitresult, Xnew))
MMI.predict_mean(model::SomeSupervisedModel, fitresult, Xnew) =
	mean.(predict(model, fitresult, Xnew))
MMI.predict_median(model::SomeSupervisedModel, fitresult, Xnew) =
	median.(predict(model, fitresult, Xnew))
```

Required, if the model is to be registered (findable by general users):

```julia
MMI.load_path(::Type{<:SomeSupervisedModel})    = ""
MMI.package_name(::Type{<:SomeSupervisedModel}) = "Unknown"
MMI.package_uuid(::Type{<:SomeSupervisedModel}) = "Unknown"
```

```julia
MMI.input_scitype(::Type{<:SomeSupervisedModel}) = Unknown
```

Strongly recommended, to constrain the form of target data passed to fit:

```julia
MMI.target_scitype(::Type{<:SomeSupervisedModel}) = Unknown
```

Optional but recommended:

```julia
MMI.package_url(::Type{<:SomeSupervisedModel})  = "unknown"
MMI.is_pure_julia(::Type{<:SomeSupervisedModel}) = false
MMI.package_license(::Type{<:SomeSupervisedModel}) = "unknown"
```

If `SomeSupervisedModel` supports sample weights or class weights,
then instead of the `fit` above, one implements

```julia
MMI.fit(model::SomeSupervisedModel, verbosity, X, y, w=nothing) -> fitresult, cache, report
```

and, if appropriate

```julia
MMI.update(model::SomeSupervisedModel, verbosity, old_fitresult, old_cache, X, y, w=nothing) =
   MMI.fit(model, verbosity, X, y, w)
```

Additionally, if `SomeSupervisedModel` supports sample weights, one must declare

```julia
MMI.supports_weights(model::Type{<:SomeSupervisedModel}) = true
```

Optionally, an implementation may add a data front-end, for
transforming user data (such as a table) into some model-specific
format (such as a matrix), and for adding methods to specify how the said
format is resampled. (This alters the meaning of `X`, `y` and `w` in
the signatures of `fit`, `update`, `predict`, etc; see [Implementing a
data front-end](@ref) for details). This can provide the MLJ user
certain performance advantages when fitting a machine.

```julia
MLJModelInterface.reformat(model::SomeSupervisedModel, args...) = args
MLJModelInterface.selectrows(model::SomeSupervisedModel, I, data...) = data
```

Optionally, to customized support for serialization of machines (see
[Serialization](@ref)), overload

```julia
MMI.save(filename, model::SomeModel, fitresult; kwargs...) = fitresult
```

and possibly

```julia
MMI.restore(filename, model::SomeModel, serializable_fitresult) -> serializable_fitresult
```

These last two are unlikely to be needed if wrapping pure Julia code.


### The form of data for fitting and predicting

The model implementer does not have absolute control over the types of
data `X`, `y` and `Xnew` appearing in the `fit` and `predict` methods
they must implement. Rather, they can specify the *scientific type* of
this data by making appropriate declarations of the traits
`input_scitype` and `target_scitype` discussed later under [Trait
declarations](@ref).

*Important Note.* Unless it genuinely makes little sense to do so, the
MLJ recommendation is to specify a `Table` scientific type for `X`
(and hence `Xnew`) and an `AbstractVector` scientific type (e.g.,
`AbstractVector{Continuous}`) for targets `y`. Algorithms requiring
matrix input can coerce their inputs appropriately; see below.


#### Additional type coercions

If the core algorithm being wrapped requires data in a different or
more specific form, then `fit` will need to coerce the table into the
form desired (and the same coercions applied to `X` will have to be
repeated for `Xnew` in `predict`). To assist with common cases, MLJ
provides the convenience method
[`MMI.matrix`](@ref). `MMI.matrix(Xtable)` has type `Matrix{T}` where
`T` is the tightest common type of elements of `Xtable`, and `Xtable`
is any table. (If `Xtable` is itself just a wrapped matrix,
`Xtable=Tables.table(A)`, then `A=MMI.table(Xtable)` will be returned
without any copying.)

Alternatively, a more performant option is to implement a data
front-end for your model; see [Implementing a data front-end](@ref).

Other auxiliary methods provided by MLJModelInterface for handling tabular data
are: `selectrows`, `selectcols`, `select` and `schema` (for extracting
the size, names and eltypes of a table's columns). See [Convenience
methods](@ref) below for details.


#### Important convention

It is to be understood that the columns of table `X` correspond to
features and the rows to observations. So, for example, the predict
method for a linear regression model might look like `predict(model,
w, Xnew) = MMI.matrix(Xnew)*w`, where `w` is the vector of learned
coefficients.


### The fit method

A compulsory `fit` method returns three objects:

```julia
MMI.fit(model::SomeSupervisedModel, verbosity, X, y) -> fitresult, cache, report
```

1. `fitresult` is the fitresult in the sense above (which becomes an
	argument for `predict` discussed below).

2.  `report` is a (possibly empty) `NamedTuple`, for example,
	`report=(deviance=..., dof_residual=..., stderror=..., vcov=...)`.
	Any training-related statistics, such as internal estimates of the
	generalization error, and feature rankings, should be returned in
	the `report` tuple. How, or if, these are generated should be
	controlled by hyperparameters (the fields of `model`). Fitted
	parameters, such as the coefficients of a linear model, do not go
	in the report as they will be extractable from `fitresult` (and
	accessible to MLJ through the `fitted_params` method described below).

3.	The value of `cache` can be `nothing`, unless one is also defining
	an `update` method (see below). The Julia type of `cache` is not
	presently restricted.

!!! note

	The  `fit` (and `update`) methods should not mutate the `model`. If necessary, `fit` can create a `deepcopy` of `model` first.


It is not necessary for `fit` to provide type or dimension checks on
`X` or `y` or to call `clean!` on the model; MLJ will carry out such
checks.

The types of `X` and `y` are constrained by the `input_scitype` and
`target_scitype` trait declarations; see [Trait declarations](@ref)
below. (That is, unless a data front-end is implemented, in which case
these traits refer instead to the arguments of the overloaded
`reformat` method, and the types of `X` and `y` are determined by the
output of `reformat`.)

The method `fit` should never alter hyperparameter values, the sole
exception being fields of type `<:AbstractRNG`. If the package is able
to suggest better hyperparameters, as a byproduct of training, return
these in the report field.

The `verbosity` level (0 for silent) is for passing to the learning
algorithm itself. A `fit` method wrapping such an algorithm should
generally avoid doing any of its own logging.

*Sample weight support.* If
`supports_weights(::Type{<:SomeSupervisedModel})` has been declared
`true`, then one instead implements the following variation on the
above `fit`:

```julia
MMI.fit(model::SomeSupervisedModel, verbosity, X, y, w=nothing) -> fitresult, cache, report
```


### The fitted_params method

A `fitted_params` method may be optionally overloaded. Its purpose is
to provide MLJ access to a user-friendly representation of the
learned parameters of the model (as opposed to the
hyperparameters). They must be extractable from `fitresult`.

```julia
MMI.fitted_params(model::SomeSupervisedModel, fitresult) -> friendly_fitresult::NamedTuple
```

For a linear model, for example, one might declare something like
`friendly_fitresult=(coefs=[...], bias=...)`.

The fallback is to return `(fitresult=fitresult,)`.


### The predict method

A compulsory `predict` method has the form

```julia
MMI.predict(model::SomeSupervisedModel, fitresult, Xnew) -> yhat
```

Here `Xnew` will have the same form as the `X` passed to
`fit`.

Note that while `Xnew` generally consists of multiple observations
(e.g., has multiple rows in the case of a table) it is assumed, in view of
the i.i.d assumption recalled above, that calling `predict(..., Xnew)`
is equivalent to broadcasting some method `predict_one(..., x)` over
the individual observations `x` in `Xnew` (a method implementing the
probability distribution `p(X |y)` above).


#### Prediction types for deterministic responses.

In the case of `Deterministic` models, `yhat` should have the same
scitype as the `y` passed to `fit` (see above). If `y` is a
`CategoricalVector` (classification) than elements of the prediction
`yhat` **must have a pool == to the pool of the target `y` presented
in training**, even if not all levels appear in the training data or
prediction itself.

Unfortunately, code not written with the preservation of categorical
levels in mind poses special problems. To help with this,
MLJModelInterface provides some utilities:
[`MLJModelInterface.int`](@ref) (for converting a `CategoricalValue`
into an integer, the ordering of these integers being consistent with
that of the pool) and `MLJModelInterface.decoder` (for constructing a
callable object that decodes the integers back into `CategoricalValue`
objects). Refer to [Convenience methods](@ref) below for important
details.

Note that a decoder created during `fit` may need to be bundled with
`fitresult` to make it available to `predict` during re-encoding. So,
for example, if the core algorithm being wrapped by `fit` expects a
nominal target `yint` of type `Vector{<:Integer}` then a `fit` method
may look something like this:

```julia
function MMI.fit(model::SomeSupervisedModel, verbosity, X, y)
	yint = MMI.int(y)
	a_target_element = y[1]                # a CategoricalValue/String
	decode = MMI.decoder(a_target_element) # can be called on integers

	core_fitresult = SomePackage.fit(X, yint, verbosity=verbosity)

	fitresult = (decode, core_fitresult)
	cache = nothing
	report = nothing
	return fitresult, cache, report
end
```

while a corresponding deterministic `predict` operation might look like this:

```julia
function MMI.predict(model::SomeSupervisedModel, fitresult, Xnew)
	decode, core_fitresult = fitresult
	yhat = SomePackage.predict(core_fitresult, Xnew)
	return decode.(yhat)
end
```

For a concrete example, refer to the
[code](https://github.com/JuliaAI/MLJModels.jl/blob/master/src/ScikitLearn.jl)
for `SVMClassifier`.

Of course, if you are coding a learning algorithm from scratch, rather
than wrapping an existing one, these extra measures may be unnecessary.


#### Prediction types for probabilistic responses

In the case of `Probabilistic` models with univariate targets, `yhat`
must be an `AbstractVector` or table whose elements are distributions.
In the common case of a vector (single target), this means one
distribution per row of `Xnew`.

A *distribution* is some object that, at the least, implements
`Base.rng` (i.e., is something that can be sampled).  Currently, all
performance measures (metrics) defined in MLJBase.jl additionally
assume that distribution is either:

- An instance of some subtype of `Distributions.Distribution`, an
  abstract type defined in the
  [`Distributions.jl`](https://juliastats.org/Distributions.jl/stable/)
  package; or

- An instance of `CategoricalDistributions.UnivariateFinite`, from the
  [CategoricalDistributions.jl](https://github.com/JuliaAI/CategoricalDistributions.jl)
  package, *which should be used for all probabilistic classifiers*,
  i.e., for predictors whose target has scientific type
  `<:AbstractVector{<:Finite}`.

All such distributions implement the probability mass or density
function `Distributions.pdf`. If your model's predictions cannot be
predict objects of this form, then you will need to implement
appropriate performance measures to buy into MLJ's performance
evaluation apparatus.

An implementation can avoid CategoricalDistributions.jl as a
dependency by using the "dummy" constructor
`MLJModelInterface.UnivariateFinite`, which is bound to the true one
when MLJBase.jl is loaded.

For efficiency, one should not construct `UnivariateFinite` instances
one at a time. Rather, once a probability vector, matrix, or
dictionary is known, construct an instance of `UnivariateFiniteVector
<: AbstractArray{<:UnivariateFinite},1}` to return. Both
`UnivariateFinite` and `UnivariateFiniteVector` objects are
constructed using the single `UnivariateFinite` function.

For example, suppose the target `y` arrives as a subsample of some
`ybig` and is missing some classes:

```julia
ybig = categorical([:a, :b, :a, :a, :b, :a, :rare, :a, :b])
y = ybig[1:6]
```

Your fit method has bundled the first element of `y` with the
`fitresult` to make it available to `predict` for purposes of tracking
the complete pool of classes. Let's call this `an_element =
y[1]`. Then, supposing the corresponding probabilities of the observed
classes `[:a, :b]` are in an `n x 2` matrix `probs` (where `n` the number of
rows of `Xnew`) then you return

```julia
yhat = MLJModelInterface.UnivariateFinite([:a, :b], probs, pool=an_element)
```

This object automatically assigns zero-probability to the unseen class
`:rare` (i.e., `pdf.(yhat, :rare)` works and returns a zero
vector). If you would like to assign `:rare` non-zero probabilities,
simply add it to the first vector (the *support*) and supply a larger
`probs` matrix.

In a binary classification problem, it suffices to specify a single
vector of probabilities, provided you specify `augment=true`, as in
the following example, *and note carefully that these probabilities are
associated with the* **last** *(second) class you specify in the
constructor:*

```julia
y = categorical([:TRUE, :FALSE, :FALSE, :TRUE, :TRUE])
an_element = y[1]
probs = rand(10)
yhat = MLJModelInterface.UnivariateFinite([:FALSE, :TRUE], probs, augment=true, pool=an_element)
```

The constructor has a lot of options, including passing a dictionary
instead of vectors. See
`CategoricalDistributions.UnivariateFinite`](@ref) for details.

See
[LinearBinaryClassifier](https://github.com/JuliaAI/MLJModels.jl/blob/master/src/GLM.jl)
for example of a Probabilistic classifier implementation.

*Important note on binary classifiers.* There is no "Binary" scitype
distinct from `Multiclass{2}` or `OrderedFactor{2}`; `Binary` is just
an alias for `Union{Multiclass{2},OrderedFactor{2}}`. The
`target_scitype` of a binary classifier will generally be
`AbstractVector{<:Binary}` and according to the *mlj* scitype
convention, elements of `y` have type `CategoricalValue`, and *not*
`Bool`. See
[BinaryClassifier](https://github.com/JuliaAI/MLJModels.jl/blob/master/src/GLM.jl)
for an example.


### The predict_joint method

!!! warning "Experimental"

	The following API is experimental. It is subject to breaking changes during minor or major releases without warning.

```julia
MMI.predict_joint(model::SomeSupervisedModel, fitresult, Xnew) -> yhat
```

Any `Probabilistic` model type `SomeModel`may optionally implement a
`predict_joint` method, which has the same signature as `predict`, but
whose predictions are a single distribution (rather than a vector of
per-observation distributions).

Specifically, the output `yhat` of `predict_joint` should be an
instance of `Distributions.Sampleable{<:Multivariate,V}`, where
`scitype(V) = target_scitype(SomeModel)` and samples have length `n`,
where `n` is the number of observations in `Xnew`.

If a new model type subtypes `JointProbabilistic <: Probabilistic` then
implementation of `predict_joint` is compulsory.


### Training losses

```@docs
MLJModelInterface.training_losses
```

Trait values can also be set using the `metadata_model` method, see below.

### Feature importances

```@docs
MLJModelInterface.feature_importances
```

Trait values can also be set using the `metadata_model` method, see below.


### Trait declarations

Two trait functions allow the implementer to restrict the types of
data `X`, `y` and `Xnew` discussed above. The MLJ task interface uses
these traits for data type checks but also for model search. If they
are omitted (and your model is registered) then a general user may
attempt to use your model with inappropriately typed data.

The trait functions `input_scitype` and `target_scitype` take
scientific data types as values. We assume here familiarity with
[ScientificTypes.jl](https://github.com/JuliaAI/ScientificTypes.jl)
(see [Getting Started](index.md) for the basics).

For example, to ensure that the `X` presented to the
`DecisionTreeClassifier` `fit` method is a table whose columns all
have `Continuous` element type (and hence `AbstractFloat` machine
type), one declares

```julia
MMI.input_scitype(::Type{<:DecisionTreeClassifier}) = MMI.Table(MMI.Continuous)
```

or, equivalently,

```julia
MMI.input_scitype(::Type{<:DecisionTreeClassifier}) = Table(Continuous)
```

If, instead, columns were allowed to have either: (i) a mixture of `Continuous` and `Missing`
values, or (ii) `Count` (i.e., integer) values, then the
declaration would be

```julia
MMI.input_scitype(::Type{<:DecisionTreeClassifier}) = Table(Union{Continuous,Missing},Count)
```

Similarly, to ensure the target is an AbstractVector whose elements
have `Finite` scitype (and hence `CategoricalValue` machine type) we declare

```julia
MMI.target_scitype(::Type{<:DecisionTreeClassifier}) = AbstractVector{<:Finite}
```

#### Multivariate targets

The above remarks continue to hold unchanged for the case multivariate
targets.  For example, if we declare

```julia
target_scitype(SomeSupervisedModel) = Table(Continuous)
```

then this constrains the target to be any table whose columns have `Continous` element scitype (i.e., `AbstractFloat`), while

```julia
target_scitype(SomeSupervisedModel) = Table(Continuous, Finite{2})
```

restricts to tables with continuous or binary (ordered or unordered)
columns.

For predicting variable-length sequences of, say, binary values
(`CategoricalValue`s) with some common size-two pool) we declare

```julia
target_scitype(SomeSupervisedModel) = AbstractVector{<:NTuple{<:Finite{2}}}
```

The trait functions controlling the form of data are summarized as follows:

method                   | return type       | declarable return values     | fallback value
-------------------------|-------------------|------------------------------|---------------
`input_scitype`          | `Type`            | some scientfic type          | `Unknown`
`target_scitype`         | `Type`            | some scientific type         | `Unknown`


Additional trait functions tell MLJ's `@load` macro how to find your
model if it is registered, and provide other self-explanatory metadata
about the model:

method                       | return type       | declarable return values           | fallback value
-----------------------------|-------------------|------------------------------------|---------------
`load_path`                  | `String`          | unrestricted                       | "unknown"
`package_name`               | `String`          | unrestricted                       | "unknown"
`package_uuid`               | `String`          | unrestricted                       | "unknown"
`package_url`                | `String`          | unrestricted                       | "unknown"
`package_license`            | `String`          | unrestricted                       | "unknown"
`is_pure_julia`              | `Bool`            | `true` or `false`                  | `false`
`supports_weights`           | `Bool`            | `true` or `false`                  | `false`
`supports_class_weights`     | `Bool`            | `true` or `false`                  | `false`
`supports_training_losses`   | `Bool`            | `true` or `false`                  | `false`
`reports_feature_importances`| `Bool`            | `true` or `false`                  | `false`


Here is the complete list of trait function declarations for
`DecisionTreeClassifier`, whose core algorithms are provided by
DecisionTree.jl, but whose interface actually lives at
[MLJDecisionTreeInterface.jl](https://github.com/JuliaAI/MLJDecisionTreeInterface.jl).

```julia
MMI.input_scitype(::Type{<:DecisionTreeClassifier}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{<:DecisionTreeClassifier}) = AbstractVector{<:MMI.Finite}
MMI.load_path(::Type{<:DecisionTreeClassifier}) = "MLJDecisionTreeInterface.DecisionTreeClassifier"
MMI.package_name(::Type{<:DecisionTreeClassifier}) = "DecisionTree"
MMI.package_uuid(::Type{<:DecisionTreeClassifier}) = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
MMI.package_url(::Type{<:DecisionTreeClassifier}) = "https://github.com/bensadeghi/DecisionTree.jl"
MMI.is_pure_julia(::Type{<:DecisionTreeClassifier}) = true
```

Alternatively, these traits can also be declared using `MMI.metadata_pkg` and `MMI.metadata_model` helper functions as:

```julia
MMI.metadata_pkg(
  DecisionTreeClassifier,
  name="DecisionTree",
  packge_uuid="7806a523-6efd-50cb-b5f6-3fa6f1930dbb",
  package_url="https://github.com/bensadeghi/DecisionTree.jl",
  is_pure_julia=true
)

MMI.metadata_model(
  DecisionTreeClassifier,
  input_scitype=MMI.Table(MMI.Continuous),
  target_scitype=AbstractVector{<:MMI.Finite},
  load_path="MLJDecisionTreeInterface.DecisionTreeClassifier"
)
```

*Important.* Do not omit the `load_path` specification. If unsure what
it should be, post an issue at
[MLJ](https://github.com/alan-turing-institute/MLJ.jl/issues).

```@docs
MMI.metadata_pkg
```

```@docs
MMI.metadata_model
```


### Iterative models and the update! method

An `update` method may be optionally overloaded to enable a call by
MLJ to retrain a model (on the same training data) to avoid repeating
computations unnecessarily.

```julia
MMI.update(model::SomeSupervisedModel, verbosity, old_fitresult, old_cache, X, y) -> fit
result, cache, report
MMI.update(model::SomeSupervisedModel, verbosity, old_fitresult, old_cache, X, y, w=nothing) -> fit
result, cache, report
```

Here the second variation applies if `SomeSupervisedModel` supports
sample weights.

If an MLJ `Machine` is being `fit!` and it is not the first time, then
`update` is called instead of `fit`, unless the machine `fit!` has
been called with a new `rows` keyword argument. However, `MLJModelInterface`
defines a fallback for `update` which just calls `fit`. For context,
see [MLJ Internals](internals.md).

Learning networks wrapped as models constitute one use case (see
[Composing Models](index.md)): one would like each component model to
be retrained only when hyperparameter changes "upstream" make this
necessary. In this case, MLJ provides a fallback (specifically, the
fallback is for any subtype of `SupervisedNetwork =
Union{DeterministicNetwork,ProbabilisticNetwork}`). A second more
generally relevant use case is iterative models, where calls to
increase the number of iterations only restarts the iterative
procedure if other hyperparameters have also changed. (A useful method
for inspecting model changes in such cases is
`MLJModelInterface.is_same_except`. ) For an example, see
[MLJEnsembles.jl](https://github.com/JuliaAI/MLJEnsembles.jl).

A third use case is to avoid repeating the time-consuming preprocessing of
`X` and `y` required by some models.

If the argument `fitresult` (returned by a preceding
call to `fit`) is not sufficient for performing an update, the author
can arrange for `fit` to output in its `cache` return value any
additional information required (for example, pre-processed versions
of `X` and `y`), as this is also passed as an argument to the `update`
method.

### Implementing a data front-end

!!! note

	It is suggested that packages implementing MLJ's model API, that later implement a data front-end, should tag their changes in a breaking release. (The changes will not break the use of models for the ordinary MLJ user, who interacts with models exclusively through the machine interface. However, it will break usage for some external packages that have chosen to depend directly on the model API.)

```julia
MLJModelInterface.reformat(model, args...) -> data
MLJModelInterface.selectrows(::Model, I, data...) -> sampled_data
```

Models optionally overload `reformat` to define transformations of
user-supplied data into some model-specific representation (e.g., from
a table to a matrix). Computational overheads associated with multiple
`fit!`/`predict`/`transform` calls (on MLJ machines) are then avoided
when memory resources allow. The fallback returns `args` (no
transformation).

The `selectrows(::Model, I, data...)` method is overloaded to specify
how the model-specific data is to be subsampled, for some observation
indices `I` (a colon, `:`, or instance of
`AbstractVector{<:Integer}`). In this way, implementing a data
front-end also allows more efficient resampling of data (in user calls
to `evaluate!`).

After detailing formal requirements for implementing a data front-end,
we give a [Sample implementation](@ref). A simple implementation
[implementation](https://github.com/Evovest/EvoTrees.jl/blob/94b58faf3042009bd609c9a5155a2e95486c2f0e/src/MLJ.jl#L23)
also appears in the EvoTrees.jl package.

Here "user-supplied data" is what the MLJ user supplies when
constructing a machine, as in `machine(models, args...)`, which
coincides with the arguments expected by `fit(model, verbosity,
args...)` when `reformat` is not overloaded.

Implementing a `reformat` data front-end is permitted for any `Model`
subtype, except for subtypes of `Static`. Here is a complete list of
responsibilities for such an implementation, for some
`model::SomeModelType` (a sample implementation follows after):

- A `reformat(model::SomeModelType, args...) -> data` method must be
  implemented for each form of `args...` appearing in a valid machine
  construction `machine(model, args...)` (there will be one for each
  possible signature of `fit(::SomeModelType, ...)`).

- Additionally, if not included above, there must be a single argument
  form of reformat, `reformat(model::SommeModelType, arg) -> (data,)`,
  serving as a data front-end for operations like `predict`. It must
  always hold that `reformat(model, args...)[1] = reformat(model,
  args[1])`.

*Important.* `reformat(model::SomeModelType, args...)` must always
  return a tuple of the same length as `args`, even if this is one.

- `fit(model::SomeModelType, verbosity, data...)` should be
  implemented as if `data` is the output of `reformat(model,
  args...)`, where `args` is the data an MLJ user has bound to `model`
  in some machine. The same applies to any overloading of `update`.

- Each implemented operation, such as `predict` and `transform` - but
  excluding `inverse_transform` - must be defined as if its data
  arguments are `reformat`ed versions of user-supplied data. For
  example, in the supervised case, `data_new` in
  `predict(model::SomeModelType, fitresult, data_new)` is
  `reformat(model, Xnew)`, where `Xnew` is the data provided by the MLJ
  user in a call `predict(mach, Xnew)` (`mach.model == model`).

- To specify how the model-specific representation of data is to be
  resampled, implement `selectrows(model::SomeModelType, I, data...)
  -> resampled_data` for each overloading of `reformat(model::SomeModel,
  args...) -> data` above. Here `I` is an arbitrary abstract integer
  vector or `:` (type `Colon`).

*Important.* `selectrows(model::SomeModelType, I, args...)` must always
return a tuple of the same length as `args`, even if this is one.

The fallback for `selectrows` is described at [`selectrows`](@ref).


#### Sample implementation

Suppose a supervised model type `SomeSupervised` supports sample
weights, leading to two different `fit` signatures, and that it has a
single operation `predict`:

	fit(model::SomeSupervised, verbosity, X, y)
	fit(model::SomeSupervised, verbosity, X, y, w)

	predict(model::SomeSupervised, fitresult, Xnew)

Without a data front-end implemented, suppose `X` is expected to be a
table and `y` a vector, but suppose the core algorithm always converts
`X` to a matrix with features as rows (features corresponding to
columns in the table).  Then a new data-front end might look like
this:

	constant MMI = MLJModelInterface

	# for fit:
	MMI.reformat(::SomeSupervised, X, y) = (MMI.matrix(X, transpose=true), y)
	MMI.reformat(::SomeSupervised, X, y, w) = (MMI.matrix(X, transpose=true), y, w)
	MMI.selectrows(::SomeSupervised, I, Xmatrix, y) =
		(view(Xmatrix, :, I), view(y, I))
	MMI.selectrows(::SomeSupervised, I, Xmatrix, y, w) =
		(view(Xmatrix, :, I), view(y, I), view(w, I))

	# for predict:
	MMI.reformat(::SomeSupervised, X) = (MMI.matrix(X, transpose=true),)
	MMI.selectrows(::SomeSupervised, I, Xmatrix) = view(Xmatrix, I)

With these additions, `fit` and `predict` are refactored, so that `X`
and `Xnew` represent matrices with features as rows.


### Supervised models with a `transform` method

A supervised model may optionally implement a `transform` method,
whose signature is the same as `predict`. In that case, the
implementation should define a value for the `output_scitype` trait. A
declaration

```julia
output_scitype(::Type{<:SomeSupervisedModel}) = T
```

is an assurance that `scitype(transform(model, fitresult, Xnew)) <: T`
always holds, for any `model` of type `SomeSupervisedModel`.

A use-case for a `transform` method for a supervised model is a neural
network that learns *feature embeddings* for categorical input
features as part of overall training. Such a model becomes a
transformer that other supervised models can use to transform the
categorical features (instead of applying the higher-dimensional one-hot
encoding representations).

### Models that learn a probability distribution


!!! warning "Experimental"

	The following API is experimental. It is subject to breaking changes during minor or major releases without warning. Models implementing this interface will not work with MLJBase versions earlier than 0.17.5.

Models that fit a probability distribution to some `data` should be
regarded as `Probablisitic <: Supervised` models with target `y = data`
and `X = nothing`.

The `predict` method should return a single distribution.

A working implementation of a model that fits a `UnivariateFinite`
distribution to some categorical data using [Laplace
smoothing](https://en.wikipedia.org/wiki/Additive_smoothing)
controlled by a hyper-parameter `alpha` is given
[here](https://github.com/JuliaAI/MLJBase.jl/blob/d377bee1198ec179a4ade191c11fef583854af4a/test/interface/model_api.jl#L36).


### Serialization

!!! warning "New in MLJBase 0.20"

	The following API is incompatible with versions of MLJBase < 0.20, even for model implementations compatible with MLJModelInterface 1^


This section may be occasionally relevant when wrapping models
implemented in languages other than Julia.

The MLJ user can serialize and deserialize machines, as she would any
other julia object. (This user has the option of first removing data
from the machine. See [Saving machines](@ref) for details.) However, a
problem can occur if a model's `fitresult` (see [The fit
method](@ref)) is not a persistent object. For example, it might be a
C pointer that would have no meaning in a new Julia session.

If that is the case a model implementation needs to implement a `save`
and `restore` method for switching between a `fitresult` and some
persistent, serializable representation of that result.


#### The save method

```julia
MMI.save(model::SomeModel, fitresult; kwargs...) -> serializable_fitresult
```

Implement this method to return a persistent serializable
representation of the `fitresult` component of the `MMI.fit` return
value.

The fallback of `save` performs no action and returns `fitresult`.


#### The restore method

```julia
MMI.restore(model::SomeModel, serializable_fitresult) -> fitresult
```

Implement this method to reconstruct a valid `fitresult` (as would be returned by
`MMI.fit`) from a persistent representation constructed using
`MMI.save` as described above.

The fallback of `restore` performs no action and returns `serializable_fitresult`.


#### Example

Refer to the model implementations at
[MLJXGBoostInterface.jl](https://github.com/JuliaAI/MLJXGBoostInterface.jl/blob/42afbd2974bd3bd734994004e367c98964ed1262/src/MLJXGBoostInterface.jl#L679).


### Document strings

To be registered, MLJ models must include a detailed document string
for the model type, and this must conform to the standard outlined
below. We recommend you simply adapt an existing compliant document
string and read the requirements below if you're not sure, or to use
as a checklist. Here are examples of compliant doc-strings (go to the
end of the linked files):

- Regular supervised models (classifiers and regressors): [MLJDecisionTreeInterface.jl](https://github.com/JuliaAI/MLJDecisionTreeInterface.jl/blob/master/src/MLJDecisionTreeInterface.jl) (see the end of the file)

- Tranformers: [MLJModels.jl](https://github.com/JuliaAI/MLJModels.jl/blob/dev/src/builtins/Transformers.jl)

A utility function is available for generating a standardized header
for your doc-strings (but you provide most detail by hand):

```@docs
MLJModelInterface.doc_header
```

#### The document string standard

Your document string must include the following components, in order:

- A *header*, closely matching the example given above.

- A *reference describing the algorithm* or an actual description of
  the algorithm, if necessary. Detail any non-standard aspects of the
  implementation. Generally, defer details on the role of
  hyper-parameters to the "Hyper-parameters" section (see below).

- Instructions on *how to import the model type* from MLJ (because a user can already inspect the doc-string in the Model Registry, without having loaded the code-providing package).

- Instructions on *how to instantiate* with default hyper-parameters or with keywords.

- A *Training data* section: explains how to bind a model to data in a machine with all possible signatures (eg, `machine(model, X, y)` but also `machine(model, X, y, w)` if, say, weights are supported);  the role and scitype requirements for each data argument should be itemized.

- Instructions on *how to fit* the machine (in the same section).

- A *Hyper-parameters* section (unless there aren't any): an itemized list of the parameters, with defaults given.

- An *Operations* section: each implemented operation (`predict`, `predict_mode`, `transform`, `inverse_transform`, etc ) is itemized and explained. This should include operations with no data arguments, such as `training_losses` and `feature_importances`.

- A *Fitted parameters* section: To explain what is returned by `fitted_params(mach)` (the same as `MLJModelInterface.fitted_params(model, fitresult)` -  see later) with the fields of that named tuple itemized.

- A *Report* section (if `report` is non-empty): To explain what, if anything, is included in the `report(mach)`  (the same as the `report` return value of `MLJModelInterface.fit`) with the fields itemized.

- An optional but highly recommended *Examples* section, which includes MLJ examples, but which could also include others if the model type also implements a second "local" interface, i.e., defined in the same module. (Note that each module referring to a type can declare separate doc-strings which appear concatenated in doc-string queries.)

- A closing *"See also"* sentence which includes a `@ref` link to the raw model type (if you are wrapping one).


## Unsupervised models

Unsupervised models implement the MLJ model interface in a very
similar fashion. The main differences are:

- The `fit` method has only one training argument `X`, as in
  `MLJModelInterface.fit(model, verbosity, X)`. However, it has
  the same return value `(fitresult, cache, report)`. An `update`
  method (e.g., for iterative models) can be optionally implemented in
  the same way.

- A `transform` method is compulsory and has the same signature as
  `predict`, as in `MLJModelInterface.transform(model, fitresult, Xnew)`.

- Instead of defining the `target_scitype` trait, one declares an
  `output_scitype` trait (see above for the meaning).

- An `inverse_transform` can be optionally implemented. The signature
  is the same as `transform`, as in
  `MLJModelInterface.inverse_transform(model, fitresult, Xout)`, which:

   - must make sense for any `Xout` for which `scitype(Xout) <:
	 output_scitype(SomeSupervisedModel)` (see below); and

   - must return an object `Xin` satisfying `scitype(Xin) <:
	 input_scitype(SomeSupervisedModel)`.

- A `predict` method may be optionally implemented, and has the same
  signature as for supervised models, as in
  `MLJModelInterface.predict(model, fitresult, Xnew)`. A use-case is
  clustering algorithms that `predict` labels and `transform` new
  input features into a space of lower dimension. See [Transformers
  that also predict](@ref) for an example.

## Outlier detection models

!!! warning "Experimental API"

	The Outlier Detection API is experimental and may change in future
	releases of MLJ.

Outlier detection or *anomaly detection* is predominantly an unsupervised
learning task, transforming each data point to an outlier score quantifying
the level of "outlierness". However, because detectors can also be
semi-supervised or supervised, MLJModelInterface provides a collection of
abstract model types, that enable the different characteristics, namely:

- `MLJModelInterface.SupervisedDetector`
- `MLJModelInterface.UnsupervisedDetector`
- `MLJModelInterface.ProbabilisticSupervisedDetector`
- `MLJModelInterface.ProbabilisticUnsupervisedDetector`
- `MLJModelInterface.DeterministicSupervisedDetector`
- `MLJModelInterface.DeterministicUnsupervisedDetector`

All outlier detection models subtyping from any of the above supertypes
have to implement `MLJModelInterface.fit(model, verbosity, X, [y])`.
Models subtyping from either `SupervisedDetector` or `UnsupervisedDetector`
have to implement `MLJModelInterface.transform(model, fitresult, Xnew)`, which
should return the raw outlier scores (`<:Continuous`) of all points in `Xnew`.

Probabilistic and deterministic outlier detection models provide an additional
option to predict a normalized estimate of outlierness or a concrete
outlier label and thus enable evaluation of those models. All corresponding
supertypes have to implement (in addition to the previously described `fit`
and `transform`) `MLJModelInterface.predict(model, fitresult, Xnew)`, with
deterministic predictions conforming to `OrderedFactor{2}`, with the first
class being the normal class and the second class being the outlier.
Probabilistic models predict a `UnivariateFinite` estimate of those classes.

It is typically possible to automatically convert an outlier detection model
to a probabilistic or deterministic model if the training scores are stored in
the model's `report`. Below mentioned `OutlierDetection.jl` package, for example,
stores the training scores under the `scores` key in the `report` returned from
`fit`. It is then possible to use model wrappers such as
`OutlierDetection.ProbabilisticDetector` to automatically convert a model to
enable predictions of the required output type.

!!! note "External outlier detection packages"

	[OutlierDetection.jl](https://github.com/OutlierDetectionJL/OutlierDetection.jl)
	provides an opinionated interface on top of MLJ for outlier detection models,
	standardizing things like class names, dealing with training scores, score
	normalization and more.

## Convenience methods

```@docs
MLJBase.table
MLJBase.matrix
```

```@docs
MLJModelInterface.int
```

```@docs
CategoricalDistributions.UnivariateFinite
```

```@docs
CategoricalDistributions.classes
```

```@docs
MLJModelInterface.decoder
```

```@docs
MLJModelInterface.select
```

```@docs
MLJModelInterface.selectrows
```

```@docs
MLJModelInterface.selectcols
```

```@docs
MLJBase.recursive_getproperty
MLJBase.recursive_setproperty!
```


### Where to place code implementing new models

Note that different packages can implement models having the same name
without causing conflicts, although an MLJ user cannot simultaneously
*load* two such models.

There are two options for making a new model implementation available
to all MLJ users:

1. *Native implementations* (preferred option). The implementation
   code lives in the same package that contains the learning
   algorithms implementing the interface. An example is
   [`EvoTrees.jl`](https://github.com/Evovest/EvoTrees.jl/blob/master/src/MLJ.jl). In
   this case, it is sufficient to open an issue at
   [MLJ](https://github.com/alan-turing-institute/MLJ.jl) requesting
   the package to be registered with MLJ. Registering a package allows
   the MLJ user to access its models' metadata and to selectively load
   them.

2. *Separate interface package*. Implementation code lives in a
   separate *interface package*, which has the algorithm-providing
   package as a dependency. See the template repository
   [MLJExampleInterface.jl](https://github.com/JuliaAI/MLJExampleInterface.jl).

Additionally, one needs to ensure that the implementation code defines
the `package_name` and `load_path` model traits appropriately, so that
`MLJ`'s `@load` macro can find the necessary code (see
[MLJModels/src](https://github.com/JuliaAI/MLJModels.jl/tree/master/src)
for examples).

### How to add models to the MLJ model registry?

The MLJ model registry is located in the [MLJModels.jl
repository](https://github.com/JuliaAI/MLJModels.jl). To
add a model, you need to follow these steps

- Ensure your model conforms to the interface defined above

- Raise an issue at
  [MLJModels.jl](https://github.com/JuliaAI/MLJModels.jl/issues)
  and point out where the MLJ-interface implementation is, e.g. by
  providing a link to the code.

- An administrator will then review your implementation and work with
  you to add the model to the registry
