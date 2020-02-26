# Adding Models for General Use

This guide outlines the specification of the MLJ model interface
and provides detailed guidelines for implementing the interface for
models intended for general use. See also the more condensed
[Quick-Start Guide to Adding Models](@ref).

For sample implementations, see
[MLJModels/src](https://github.com/alan-turing-institute/MLJModels.jl/tree/master/src).

The machine learning tools provided by MLJ can be applied to the
models in any package that imports the package
[MLJModelInterface](https://github.com/alan-turing-institute/MLJModelInterface.jl) and
implements the API defined there, as outlined below. For a
quick-and-dirty implementation of user-defined models see [Simple User
Defined Models](simple_user_defined_models.md).  To make new models
available to all MLJ users, see [Where to place code implementing new
models](@ref).


#### Important

[MLJModelInterface](https://github.com/alan-turing-institute/MLJModelInterface.jl)
is a very light-weight interface allowing you to *define* your
interface, but does not provide the functionality required to use or
test your interface; this requires
[MLJBase](https://github.com/alan-turing-institute/MLJBase.jl).  So,
while you only need to add `MLJModelInterface` to your project's
[deps], for testing purposes you need to add
[MLJBase](https://github.com/alan-turing-institute/MLJBase.jl) to your
project's [extras] and [targets]. In testing, simply use `MLJBase` in
place of `MLJModelInterface`.

It is assumed the reader has read [Getting Started](index.md).
To implement the API described here, some familiarity with the
following packages is also helpful:

- [MLJScientificTypes.jl](https://github.com/alan-turing-institute/MLJScientificTypes.jl)
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
interface described here, is the *machine interface*. After a first
reading of this document, the reader may wish to refer to [MLJ
Internals](internals.md) for context.


## Overview

A *model* is an object storing hyperparameters associated with some
machine learning algorithm.  In MLJ, hyperparameters include configuration
parameters, like the number of threads, and special instructions, such
as "compute feature rankings", which may or may not affect the final
learning outcome.  However, the logging level (`verbosity` below) is
excluded.

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

Here is an example of a concrete supervised model type declaration:

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

An alternative to declaring the model struct, clean! method and keyword
constructor, is to use the `@mlj_model` macro, as in the following example:

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


## Supervised models

The compulsory and optional methods to be implemented for each
concrete type `SomeSupervisedModel <: MMI.Supervised` are
summarized below. An `=` indicates the return value for a fallback
version of the method.


### Summary of methods

Compulsory:

```julia
MMI.fit(model::SomeSupervisedModel, verbosity::Integer, X, y) -> fitresult, cache, report
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

If `SomeSupervisedModel` supports sample weights, then instead of the `fit` above, one implements

```julia
MMI.fit(model::SomeSupervisedModel, verbosity::Integer, X, y, w=nothing) -> fitresult, cache, report
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
`MMI.matrix`. `MMI.matrix(Xtable)` has type `Matrix{T}` where
`T` is the tightest common type of elements of `Xtable`, and `Xtable`
is any table.

Other auxiliary methods provided by MLJModelInterface for handling tabular data
are: `selectrows`, `selectcols`, `select` and `schema` (for extracting
the size, names and eltypes of a table's columns). See [Convenience
methods](@ref) below for details.


#### Important convention

It is to be understood that the columns of the table `X` correspond to
features and the rows to observations. So, for example, the predict
method for a linear regression model might look like `predict(model,
w, Xnew) = MMI.matrix(Xnew)*w`, where `w` is the vector of learned
coefficients.


### The fit method

A compulsory `fit` method returns three objects:

```julia
MMI.fit(model::SomeSupervisedModel, verbosity::Int, X, y) -> fitresult, cache, report
```

*Note.* The `Int` typing of `verbosity` cannot be omitted.

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

It is not necessary for `fit` to provide type or dimension checks on
`X` or `y` or to call `clean!` on the model; MLJ will carry out such
checks.

The method `fit` should never alter hyperparameter values, the sole
exception being fields of type `<:AbstractRNG`. If the package is able
to suggest better hyperparameters, as a byproduct of training, return
these in the report field.

The `verbosity` level (0 for silent) is for passing to learning
algorithm itself. A `fit` method wrapping such an algorithm should
generally avoid doing any of its own logging.

*Sample weight support.* If
`supports_weights(::Type{<:SomeSupervisedModel})` has been declared
`true`, then one instead implements the following variation on the
above `fit`:

```julia
MMI.fit(model::SomeSupervisedModel, verbosity::Int, X, y, w=nothing) -> fitresult, cache, report
```


### The fitted_params method

A `fitted_params` method may be optionally overloaded. It's purpose is
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

Here `Xnew` will have the same form as the `X` passed to `fit`.

#### Prediction types for deterministic responses.

In the case of `Deterministic` models, `yhat` should have the same
scitype as the `y` passed to `fit` (see above). Any `CategoricalValue`
or `CategoricalString` elements of `yhat` **must have a pool == to the
pool of the target `y` presented in training**, even if not all levels
appear in the training data or prediction itself. For example, in the
case of a univariate target, such as `scitype(y) <:
AbstractVector{Multiclass{3}}`, one requires `MLJ.classes(yhat[i]) ==
MLJ.classes(y[j])` for all admissible `i` and `j`. (The method
`classes` is described under [Convenience methods](@ref) below).

Unfortunately, code not written with the preservation of categorical
levels in mind poses special problems. To help with this, MLJModelInterface
provides three utility methods: `int` (for converting a
`CategoricalValue` or `CategoricalString` into an integer, the
ordering of these integers being consistent with that of the pool),
`decoder` (for constructing a callable object that decodes the
integers back into `CategoricalValue`/`CategoricalString` objects),
and `classes`, for extracting all the `CategoricalValue` or
`CategoricalString` objects sharing the pool of a particular
value/string. Refer to [Convenience methods](@ref) below for important
details.

Note that a decoder created during `fit` may need to be bundled with
`fitresult` to make it available to `predict` during re-encoding. So,
for example, if the core algorithm being wrapped by `fit` expects a
nominal target `yint` of type `Vector{<:Integer}` then a `fit` method
may look something like this:

```julia
function MMI.fit(model::SomeSupervisedModel, verbosity, X, y)
    yint = MMI.int(y)
    a_target_element = y[1]                    # a CategoricalValue/String
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
    return decode.(yhat)  # or decode(yhat) also works
end
```

For a concrete example, refer to the
[code](https://github.com/alan-turing-institute/MLJModels.jl/blob/master/src/ScikitLearn.jl)
for `SVMClassifier`.

Of course, if you are coding a learning algorithm from scratch, rather
than wrapping an existing one, these extra measures may be unnecessary.

#### Prediction types for probabilistic responses

In the case of `Probabilistic` models with univariate targets, `yhat`
must be an `AbstractVector` whose elements are distributions (one distribution
per row of `Xnew`).

Presently, a *distribution* is any object `d` for which
`MMI.isdistribution(::d) = true`, which is currently restricted to
objects subtyping `Distributions.Sampleable` from the package
Distributions.jl.

Use the distribution `MMI.UnivariateFinite` for `Probabilistic`
models predicting a target with `Finite` scitype (classifiers). In
this case each element of the training target `y` is a
`CategoricalValue` or `CategoricalString`, as in this contrived example:

```julia
using CategoricalArrays
y = Any[categorical([:yes, :no, :no, :maybe, :maybe])...]
```

Note that, as in this case, we cannot assume `y` is a
`CategoricalVector`, and we rely on elements for pool information (if
we need it); this is accessible using the convenience method
`MLJ.classes`:

```julia
julia> yes = y[1]
julia> levels = MMI.classes(yes)
3-element Array{CategoricalValue{Symbol,UInt32},1}:
 :maybe
 :no
 :yes
```

Now supposing that, for some new input pattern, the elements `yes =
y[1]` and `no = y[2]` are to be assigned respective probabilities of
0.2 and 0.8. Then the corresponding distribution `d` is constructed as
follows:

```julia
julia> d = MMI.UnivariateFinite([yes, no], [0.2, 0.8])
UnivariateFinite(:yes=>0.2, :maybe=>0.0, :no=>0.8)

julia> pdf(d, yes)
0.2

julia> maybe = y[4]; pdf(d, maybe)
0.0
```

Alternatively, a dictionary can be passed to the constructor.

See
[LinearBinaryClassifier](https://github.com/alan-turing-institute/MLJModels.jl/blob/master/src/GLM.jl)
for an example of a Probabilistic classifier implementation.


```@docs
UnivariateFinite
```

*Important note on binary classifiers.* There is no "Binary" scitype
distinct from `Multiclass{2}` or `OrderedFactor{2}`; `Binary` is just
an alias for `Union{Multiclass{2},OrderedFactor{2}}`. The
`target_scitype` of a binary classifier will generally be
`AbstractVector{<:Binary}` and according to the *mlj* scitype
convention, elements of `y` have type `CategoricalValue` or
`CategoricalString`, and *not* `Bool`. See
[BinaryClassifier](https://github.com/alan-turing-institute/MLJModels.jl/blob/master/src/GLM.jl)
for an example.


### Trait declarations

Two trait functions allow the implementer to restrict the types of
data `X`, `y` and `Xnew` discussed above. The MLJ task interface uses
these traits for data type checks but also for model search. If they
are omitted (and your model is registered) then a general user may
attempt to use your model with inappropriately typed data.

The trait functions `input_scitype` and `target_scitype` take
scientific data types as values. We assume here familiarity with
[MLJScientificTypes.jl](https://github.com/alan-turing-institute/MLJScientificTypes.jl)
(see [Getting Started](index.md) for the basics).

For example, to ensure that the `X` presented to the
`DecisionTreeClassifier` `fit` method is a table whose columns all have `Continuous` element type
(and hence `AbstractFloat` machine type), one declares

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
have `Finite` scitype (and hence `CategoricalValue` or
`CategoricalString` machine type) we declare

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

For predicting variable length sequences of, say, binary values
(`CategoricalValue`s or `CategoricalString`s with some common size-two
pool) we declare

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

method                   | return type       | declarable return values           | fallback value
-------------------------|-------------------|------------------------------------|---------------
`load_path`              | `String`          | unrestricted                       | "unknown"
`package_name`           | `String`          | unrestricted                       | "unknown"
`package_uuid`           | `String`          | unrestricted                       | "unknown"
`package_url`            | `String`          | unrestricted                       | "unknown"
`package_license`        | `String`          | unrestricted                       | "unknown"
`is_pure_julia`          | `Bool`            | `true` or `false`                  | `false`
`supports_weights`       | `Bool`            | `true` or `false`                  | `false`

**New.** A final trait you can optionally implement is the
`hyperparamter_ranges` trait. It declares default `ParamRange` objects
for one or more of your model's hyperparameters. This is for use (in
the future) by tuning algorithms (e.g., grid generation). It does not
represent the full space of *allowed values*. This information is
encoded in your `clean!` method (or `@mlj_model` call).

The value returned by `hyperparamter_ranges` must be a tuple of
`ParamRange` objects (query `?range` for details) whose length is the
number of hyperparameters (fields of your model). Note that varying a
hyperparameter over a specified range should not alter any type
parameters in your model struct (this never applies to numeric
ranges). If it doesn't make sense to provide a range for a parameter,
a `nothing` entry is allowed. The fallback returns a tuple of
`nothing`s.

For example, a three parameter model of the form

```julia
mutable struct MyModel{D} <: Deterministic
    alpha::Float64
	beta::Int
    distribution::D
end
```
you might declare (order matters):

```julia
MMI.hyperparameter_ranges(::Type{<:MyModel}) =
    (range(Float64, :alpha, lower=0, upper=1, scale=:log),
	 range(Int, :beta, lower=1, upper=Inf, origin=100, unit=50, scale=:log),
	 nothing)
```

Here is the complete list of trait function declarations for `DecisionTreeClassifier`
([source](https://github.com/alan-turing-institute/MLJModels.jl/blob/master/src/DecisionTree.jl)):

```julia
MMI.input_scitype(::Type{<:DecisionTreeClassifier}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{<:DecisionTreeClassifier}) = AbstractVector{<:MMI.Finite}
MMI.load_path(::Type{<:DecisionTreeClassifier}) = "MLJModels.DecisionTree_.DecisionTreeClassifier"
MMI.package_name(::Type{<:DecisionTreeClassifier}) = "DecisionTree"
MMI.package_uuid(::Type{<:DecisionTreeClassifier}) = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
MMI.package_url(::Type{<:DecisionTreeClassifier}) = "https://github.com/bensadeghi/DecisionTree.jl"
MMI.is_pure_julia(::Type{<:DecisionTreeClassifier}) = true
```

Alternatively these traits can also be declared using `MMI.metadata_pkg` and `MMI.metadata_model` helper functions as:

```julia
MMI.metadata_pkg(DecisionTreeClassifier,name="DecisionTree",
                     uuid="7806a523-6efd-50cb-b5f6-3fa6f1930dbb",
                     url="https://github.com/bensadeghi/DecisionTree.jl",
                     julia=true)   

MMI.metadata_model(DecisionTreeClassifier,
                        input=MMI.Table(MMI.Continuous),
                        target=AbstractVector{<:MMI.Finite},
                        path="MLJModels.DecisionTree_.DecisionTreeClassifier")
```

*Important.* Do not omit the `path` specifcation. 

```@docs
MMI.metadata_pkg
```

```@docs
MMI.metadata_model
```

You can test all your declarations of traits by calling `MLJBase.info_dict(SomeModel)`.


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

Learning networks wrapped as models constitute one use-case (see
[Composing Models](index.md)): one would like each component model to
be retrained only when hyperparameter changes "upstream" make this
necessary. In this case MLJ provides a fallback (specifically, the
fallback is for any subtype of `SupervisedNetwork =
Union{DeterministicNetwork,ProbabilisticNetwork}`). A second more
generally relevant use-case is iterative models, where calls to
increase the number of iterations only restarts the iterative
procedure if other hyperparameters have also changed. (A useful method
for inspecting model changes in such cases is
`MLJModelInterface.is_same_except`. ) For an example, see the MLJ [ensemble
code](https://github.com/alan-turing-institute/MLJ.jl/blob/master/src/ensembles.jl).

A third use-case is to avoid repeating time-consuming preprocessing of
`X` and `y` required by some models.

In the event that the argument `fitresult` (returned by a preceding
call to `fit`) is not sufficient for performing an update, the author
can arrange for `fit` to output in its `cache` return value any
additional information required (for example, pre-processed versions
of `X` and `y`), as this is also passed as an argument to the `update`
method.


## Unsupervised models

TODO

This is basically the same but with no target `y` appearing in the
signatures, and no `target_scitype` trait to declare. Instead, one
declares an `output_scitype` trait. Instead of implementing a
`predict` methods, one implements a `transform` operation, and an
optional `inverse_transform` operation.


## Convenience methods

```@docs
MLJModelInterface.int
```

```@docs
MLJModelInterface.classes
```

```@docs
MLJModelInterface.decoder
```

```@docs
MLJModelInterface.matrix
```

```@docs
MLJModelInterface.table
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


### Where to place code implementing new models

Note that different packages can implement models having the same name
without causing conflicts, although an MLJ user cannot simultaneously
*load* two such models.

There are two options for making a new model implementation available
to all MLJ users:

1. **Native implementations** (preferred option). The implementation
   code lives in the same package that contains the learning
   algorithms implementing the interface. In this case, it is
   sufficient to open an issue at
   [MLJ](https://github.com/alan-turing-institute/MLJ.jl)
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
   implementation code, one needs to: (i) register `SomePkg` with
   MLJ as explained above; and (ii) add a corresponding
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
registration. If changes are made, lodge an new issue at
[MLJ](https://github.com/alan-turing-institute/MLJ) requesting your
changes to be updated.

### How add models to the MLJ model registry?

The MLJ model registry is located in the [MLJModels.jl
repository](https://github.com/alan-turing-institute/MLJModels.jl). To
add a model, you need to follow these steps

- Ensure your model conforms to the interface defined above

- Raise an issue at
  https://github.com/alan-turing-institute/MLJModels.jl/issues and
  point out where the MLJ-interface implementation is, e.g. by
  providing a link to the code.

- An administrator will then review your implementation and work with
  you to add the model to the registry
