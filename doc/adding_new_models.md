# Implementing the MLJ interface for a learning algorithm 

This guide outlines the specification of the MLJ model interface. The
machine learning tools provided by MLJ.jl can be applied to the models
in any package implementing this interface. 

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


## Models

A *model* is an object storing hyper-parameters associated with some
machine learning algorithm, where "learning algorithm" is broadly
interpreted.  In MLJ, hyper-parameters include configuration
parameters, like the number of threads or the target element type, which
may or may not affect the final learning outcome.  However, the logging
level, `verbosity`, is excluded.

The name of the Julia type associated with a model indicates the
associated algorithm (e.g., `DecisionTreeClassifier`). The outcome of
training a learning algorithm is here called a *fit-result*. For
ordinary multilinear regression, for example, this would be the
coefficients and intercept. For a general supervised model, it is the
(generally minimal) information needed to make new predictions.

The ultimate supertype of all models is `MLJ.Model`.

Models defined outside of `MLJ` will fall into one of two subtypes:
`MLJ.Supervised` and `MLJ.Unsupervised`:

````julia
abstract type Supervised{R} <: Model end
abstract type Unsupervised <: Model end
````

Here the parameter `R` refers to a fit-result type. By declaring a
model to be a subtype of `Supervised{R}` you guarantee the fit-result
to be of type `R` and, if `R` is concrete, one may improve the
performance of homogeneous ensembles of the model (as defined by the
built-in MLJ module `Ensembles`). There is no abstract type for
fit-results because these types are generally declared outside of MLJ.

Associated with every concrete subtype of `Model` there must be a
`fit` method, which implements the associated algorithm to produce the
fit-result. Additionally, every `Supervised` model has a `predict`
method, while `Unsupersvised` models generally have a `transform`
method. More generally, methods such as these that are dispatched on a
model instance and a fit-result (plus other data) are called
*operations*. At present classifiers that predict probabilities may
optionally implement a `predict_class` operation, and `Unsupervised`
models may implement an `inverse_transform` operation.

## The Model API

<!-- Every package interface should live inside a submodule for namespace -->
<!-- hygiene (see the template at -->
<!-- "src/interfaces/DecisionTree.jl"). Ideally, package interfaces should -->
<!-- export no `struct` outside of the new model types they define, and -->
<!-- import only abstract types. All "structural" design should be -->
<!-- restricted to the MLJ core to prevent rewriting glue code when there -->
<!-- are design changes. -->

### New model type declarations

Here is an example of a concrete model type declaration:

````julia
import MLJ

R{S,T} = Tuple{Matrix{S},Vector{T}} where {S<:AbstractFloat,T<:AbstractFloat}

mutable struct KNNRegressor{S,T,M,K} <: MLJ.Supervised{R{S,T}}
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
(see below).


### Supervised models

Below we describe the compulsory and optional methods to be specified
for each concrete type `SomeSupervisedModel{R} <: Supervised{R}`. We
restrict attention to algorithms handling a *single* (univariate)
target. Differences in the multivariate case are described later.


#### Data for training and prediction, and the coerce method

The MLJ model specification has no explicit requirement for the
type of `X`, the argument representing input features appearing in the
compulsory `fit` and `predict` methods described below. However, the
MLJ user is free to present input features in any format implementing
the Queryverse [iterable tables
interface](https://github.com/queryverse/IterableTables.jl). If the
`fit` and `predict` methods require data in a different or more
specific form, one must overload the following method (whose fallback
just returns `Xtable`):

````julia
MLJ.coerce(model::Supervised{R}, Xtable) -> X
````

To this end, MLJ provides the convenience method `MLJ.matrix`;
`MLJ.matrix(Xtable)` is a two-dimensional `Array{T}` where `T` is the
tightest common type of elements of `Xtable`, and `Xtable` is any
iterable table.

In contrast, the target data `y` passed to training will always be an
`Vector{F}` for some `F<:AbstractFloat` - in the case of regressors -
or a `CategoricalVector` - in the case of classifiers. (At present only
target `CategoricalVector`s of the default reference type `UInt32` are
supported.) 


#### Compulsory methods

**The fit method.** The `fit` method returns three objects:

````julia 
MLJ.fit(model::SomeSupervisedModelType, verbosity::Int, X, y) -> fitresult, cache, report 
````

Here `fitresult::R` is the fit-result in the sense above. Any
training-related statistics, such as internal estimates of the
generalization error, feature rankings, and coefficients in linear
models, should be returned in the `report` object. How, or if, these
are generated should be controlled by hyper-parameters (the fields of
`model`). The `report` object returned is either a `Dict{Symbol,Any}`
object, or `nothing` if there is nothing to report. So for example,
`fit` might declare `report[:feature_importances] = ...`.  Reports get
merged with those generated by previous calls to `fit` by MLJ. The
value of `cache` can be `nothing` unless one is also defining an
`update` method (see below). The Julia type of `cache` is not
presently restricted.

It is not necessary for `fit` to provide dimension checks or to call
`clean!` on the model; MLJ will carry out such checks.

The method `fit` should never alter hyper-parameter values. If the
package is able to suggest better hyper-parameters, as a byproduct of
training, return these in the report field.

The `verbosity` level (0 for silent) is for passing to learning
algorithm itself. A `fit` method wrapping such an algorithm should
generally avoid doing any of its own logging.

**The predict method.**

````julia
MLJ.predict(model::SomeSupervisedModelType, fitresult, Xnew) -> yhat
````

Here `Xnew` is understood to be of the same type as `X` in the `fit`
method (MLJ will call `coerce` on the data provided by the user). 

**Prediction types for deterministic responses.** If the learning
algorithm predicts ordinary "point" values (as opposed to
probabilities) then `yhat` must have the same type as the target `y`
passed to the `fit` method. Note, in particular, that for point-value
predicting classifiers, the categorical vector returned by `predict`
**must have the same levels of the target data presented in
training**, even if not all levels appear in the prediction
itself. That is, we require `levels(yhat) == levels(y)`.

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
function MLJ.fit(model::SomeSupervisedModelType, verbosity, X, y)
    decoder = MLJ.CategoricalDecoder(y, eltype=Int64)
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
function MLJ.predict(model::SomeSupervisedModelType, fitresult, Xnew)
    decoder, core_fitresult = fitresult
    yhat = SomePackage.predict(core_fitresult, Xnew)
	return inverse_transform(decoder, yhat)
end
````
Query `?MLJ.DecodeCategorical` for more information.

**Prediction types for probablistic responses.** If, instead, an algorithm
learns a probability distribution (as in classification by logistic
regression, for example) then `yhat` must be a `Vector` whose elements
are distributions (one distribution per row of `Xnew`). 

A *distribution* is any instance of a subtype of
`Distributions.Distribution` from the package Distributions.jl, or an
instance of the additional types `UnivariateNominal` and
`MultivariateNominal` defined in MLJInterface.jl (or any other type
having, as a bare minimum, implementations of `Base.rand` and
`Distributions.pdf`). Use `UnivariateNominal` for probabilistic
classifiers with a single nominal target. For example, suppose
`levels(y)=["yes", "no", "maybe"]` and set `L=levels(y)`. Then, if the
predicted probabilities for some input pattern are `[0.1, 0.7, 0.2]`,
respectively, then the prediction returned for that pattern will be
`UnivariateNominal(L, [0.1, 0.7, 0.2])`. Query `?UnivariateNominal`
for more information.


#### Optional methods

Probabilistic models (in the sense above) may optionally implement a
`predict_mode` (classifiers) or `predict_mean` (regressors) operation
that returns point estimates instead of a probability distribution.

**Metadata.** Ideally, a model `metadata` method should be
provided. This allows the `MLJ` user, through the `Task` interface, to
discover models that meet a given task specification. For a supervised
model type `SomeModelType`, `metadata(SomeModelType)` should return a
dictionary with all keys shown in the example below:

````julia
function MLJ.metadata(::Type{RidgeRegressor})
    d = Dict{String,String}()
    d["package name"] = "MultivariateStats"
    d["package uuid"] = "6f286f6a-111f-5878-ab1e-185364afe411"
	d["is_pure_julia"] = "yes"
    d["properties"] = ["can rank feature importances",]
    d["operations"] = ["predict",]
    d["inputs_can_be"] = ["numeric",]
    d["outputs_are"] = ["numeric", "deterministic", "univariate"]
    return d
end
````

Note that for the last four keys, `metadata(SomeModelType)[key]` is a
vector of strings.  Permitted elements are indicated below:

key              | permitted values
-----------------|----------------------------------------------------
`"is_pure_julia"`| `"yes"`, `"no"`
`"properties"`   | unrestricted
`"operations"`   | `"predict"`, `"predict_mean"`, `"predict_mode"`, `"transform"`, `"inverse_transform"`
`"inputs_can_be"`| `"numeric"`, `"nominal"`, `"missing"`
`"outputs_are"`  | `"numeric"`/`"nominal"`, `"binary"`/`"multiclass"`, `"deterministic"`/`"probabilistic"`, `"univariate"`/`multivariate"`

For `metadata(SomeModelType)["inputs_are"]` list all that apply. For
`metadata(SomeModelType)["outputs_are"]` you must specify one from
each pair of values shown, with the exception of
`"binary"`/`"multiclass"` if `SomeModelType` is a regressor.  

A supervised model is "multivariate" if it can handle multiple targets
(see below).

**The clean! method.**

````julia
MLJ.clean!(model::Supervised) -> message::String
````

This method checks and corrects for invalid fields (hyper-parameters),
returning a warning `message` explaining what has been changed. It should
only throw an exception as a last resort. This method is called by the
model constructor, and by MLJ before any `fit` call.

**The update! method.**

````julia
MLJ.update(model::SomeSupervisedModelType, verbosity, old_fitresult, old_cache, X, y) -> fitresult, cache, report
````

An `update` method may be overloaded to enable a call by MLJ to
retrain a model (on the same training data) to avoid repeating
computations unnecessarily.  For context, see ["MLJ
Internals"](internals.md). A fallback just calls `fit`.  Learning
networks wrapped as models constitute one use-case: One would
like component models to be retrained only when new hyper-parameter
values make this necessary. In this case MLJ provides a fallback
(specifically, the fallback is for any subtype of
`Supervised{Node}`. A second important use-case is iterative models,
where calls to increase the number of iterations only restarts the
iterative procedure if other hyper-parameters have also changed. For
an example see `builtins/Ensembles.jl`.

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

