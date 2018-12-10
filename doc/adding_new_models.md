# Implementing the MLJ interface for a learning algorithm 

This guide outlines the specification of the MLJ model interface. The
machine learning tools provided by MLJ.jl can be applied to the models
in any package implementing this interface. As a temporary measure,
the MLJ package also implements the MLJ interface for some
non-compliant packages, using lazily loaded modules ("glue code")
residing in
[src/interfaces](https://github.com/alan-turing-institute/MLJ.jl/tree/master/src/interfaces)
of the MLJ.jl repository. A checklist for adding models in this latter
way is given at the end; a template is given here:
["src/interfaces/DecisionTree.jl"](https://github.com/alan-turing-institute/MLJ.jl/tree/master/src/interfaces/DecisionTree.jl).

In MLJ the most elementary interface exposed to the user (built atop
the model interface described here) is the *machine interface*. Those
implementing the MLJ model interface for new algorithms may benefit
from the the simplified description of this interface appearing under
["MLJ Internals"](internals.md).

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
machine learning algorithm, where "learning algorithm" is
broadly interpreted.  In MLJ, hyper-parameters include configuration
parameters, like the number of threads, which may not affect the final
learning outcome.  However, the logging level, `verbosity`, is
excluded.

The name of the Julia type associated with a model indicates the
associated algorithm (e.g., `DecisionTreeClassifier`). The outcome of
training a learning algorithm is here called a *fit-result*. For a
linear model, for example, this would be the coefficients and
intercept. For a supervised model, it is the information needed to make
new predictions.

The ultimate supertype of all models is `MLJ.Model`.

Models defined outside of `MLJ` will fall into one of two subtypes: `MLJ.Supervised` and
`MLJ.Unsupervised`:

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
fit-result. At least one method dispatched on model
instances and a fit-result (`predict` for supervised
algorithms) must also be provided. Such methods, here called
*operations*, can have one of the following names: `predict`,
`predict_proba`, `transform`, `inverse_transform`, `se`, or
`evaluate`. 


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
every field) and checks their validity, by calling a `clean!` (see
below).


### Supervised models

Below we describe the compulsory and optional methods to be specified
for each concrete type `SomeSupervisedModel{R} <: Supervised{R}`.

#### Data for training and prediction

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

In contrast, the target data `y` passed to training will always be a
`Vector{F}` for some `F<:AbstractFloat` - in the case of regressors - or a
`CategoricalVector` - in the case of classifiers. (At present only
target `CategoricalVector`s of the default reference type `UInt32` are
supported.)

#### Compulsory methods

````julia
MLJ.fit(model::SomeSupervisedModelType, verbosity::Int, X, y) -> 
    fitresult, cache, report
````

Here `fitresult::R` is the fit-result in the sense above. Any
training-related statistics, such as internal estimates of the
generalization error, feature rankings, and coefficients in linear
models, should be returned in the `report` object. How, or if, these
are generated should be controlled by hyper-parameters. The `report`
object returned is either a `Dict{Symbol,Any}` object, or `nothing` if
there is nothing to report. So for example, `fit` might declare
`report[:feature_importances] = ...`.  Reports get merged with those
generated by previous calls to `fit` by MLJ. The value of `cache` can
be `nothing` unless one is also defining an `update` method (see
below). The Julia type of `cache` is not presently restricted.


It is not necessary for `fit` and `predict` methods to provide
dimension matching checks or to call `clean!` on the model; MLJ will
carry out such checks.

The method `fit` should never alter hyper-parameter values. If the
package is able to suggest better hyper-parameters, as a byproduct of
training, return these in the report field.

The `verbosity` level (0 for silent) is for passing to learning
algorithm itself. A `fit` method wrapping such an algorithm should
generally avoid doing any of its own logging.

````julia
yhat = MLJ.predict(model::SomeSupervisedModelType, fitresult, Xnew)
````

Here `Xnew` is understood to be of the same type as `X` in the `fit`
method (MLJ will call `coerce` on the data provided by the user).

The `predict` method should return objects of the same type as the
target `y` presented to `fit`. 

**Important note for classifiers.** In the case of classifiers, the
categorical array returned by `predict` *must have the same levels of
the target data presented in training*, even if not all levels appear
in the prediction itself. That is, we require

````
levels(predict(model, fitresult, Xnew)) == levels(y)
````

to be `true`. For code not written with the preservation of categorical
levels in mind, MLJ provides a utility `CategoricalDecoder` which can
decode a `CategoricalArray` into a plain array, and re-encode a
prediction with the original levels intact. The `CategoricalDecoder`
object created during `fit` will need to be bundled with `fitresult`
to make it available to `predict` during re-encoding.


#### Optional methods

**Metadata.** Ideally methods encoding certain model-type metadata
should be provided. This allows the `MLJ` user, through the `Task`
interface, to discover models that meet the task specification. For
example, in the current `DecisionTreeClassifier`, metadata is declared
as follows:

````julia
MLJ.properties(::Type{DecisionTreeClassifier}) = ()
MLJ.operations(::Type{DecisionTreeClassifier}) = (MLJ.predict, MLJ.predict_proba)
MLJ.inputs_can_be(::Type{DecisionTreeClassifier}) = (Numeric())
MLJ.outputs_are(::Type{DecisionTreeClassifier}) = (Nominal())
````

Available options can be gleaned from this code extract:

````julia
# `property(SomeModelType)` is a tuple of instances of:
""" Classification models with this property allow weighting of the target classes """
struct CanWeightTarget <: Property end
""" Models with this property can provide feature rankings or importance scores """
struct CanRankFeatures <: Property end

# `inputs_can_be(SomeModelType)` and `outputs_are(SomeModelType)` are tuples of
# instances of:
struct Nominal <: Property end
struct Numeric <: Property end
struct NA <: Property end

# additionally, `outputs_are(SomeModelType)` can include:
struct Probabilistic <: Property end
struct Multivariate <: Property end

# for `Model`s with nominal targets (classifiers)
# `outputs_are(SomeModelType)` could also include:
struct Multiclass <: Property end # can handle more than two classes
````

**Binary classifiers.** If `Probabilistic()` is in the tuple returned
by `outputs_are` then a `predict_proba` method must be provided (in
addition to `predict`) to predict probabilities instead of labels. It
should return a `Vector` of probabilities (one probability per input
pattern), this probability corresponding to the *first* level in
`levels(y)`.

**Multilabel classifiers.** If `Probability()` and `Multiclass()` are
both in the tuple returned by `outputs_are` then a `predict_proba`
method is to be implemented but its return value `yhat` must be an
`Array` object of size `(nrows, k)` where `nrows` is the number of
input patterns (the number of rows of the input data `Xnew`) and
`k=levels(y)`, where `y` is the data presented in training. The order
of the columns should coincide with the order of `levels(y)`. However,
in the special case `k=1` a single column should be output.

````julia
MLJ.clean!(model::Supervised) -> message::String
````

This method checks and corrects for invalid fields (hyper-parameters), returning a
warning `message`. Should only throw an exception as a last
resort. This method is called by the model constructor, and by MLJ
before any `fit` call.

````julia
MLJ.update(model::SomeSupervisedModelType, verbosity, old_fitresult, old_cache, X, y) ->  
    fitresult, cache, report
````

An `update` method may be overloaded to enable a call by MLJ to
retrain a model (on the same training data) to avoid repeating
computations unnecessarily. (A fallback just calls `fit`.)  Composite
models (subtypes of `Supervised{Node}`) constitute one use-case
(component models are only retrained when new hyper-parameter values
make this necessary) and in this case MLJ provides a fallback. A
second important use-case is iterative models, where calls to increase
the number of iterations only restarts the iterative procedure if
other hyper-parameters have also changed. For an example see
`builtins/Ensembles.jl`.

In the event that the argument `fitresult` (returned by a preceding
call to `fit`) is not sufficient for performing an update, the author
can arrange for `fit` to output in its `cache` return value any
additional information required, as this is also passed as an argument
to the `update` method.


##  Checklist for new adding models 

At present the following checklist is just for supervised models in
lazily loaded interface implementations.

- Copy and edit file
["src/interfaces/DecisionTree.jl"](../src/interfaces/DecisionTree.jl)
which is annotated for use as a template. Give your new file a name
identical to the package name, including ".jl" extension, such as
"DecisionTree.jl". Put this file in "src/interfaces/".

- Register your package for lazy loading with MLJ by finding out the
UUID of the package and adding an appropriate line to the `__init__`
method at the end of "src/MLJ.jl". It will look something like this:

````julia
function __init__()
   @load_interface DecisionTree "7806a523-6efd-50cb-b5f6-3fa6f1930dbb" lazy=true
   @load_interface NewExternalPackage "893749-98374-9234-91324-1324-9134-98" lazy=true
end
````

With `lazy=true`, your glue code only gets loaded by the MLJ user
after they run `import NewExternalPackage`. For testing in your local
MLJ fork, you may want to set `lazy=false` but to use `Revise` you
will also need to move the `@load_interface` line out outside of the
`__init__` function. 

- Write self-contained test-code for the methods defined in your glue
code, in a file with name like "TestExternalPackage.jl", but
placed in "test/". This code should be wrapped in a module to prevent
namespace conflicts with other test code. For a module name, just
prepend "Test", as in "TestDecisionTree". See "test/TestDecisionTree.jl"
for an example.

- Do not add the external package to the `Project.toml` file in the
  usual way. Rather, add its UUID to the `[extras]` section of
  `Project.toml` and add the package name to `test = [Test", "DecisionTree",
  ...]`.

- Add suitable lines to ["test/runtests.jl"](../test/runtests.jl) to
`include` your test file, for the purpose of testing MLJ core and all
currently supported packages, including yours. You can Test your code
by running `test MLJ` from the Julia interactive package manager. You
will need to `Pkg.dev` your local MLJ fork first. To test your code in
isolation, locally edit "test/runtest.jl" appropriately.

