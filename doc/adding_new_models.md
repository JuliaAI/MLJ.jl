# Adding new models to MLJ

This guide outlines the specification for the lowest level of the MLJ
application interface. It is is guide for those adding new models by:
(i) Writing glue code for lazily loaded external packages (the main
case); and (ii) Writing code that is directly included in MLJ in the
form of an include file.

A checklist for adding models is given at the end, and a template for
adding supervised learner models from external packages is at
["src/interfaces/DecisionTree.jl"](../src/interfaces/DecisionTree.jl)


<!-- ### MLJ types -->

<!-- Every type introduced the core MLJ package should be a subtype of: -->

<!-- ``` -->
<!-- abstract type MLJType end -->
<!-- ``` -->

<!-- The Julia `show` method is informatively overloaded for this -->
<!-- type. Variable bindings declared with `@constant` "register" the -->
<!-- binding, which is reflected in the output of `show`. -->


## Models

A *model* is an object storing hyperparameters associated with some
machine learning algorithm, where "learning algorithm" is to be
broadly interpreted.  In MLJ, hyperparameters include configuration
parameters, like number of threads, which may not affect the final
learning outcome.  However, the logging level, `verbosity`, is
excluded.

The name of the Julia type associated with a model indicates the
associated algorithm (e.g., `DecisionTreeClassifier`). The ultimate
supertype of all models is:

````julia
abstract type Model <: MLJType end 
````

Associated with every concrete subtype of `Model` there must be a
`fit` method, which implements the associated algorithm to
produce a *fit-result* (see below). 

Informally, we divide some common learning algorithms into those
intended for making "predictions", called *learners* (e.g., the CART
decision tree algorithm) and those intended for "transforming" data
(based on previously seen data), called *transformers* (e.g., the PCA
feature-reduction algorithm).  Generally, only transformers may
convert data in two directions (can *inverse* transform) and only
*supervised* learners have more input variables for training than for
prediction, although the distinction might otherwise be vague. 

More formally, a model is a `Learner` if it has a `predict`
method. Learners are subdivided into `Supervised` and `Unsupervised`
types, depending on the signature of their `fit` method. Models are
of `Transformer` type if they have a `transform` method, etc. Details
follow below.

In addition to basic learning algorithms are "meta-algorithms" like
cross-validation and hyperparameter tuning, which have also
hyperparameters (e.g., number of folds for cross-validation) and so
have associated models. In place of methods like `predict` or
`transform` will be methods like `tune` or `score`. These models are
not described here.

The outcome of training a learning algorithm is here called a
*fit-result*. It is what a learner needs to predict or what a
transformer needs to transform, etc. For a linear model, for example,
this would be the coefficients and intercept.  There is no abstract
type for fit-results because these types are generally declared in
external packages. However, in MLJ the abstact supervised model type
is parametrized by the fit-result type `R`, for efficient
implementation of large ensembles of learners of uniform type.

At present a new model type should be declared as a subtype of a leaf
in the following abstract model heirachy:

````julia
abstract type Learner <: Model end
    abstract type Supervised{R} <: Learner end
	    abstract type Regressor{R} <: Supervised{R} end
		abstract type Classifier{R} <: Supervised{R} end
        abstract type Mulitilabel{R} <: Supervised{R} end
    abstract type Unsupervised <: Learner end

abstract type Transformer <: Model end 
````

## Package interfaces (glue code)

Note: Most of the following remarks also apply to built-in learning
algorithms (i.e., not defined in external packages) and presently
located in "src/builtins/". In particular "src/transforms.jl" will
contain a number of common preprocessing transformations. External
package interfaces go in "src/interfaces/".

Every package interface should live inside a submodule for namespace
hygiene (see the template at
"src/interfaces/DecisionTree.jl"). Ideally, package interfaces should
export no `struct` outside of the new model types they define, and
import only abstract types. All "structural" design should be
restricted to the MLJ core to prevent rewriting glue code when there
are design changes.

### New model type declarations

Here is an example of a concrete model type declaration:

````julia

R = Tuple{Matrix{Float64},Vector{Float64}}

mutable struct KNNRegressor{M,K} <: Regressor{R}
    K::Int          
    metric::M
    kernel::K
end

````

Models (which are mutable) should never have internally defined
constructors but should always be given an external lazy keyword
constructor of the same name that defines default values and checks
their validity, by calling an optional `clean!` method (see below).


### Supervised models

For every concrete type `ConcreteModel{R} <: Supervised{R}` a number
of "basement-level" methods are defined. These are what go into
package interfaces, together with model declerations.


#### Compulsory methods

````julia
fitresult, cache, report =  fit(learner::ConcreteModel, verbosity::Int, rows, X, y)
````

Here `fitresult::R` is the fit-result in the sense above. Any
training-related statistics, such as internal estimates of the
generalization error, and feature rankings (controlled by model
hyperparameters) should be returned in the `report` object. This is
either a `Dict{Symbol,Any}` object, or `nothing` if there is nothing
to report. So for example, `fit` might declare
`report[:feature_importances]=...`.  Reports get merged with those
generated by previous calls to `fit` at higher levels of the MLJ
interface. The value of `cache` can be `nothing` unless one is also
defining an `update` method (see below). The Julia type of `cache` is
not presently restricted.

The types of the training data `X` and `y` should be whatever is
required by the package for the training algorithm and declared in the
`fit` type signature for safety.  It is understood that `fit` only
uses `rows` for training, except in special cases requiring all available
data (e.g., to determine all possibly classes for a
categorical feature) and data leakage is not likely by doing
so. Checks not specific to the package (e.g., dimension matching
checks) should be left to higher levels of the interface to avoid code
duplication.

The method `fit` should initially call `clean!` on `learner` and issue
the returned warning indicating the changes to `learner`. The `clean!`
method has a trivial fall-back (which needs to be imported from MLJ)
but can be extended (see below, and the template). This is the only
time `fit` should alter hyperparameter values. If the package is able
to suggest better hyperparameters, as a biproduct of training, return
these in the report field.

The `verbosity` level (0 for silent) is for passing to the fit method
of the external package. The `fit` method should generally avoid doing
its own logging to avoid duplication at higher levels of the
interface.

> Presently, MLJ has a thin wrapper for fit-results called `ModelFit`
> and the output of `fit` in package interfaces is of this type. I
> suggest any such wrapping occur *outside* of package interfaces, to
> avoid rewriting them if the core design changes. For the same
> reason, I suggest that package interfaces import as little as
> possible from core.

````julia
yhat = predict(learner::ConcreteModel, fitresult, Xnew)
````

Here `Xnew` is understood to be of the same type as `X` in the `fit`
method. (So to get a prediction on a single pattern, a user may need
to suitably wrap the pattern before passing to `predict` - as a
single-row `DataFrame`, for example - and suitably unwrap
`yhat`, which must have the same type as `y` in the `fit`
method.)


#### Optional methods

**Binary classifiers.** A learner of `Classifier` type can implement a
`predict_proba` method to predict probabilities instead of labels, and
will have the same type signature as `predict` for its inputs. It
should return a single `Float64` probability per input pattern.

**Multilabel classifiers.** A learner of `Multilabel` type can also
implement a `predict_proba` method, but its return value `yhat` must
be an `Array{Float64}` object of size `(nrows, k - 1)` where `nrows`
is the number of input patterns (the number of rows of the input data
`Xnew`) and `k` is the number of labels. In addition, a method
`labels` must be implemented such that `labels(fitresult)` is a the
vector of the unique labels, ordered such that the probalibites in the
`j`th column of `yhat` correspond to the `j`th label (`j < k`).

````julia
message::String = clean!(learner::Supervised)
````

Checks and corrects for invalid fields (hyperparameters), returning a
warning `messsage`. Should only throw an exception as a last resort.

````julia
    fitresult, cache, report = 
	    update(learner::ConcreteModel, verbosity, old_fitresult, old_cache, X, y; kwargs...) 
````

A package interface author may overload an `update` method to enable
a call to retrain a model at higher levels of the API (on the same
training data) to avoid repeating computations unecessarily. The main
use-case is composite models, where, depending on the new
hyperparemeter values, it may unecessary to retrain each individual
component of the model. A second important use case is iterative
models, where calls to increase the number of iterations only restarts
the iterative procedure if other hyperparameters have also changed.

**Iterative learners.** An iterative model should import the
`iteration_parameter` method and overload it to indicate the name of the model's
field indicating the number of iterations:

````julia
iteration_parameter(model::ConcreteModel) = <symbol of iteration hyperparameter>
````

In the event that the argument `fitresult` (returned by a preceding
call to `fit`) is not sufficient for performing an update, the author
can arrange for `fit` to output in its `cache` return value any
additional information required, as this is also passed as an argument
to the `update` method.

Additionally, one can use `update` to implement package-specific
post-training functionality, such as pruning an existing decision
tree. The caller of `update` requests such functionality by providing
appropriate `kwargs`, such as `prune_only=true`. If any `kwargs` are
provided, then it is to be understood that no retraining of the model
is to be carried out.


##  Checklist for new adding models 

At present the following checklist is just for supervised learner models in
lazily loaded external packages.

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
code, in a file with an identical name, but placed in "test/" (as in
["test/DecisionTree.jl"](../test/DecisionTree.jl)). This
code should be wrapped in a module to prevent namespace conflicts with
other test code. For a module name, just prepend "Test", as in
"TestDecisionTree". See "test/DecisionTree.jl" for an example. 

- Add a line to ["test/runtests.jl"](../test/runtests.jl) to
`include` your test file, for the purpose of testing MLJ core and all
currently supported packages, including yours. You can Test your code
by running `test MLJ` from the Julia interactive package manager. You
will need to `Pkg.dev` your local MLJ fork first. To test your code in
isolation, locally edit "test/runtest.jl" appropriately.


