# Adding new models to MLJ

> At present, more in the nature of a proposal

**Warning:** Not presently in synch with code

This guide outlines the specification for the lowest level of the MLJ
application interface. It is is guide for those adding new models by:
(i) writing glue code for lazily loaded external packages (the main
case); and (ii) writing code that is directly included in MLJ in the
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
parameters, like the number of threads, which may not affect the final
learning outcome. (However,the logging level, `verbosity`, is
excluded). 

The name of the Julia type associated with a model indicates the
associated algorithm (e.g., `DecisionTreeClassifier`). The ultimate
supertype of all models is:

````julia
abstract type Model <: MLJType end 
````

We use the words "learner" and "transformer" both informally to refer
to algorithms, and to refer to associated models in the above sense.

<!-- Informally, we divide basic learning algorithms into those intended -->
<!-- for making "predictions", called *learners* (e.g., the CART decision -->
<!-- tree algorithm) and those intended for "transforming" data (based on -->
<!-- previously seen data), called *transformers* (e.g., the PCA -->
<!-- feature-reduction algorithm).  Generally, only transformers convert -->
<!-- data in two directions (can *inverse* transform) and only supervised -->
<!-- learners have more input variables for training than for prediction -->
<!-- (but the distinction might otherwise be vague). We use the same words, -->
<!-- *learner* and *transformer*, for the *models* associated with these -->
<!-- algorithms. -->

<!-- In addition to basic learning algorithms are "meta-algorithms" like -->
<!-- cross-validation and hyperparameter tuning, which have hyperparameters -->
<!-- like any other learning algorithm (e.g., number of folds for -->
<!-- cross-validation) and so have associated models. In place of methods -->
<!-- (functions) like "predict" or "transform" will be methods like "tune" -->
<!-- or "score". -->

By a *fit-result* we shall mean an object storing the "weights" or
"paramaters" learned by an algorithm using the hyperparameters
specified by a model (e.g., what a learner needs to predict or what a
transformer needs to transform). There is no abstract type for
fit-results because these types are generally declared in external
packages. However, in MLJ the abstact supervised model type is
parametrized by the fit-result type `R`, for efficient
implementation of large ensembles of learners of uniform type.

At present a new model should be declared as a subtype of a leaf in
the following abstract model heirachy:


````julia
abstract type Learner <: Model end
    abstract type Supervised{R} <: Learner end
	    abstract type Regressor{R} <: Supervised{R} end
		abstract type Classifier{R} <: Supervised{R} end

    abstract type Unsupervised <: Learner end

abstract type Transformer <: Model end 
````

> TODO: Update heirachy for iterative models. Perhaps switch to trait
> abstractions.

> Later we may introduce other types for composite learning networks
> (composite types), resampling and tuning

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

mutable struct KNNRegressor <: Regressor{R}
    K::Int           # number of local target values averaged
    metric::Function
    kernel::Function # each target value is weighted by `kernel(distance^2)`
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
fitresult, cache, report =  fit(learner::ConcreteModel, verbosity::Int, X, y)
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
defining an `update!` method (see below). The Julia type of `cache` is not
presently restricted.

The types of the training data `X` and `y` should be whatever is
required by the package for the training algorithm and declared in the
`fit` type signature for safety.  Checks not specific to the package
(e.g., dimension matching checks) should be left to higher levels of
the interface to avoid code duplication.

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
prediction = predict(learner::ConcreteModel, fitresult, Xnew)
````

Here `Xnew` is understood to be of the same type as `X` in the `fit` method. (So to
get a prediction on a single pattern, a user may need to suitably wrap
the pattern before passing to `predict` - as a single-row `DataFrame`,
for example - and suitably unwrap `prediction`, which must have the
same type as `y` in the `fit` method.)

#### Optional methods

A learner of `Classifier` type can implement a `predict_proba` method
to predict probabilities instead of labels, and will have the same
type signature as `predict` for its inputs. 

> I guess the `fit` method for a multi-class classifier will need to
> declare an order for the classes in its `fit_result` method, to be
> available to `predict` which calls `predict_proba` first.

````julia
message::String = clean!(learner::Supervised)
````

Checks and corrects for invalid fields (hyperparameters), returning a
warning `messsage`. Should only throw an exception as a last resort.

````julia
    report = update!(learner::ConcreteModel, verbosity, fitresult, cache, X, y; kwargs...) 
````

A package interface author may overload an `update!` method to enable
a call to retrain a model at higher levels of the API (on the same
training data) to avoid repeating computations unecessarily. The main
use-case is composite models, where, depending on the new
hyperparemeter values, it may unecessary to retrain each individual
component of the model. (Another use case is iterative models, where
calls to increase the number of iterations should only restart the
iterative procedure if other hyperparameters have also changed. In
this case however, this is accomplished automatically by an `update!`
fallback; see Iterative Supervised Learners below.)

> TODO: write section for iterative learners.

In the event that the argument `fitresult` (returned by a preceding
call to `fit`) is not sufficient for performing an update, the author
can arrange for `fit` to output in its `cache` return value any
additional information required, as this is also passed as an argument
to the `update!` method.

Additionally, one can use `update!` to implement package-specific
post-training functionality, such as pruning an existing decision
tree. The caller of `update!` requests such functionality by providing
appropriate `kwargs`, such as `prune_only=true`. If any `kwargs` are
provided, then it is to be understood that no retraining of the model
is carried out and `fitresult` and `cache` are not mutated. 


##  Checklist for new adding models 

At present the following checklist is just for supervised learner models in
lazily loaded external packages.

> This checklist does not apply yet. It supposes we have made certain
> organizational changes to MLJ not yet discussed.

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


