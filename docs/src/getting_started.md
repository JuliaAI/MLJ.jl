# Getting Started

For an outline of MLJ's **goals** and **features**, see
[About MLJ](@ref).

This page introduces some MLJ basics, assuming some familiarity with
machine learning. For a complete list of other MLJ learning resources,
see [Learning MLJ](@ref).

This section introduces only the most basic MLJ operations and
concepts. It assumes MLJ has been successfully installed. See
[Installation](@ref) if this is not the case. 


```@setup doda
import Random.seed!
using MLJ
using InteractiveUtils
MLJ.color_off()
seed!(1234)
```

## Choosing and evaluating a model

The following code loads Fisher's famous iris data set as a named tuple of
column vectors:

```@repl doda
using MLJ
iris = load_iris();
selectrows(iris, 1:3)  |> pretty
schema(iris)
```

Because this data format is compatible with
[Tables.jl](https://tables.juliadata.org/stable/), many MLJ methods
(such as `selectrows`, `pretty` and `schema` used above) as well as
many MLJ models can work with it. However, as most new users are
already familiar with the access methods particular to
[DataFrames](https://dataframes.juliadata.org/stable/) (also
compatible with Tables.jl) we'll put our data into that format here:

```@example doda
import DataFrames
iris = DataFrames.DataFrame(iris);
nothing # hide
```

Next, let's split the data "horizontally" into input and target parts,
and specify an RNG seed, to force observations to be shuffled:

```@repl doda
y, X = unpack(iris, ==(:target); rng=123);
first(X, 3) |> pretty
```

This call to `unpack` splits off any column with name `==`
to `:target` into something called `y`, and all the remaining columns
into `X`.

To list *all* models available in MLJ's [model
registry](@ref model_search) do `models()`. Listing the models
compatible with the present data:

```@repl doda
models(matching(X,y))
```

In MLJ a *model* is a struct storing the hyperparameters of the
learning algorithm indicated by the struct name (and nothing
else). For common problems matching data to models, see [Model
Search](@ref model_search) and [Preparing Data](@ref).

To see the documentation for `DecisionTreeClassifier` (without
loading its defining code) do

```julia
doc("DecisionTreeClassifier", pkg="DecisionTree")
```

Assuming the MLJDecisionTreeInterface.jl package is in your load path
(see [Installation](@ref)) we can use `@load` to import the
`DecisionTreeClassifier` model type, which we will bind to `Tree`:

```@repl doda
Tree = @load DecisionTreeClassifier pkg=DecisionTree
```

(In this case, we need to specify `pkg=...` because multiple packages
provide a model type with the name `DecisionTreeClassifier`.) Now we can
instantiate a model with default hyperparameters:

```@repl doda
tree = Tree()
```

*Important:* DecisionTree.jl and most other packages implementing
machine learning algorithms for use in MLJ are not MLJ
dependencies. If such a package is not in your load path you will
receive an error explaining how to add the package to your current
environment. Alternatively, you can use the interactive macro
`@iload`. For more on importing model types, see [Loading Model
Code](@ref).

Once instantiated, a model's performance can be evaluated with the
`evaluate` method. Our classifier is a *probabilistic* predictor (check
`prediction_type(tree) == :probabilistic`) which means we can specify
a probabilistic measure (metric) like `log_loss`, as well
deterministic measures like `accuracy` (which are applied after
computing the mode of each prediction):

```@repl doda
evaluate(tree, X, y,
         resampling=CV(shuffle=true),
                 measures=[log_loss, accuracy],
                 verbosity=0)
```

Under the hood, `evaluate` calls lower level functions `predict` or
`predict_mode` according to the type of measure, as shown in the
output. We shall call these operations directly below.

For more on performance evaluation, see [Evaluating Model
Performance](evaluating_model_performance.md) for details.


## A preview of data type specification in MLJ

The target `y` above is a categorical vector, which is appropriate
because our model is a decision tree *classifier*:

```@repl doda
typeof(y)
```

However, MLJ models do not prescribe the machine types for
the data they operate on. Rather, they specify a *scientific type*,
which refers to the way data is to be *interpreted*, as opposed to how
it is *encoded*:

```@repl doda
target_scitype(tree)
```

Here `Finite` is an example of a "scalar" scientific type with two
subtypes:

```@repl doda
subtypes(Finite)
```

We use the `scitype` function to check how MLJ is going to interpret
given data. Our choice of encoding for `y` works for
`DecisionTreeClassifier`, because we have:

```@repl doda
scitype(y)
```

and `Multiclass{3} <: Finite`. If we would encode with integers
instead, we obtain:

```@repl doda
yint = int.(y);
scitype(yint)
```

and using `yint` in place of `y` in classification problems will
fail. See also [Working with Categorical Data](@ref).

For more on scientific types, see [Data containers and scientific
types](@ref) below.


## Fit and predict

To illustrate MLJ's fit and predict interface, let's perform our
performance evaluations by hand, but using a simple holdout set,
instead of cross-validation.

Wrapping the model in data creates a *machine* which will store
training outcomes:

```@repl doda
mach = machine(tree, X, y)
```

Training and testing on a hold-out set:

```@repl doda
train, test = partition(eachindex(y), 0.7); # 70:30 split
fit!(mach, rows=train);
yhat = predict(mach, X[test,:]);
yhat[3:5]
log_loss(yhat, y[test]) |> mean
```

Note that `log_loss` and `cross_entropy` are aliases for `LogLoss()`
(which can be passed an optional keyword parameter, as in
`LogLoss(tol=0.001)`). For a list of all losses and scores, and their
aliases, run `measures()`.

Notice that `yhat` is a vector of `Distribution` objects, because
DecisionTreeClassifier makes probabilistic predictions. The methods
of the [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
package can be applied to such distributions:

```@repl doda
broadcast(pdf, yhat[3:5], "virginica") # predicted probabilities of virginica
broadcast(pdf, yhat, y[test])[3:5] # predicted probability of observed class
mode.(yhat[3:5])
```

Or, one can explicitly get modes by using `predict_mode` instead of
`predict`:

```@repl doda
predict_mode(mach, X[test[3:5],:])
```

Finally, we note that `pdf()` is overloaded to allow the retrieval of
probabilities for all levels at once:

```@repl doda
L = levels(y)
pdf(yhat[3:5], L)
```

Unsupervised models have a `transform` method instead of `predict`,
and may optionally implement an `inverse_transform` method:

```@repl doda
v = Float64[1, 2, 3, 4]
stand = Standardizer() # this type is built-in
mach2 = machine(stand, v)
fit!(mach2)
w = transform(mach2, v)
inverse_transform(mach2, w)
```

[Machines](machines.md) have an internal state which allows them to
avoid redundant calculations when retrained, in certain conditions -
for example when increasing the number of trees in a random forest, or
the number of epochs in a neural network. The machine-building syntax
also anticipates a more general syntax for composing multiple models,
an advanced feature explained in [Learning Networks](@ref).

There is a version of `evaluate` for machines as well as models. This
time we'll use a simple holdout strategy as above. (An exclamation
point is added to the method name because machines are generally
mutated when trained.)

```@repl doda
evaluate!(mach, resampling=Holdout(fraction_train=0.7),
                measures=[log_loss, accuracy],
                verbosity=0)
```
Changing a hyperparameter and re-evaluating:

```@repl doda
tree.max_depth = 3
evaluate!(mach, resampling=Holdout(fraction_train=0.7),
          measures=[log_loss, accuracy],
          verbosity=0)
```

## Next steps

To learn a little more about what MLJ can do, browse [Common MLJ
Workflows](common_mlj_workflows.md) or [Data Science Tutorials in
Julia](https://alan-turing-institute.github.io/DataScienceTutorials.jl/)
or try the [JuliaCon2020
Workshop](https://github.com/ablaom/MachineLearningInJulia2020) on MLJ
(recorded
[here](https://www.youtube.com/watch?time_continue=27&v=qSWbCn170HU&feature=emb_title))
returning to the manual as needed.

*Read at least the remainder of this page before considering serious
use of MLJ.*


## Data containers and scientific types

The MLJ user should acquaint themselves with some basic assumptions
about the form of data expected by MLJ, as outlined below. The basic
`machine` constructors look like this (see also [Constructing
machines](@ref)):

```
machine(model::Unsupervised, X)
machine(model::Supervised, X, y)
```

Each supervised model in MLJ declares the permitted *scientific type*
of the inputs `X` and targets `y` that can be bound to it in the first
constructor above, rather than specifying specific machine types (such
as `Array{Float32, 2}`). Similar remarks apply to the input `X` of an
unsupervised model.

Scientific types are julia types defined in the package
[ScientificTypesBase.jl](https://github.com/JuliaAI/ScientificTypesBase.jl);
the package
[ScientificTypes.jl](https://JuliaAI.github.io/ScientificTypes.jl/dev/)
implements the particular convention used in the MLJ universe for
assigning a specific scientific type (interpretation) to each julia
object (see the `scitype` examples below).

The basic "scalar" scientific types are `Continuous`, `Multiclass{N}`,
`OrderedFactor{N}`, `Count` and `Textual`. `Missing` and `Nothing` are
also considered scientific types. Be sure you read [Scalar scientific
types](@ref) below to guarantee your scalar data is interpreted
correctly. Tools exist to coerce the data to have the appropriate
scientific type; see
[ScientificTypes.jl](https://JuliaAI.github.io/ScientificTypes.jl/dev/)
or run `?coerce` for details.

Additionally, most data containers - such as tuples, vectors, matrices
and tables - have a scientific type parameterized by scitype of the
elements they contain.

![](img/scitypes_small.png)

*Figure 1. Part of the scientific type hierarchy in*
[ScientificTypesBase.jl](https://JuliaAI.github.io/ScientificTypes.jl/dev/).

```@repl doda
scitype(4.6)
scitype(42)
x1 = coerce(["yes", "no", "yes", "maybe"], Multiclass);
scitype(x1)
X = (x1=x1, x2=rand(4), x3=rand(4))  # a "column table"
scitype(X)
```

### Two-dimensional data

Generally, two-dimensional data in MLJ is expected to be *tabular*.
All data containers compatible with the
[Tables.jl](https://github.com/JuliaData/Tables.jl) interface (which
includes all source formats listed
[here](https://github.com/JuliaData/Tables.jl/blob/master/INTEGRATIONS.md))
have the scientific type `Table{K}`, where `K` depends on the
scientific types of the columns, which can be individually inspected
using `schema`:

```@repl doda
schema(X)
```

#### Matrix data

MLJ models expecting a table do not generally accept a matrix
instead. However, a matrix can be wrapped as a table, using
[`MLJ.table`](@ref):

```julia
matrix_table = MLJ.table(rand(2,3))
schema(matrix_table)
```

```
┌─────────┬─────────┬────────────┐
│ _.names │ _.types │ _.scitypes │
├─────────┼─────────┼────────────┤
│ x1      │ Float64 │ Continuous │
│ x2      │ Float64 │ Continuous │
│ x3      │ Float64 │ Continuous │
└─────────┴─────────┴────────────┘
_.nrows = 2

```

The matrix is *not* copied, only wrapped.  To manifest a table as a
matrix, use [`MLJ.matrix`](@ref).

### Observations correspond to rows, not columns

When supplying models with matrices, or wrapping them in tables, each
*row* should correspond to a different observation. That is, the
matrix should be `n x p`, where `n` is the number of observations and
`p` the number of features. However, *some models may perform better* if
supplied the *adjoint* of a `p x n` matrix instead, and observation
resampling is always more efficient in this case.


### Inputs

Since an MLJ model only specifies the scientific type of data, if that
type is `Table` - which is the case for the majority of MLJ models -
then any [Tables.jl](https://github.com/JuliaData/Tables.jl) format is
permitted.

Specifically, the requirement for an arbitrary model's input is `scitype(X)
<: input_scitype(model)`.

### Targets

The target `y` expected by MLJ models is generally an
`AbstractVector`. A multivariate target `y` will generally be a table.

Specifically, the type requirement for a model target is `scitype(y) <:
target_scitype(model)`.


### Querying a model for acceptable data types

Given a model instance, one can inspect the admissible scientific
types of its input and target, and without loading the code defining
the model;

```@setup doda
tree = @load DecisionTreeClassifier pkg=DecisionTree
```

```@repl doda
i = info("DecisionTreeClassifier", pkg="DecisionTree")
i.input_scitype
i.target_scitype
```

This output indicates that any table with `Continuous`, `Count` or
`OrderedFactor` columns is acceptable as the input `X`, and that any
vector with element scitype `<: Finite` is acceptable as the target
`y`.

For more on matching models to data, see [Model Search](@ref model_search).

### Scalar scientific types

Models in MLJ will always apply the `MLJ` convention described in
[ScientificTypes.jl](https://JuliaAI.github.io/ScientificTypes.jl/dev/)
to decide how to interpret the elements of your container types. Here
are the key features of that convention:

- Any `AbstractFloat` is interpreted as `Continuous`.

- Any `Integer` is interpreted as `Count`.

- Any `CategoricalValue` `x`, is interpreted as `Multiclass` or
  `OrderedFactor`, depending on the value of `isordered(x)`.

- `String`s and `Char`s are *not* interpreted as `Multiclass` or
  `OrderedFactor` (they have scitypes `Textual` and `Unknown`
  respectively).

- In particular, *integers* (including `Bool`s) *cannot be used to
  represent categorical data.* Use the preceding `coerce` operations
  to coerce to a `Finite` scitype.

- The scientific types of `nothing` and `missing` are `Nothing` and
  `Missing`, native types we also regard as scientific.

Use `coerce(v, OrderedFactor)` or `coerce(v, Multiclass)` to coerce a
vector `v` of integers, strings or characters to a vector with an
appropriate `Finite` (categorical) scitype.  See [Working with
Categorical Data](@ref).

For more on scitype coercion of arrays and tables, see [`coerce`](@ref),
[`autotype`](@ref) and [`unpack`](@ref) below and the examples at
[ScientificTypes.jl](https://JuliaAI.github.io/ScientificTypes.jl/dev/).



```@docs
scitype
coerce
autotype
```
