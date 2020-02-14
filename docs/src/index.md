### [Installation](https://github.com/alan-turing-institute/MLJ.jl/blob/master/README.md) | [Cheatsheet](mlj_cheatsheet.md) | [Workflows](common_mlj_workflows.md)


# Getting Started


```@setup doda
import Random.seed!
using MLJ
using InteractiveUtils
MLJ.color_off()
seed!(1234)
```

## Choosing and evaluating a model

To load some demonstration data, add
[RDatasets](https://github.com/JuliaStats/RDatasets.jl) to your load
path and enter
```@repl doda
using RDatasets
iris = dataset("datasets", "iris"); # a DataFrame
```
and then split the data into input and target parts:

```@repl doda
using MLJ
y, X = unpack(iris, ==(:Species), colname -> true);
first(X, 3) |> pretty
```

To list all models available in MLJ's [model
registry](model_search.md):

```@repl doda
models()
```

In MLJ a *model* is a struct storing the hyperparameters of the
learning algorithm indicated by the struct name.  

Assuming the DecisionTree.jl package is in your load path, we can use
`@load` to load the code defining the `DecisionTreeClassifier` model
type. This macro also returns an instance, with default
hyperparameters.

Drop the `verbosity=1` declaration for silent loading:

```@repl doda
tree_model = @load DecisionTreeClassifier verbosity=1
```

*Important:* DecisionTree.jl and most other packages implementing machine
learning algorithms for use in MLJ are not MLJ dependencies. If such a
package is not in your load path you will receive an error explaining
how to add the package to your current environment.

Once loaded, a model can be evaluated with the `evaluate` method:

```@repl doda
evaluate(tree_model, X, y,
         resampling=CV(shuffle=true), measure=cross_entropy, verbosity=0)
```

Evaluating against multiple performance measures is also possible. See
[Evaluating Model Performance](evaluating_model_performance.md) for details.


## A preview of data type specification in MLJ

The target `y` above is a categorical vector, which is appropriate
because our model is a decision tree *classifier*:

```@repl doda
typeof(y)
```

However, MLJ models do not actually prescribe the machine types for
the data they operate on. Rather, they specify a *scientific type*,
which refers to the way data is to be *interpreted*, as opposed to how
it is *encoded*:

```julia
julia> info("DecisionTreeClassifier").target_scitype
AbstractArray{<:Finite, 1}
```

Here `Finite` is an example of a "scalar" scientific type with two
subtypes:

```@repl doda
subtypes(Finite)
```

We use the `scitype` function to check how MLJ is going to interpret
given data. Our choice of encoding for `y` works for
`DecisionTreeClassfier`, because we have:

```@repl doda
scitype(y)
```

and `Multiclass{3} <: Finite`. If we would encode with integers
instead, we obtain:

```@repl doda
yint = Int.(y.refs);
scitype(yint)
```

and using `yint` in place of `y` in classification problems will fail.

For more on scientific types, see [Data containers and scientific
types](@ref) below.


## Fit and predict

To illustrate MLJ's fit and predict interface, let's perform our
performance evaluations by hand, but using a simple holdout set,
instead of cross-validation.

Wrapping the model in data creates a *machine* which will store
training outcomes:

```@repl doda
tree = machine(tree_model, X, y)
```

Training and testing on a hold-out set:

```@repl doda
train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
fit!(tree, rows=train);
yhat = predict(tree, X[test,:]);
yhat[3:5]
cross_entropy(yhat, y[test]) |> mean
```

Notice that `yhat` is a vector of `Distribution` objects (because
DecisionTreeClassifier makes probabilistic predictions). The methods
of the [Distributions](https://github.com/JuliaStats/Distributions.jl)
package can be applied to such distributions:

```@repl doda
broadcast(pdf, yhat[3:5], "virginica") # predicted probabilities of virginica
mode.(yhat[3:5])
```

Or, one can explicitly get modes by using `predict_mode` instead of
`predict`:

```@repl doda
predict_mode(tree, rows=test[3:5])
```

Unsupervised models have a `transform` method instead of `predict`,
and may optionally implement an `inverse_transform` method:

```@repl doda
v = [1, 2, 3, 4]
stand_model = UnivariateStandardizer()
stand = machine(stand_model, v)
fit!(stand)
w = transform(stand, v)
inverse_transform(stand, w)
```

[Machines](machines.md) have an internal state which allows them to
avoid redundant calculations when retrained, in certain conditions -
for example when increasing the number of trees in a random forest, or
the number of epochs in a neural network. The machine building syntax
also anticipates a more general syntax for composing multiple models,
as explained in [Composing Models](composing_models.md).

There is a version of `evaluate` for machines as well as models. This
time we'll add a second performance measure. (An exclamation point is
added to the method name because machines are generally mutated when
trained):

```@repl doda
evaluate!(tree, resampling=Holdout(fraction_train=0.7, shuffle=true),
                measures=[cross_entropy, BrierScore()],
                verbosity=0)
```
Changing a hyperparameter and re-evaluating:

```@repl doda
tree_model.max_depth = 3
evaluate!(tree, resampling=Holdout(fraction_train=0.7, shuffle=true),
          measures=[cross_entropy, BrierScore()],
          verbosity=0)
```

## Next steps

To learn a little more about what MLJ can do, browse [Common MLJ
Workflows](common_mlj_workflows.md) or MLJ's
[tutorials](https://alan-turing-institute.github.io/MLJTutorials/),
returning to the manual as needed. *Read at least the remainder of
this page before considering serious use of MLJ.*


## Prerequisites

MLJ assumes some familiarity with
[CategoricalArrays.jl](https://github.com/JuliaData/CategoricalArrays.jl),
used here for representing arrays of categorical data. For
probabilistic predictors, a basic acquaintance with
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl) is
also assumed.


## Data containers and scientific types

The MLJ user should acquaint themselves with some
basic assumptions about the form of data expected by MLJ, as outlined
below.

```
machine(model::Supervised, X, y)
machine(model::Unsupervised, X)
```

Each supervised model in MLJ declares the permitted *scientific type*
of the inputs `X` and targets `y` that can be bound to it in the first
constructor above, rather than specifying specific machine types (such
as `Array{Float32, 2}`). Similar remarks apply to the input `X` of an
unsupervised model.

Scientific types are julia types defined in the package
[ScientificTypes.jl](https://github.com/alan-turing-institute/ScientificTypes.jl);
the package
[MLJScientificTypes](https://github.com/alan-turing-institute/MLJScientificTypes.jl)
implements the particular convention used in the MLJ universe for
assigning a specific scientific type (interpretation) to each julia
object (see the `scitype` examples below).

The basic "scalar" scientific types are `Continuous`, `Multiclass{N}`,
`OrderedFactor{N}` and `Count`. Be sure you read [Container element
types](@ref) below to be guarantee your scalar data is interpreted
correctly. Tools exist to coerce the data to have the appropriate
scientfic type; see
[MLJScientificTypes.jl](https://github.com/alan-turing-institute/MLJScientificTypes.jl)
or run `?coerce` for details.

Additionally, most data containers - such as tuples,
vectors, matrices and tables - have a scientific type.


![](img/scitypes.png)

*Figure 1. Part of the scientific type hierarchy in* [ScientificTypes.jl](https://github.com/alan-turing-institute/ScientificTypes.jl).

```@repl doda
scitype(4.6)
scitype(42)
x1 = categorical(["yes", "no", "yes", "maybe"]);
scitype(x1)
X = (x1=x1, x2=rand(4), x3=rand(4))  # a "column table"
scitype(X)
```

### Tabular data

All data containers compatible with the
[Tables.jl](https://github.com/JuliaData/Tables.jl) interface (which
includes all source formats listed
[here](https://github.com/queryverse/IterableTables.jl)) have the
scientific type `Table{K}`, where `K` depends on the scientific types
of the columns, which can be individually inspected using `schema`:

```@repl doda
schema(X)
```

### Inputs

Since an MLJ model only specifies the scientific type of data, if that
type is `Table` - which is the case for the majority of MLJ models -
then any [Tables.jl](https://github.com/JuliaData/Tables.jl) format is
permitted. However, the Tables.jl API excludes matrices. If `Xmatrix`
is a matrix, convert it to a column table using `X =
MLJ.table(Xmatrix)`.

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
the model:

```@setup doda
tree = @load DecisionTreeClassifier
```

```@repl doda
i = info("DecisionTreeClassifier")
i.input_scitype
i.target_scitype
```

### Container element types

Models in MLJ will always apply the `MLJ` convention described in
[MLJScientificTypes.jl](https://github.com/alan-turing-institute/MLJScientificTypes.jl)
to decide how to interpret the elements of your container types. Here
are the key features of that convention:

- Any `AbstractFloat` is interpreted as `Continuous`.

- Any `Integer` is interpreted as `Count`.

- Any `CategoricalValue` or `CategoricalString`, `x`, is interpreted
  as `Multiclass` or `OrderedFactor`, depending on the value of
  `x.pool.ordered`.

- `String`s and `Char`s are *not* interpreted as `Multiclass` or
  `OrderedFactor (they have scitypes `Textual` and `Unknown`
  respectively). 
  
- In particular, *integers* (including `Bool`s) *cannot be used to
  represent categorical data.* Use the preceding `coerce` operations
  to coerce to a `Finite` scitype. 

Use `coerce(v, OrderedFactor)` or `coerce(v, Multiclass)` to coerce a
vector `v` of integers, strings or characters to a vector with an
appropriate `Finite` (categorical) scitype. For more on scitype
coercion of arrays and tables, see `coerce`, `autotype` and `unpack`
below.

To designate an intrinsic "true" class for binary data (for purposes
of applying MLJ measures, such as `truepositive`), data should be
represented by an ordered `CategoricalValue` or
`CategoricalString`. This data will have scitype `OrderedFactor{2}`
and the "true" class is understood to be the *second* class in the
ordering.

```@docs
coerce
autotype
unpack
```
