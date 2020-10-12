# Getting Started

For an outline of MLJ's **goals** and **features**, see the
[Introduction](@ref).

This section introduces the most basic MLJ operations and concepts. It
assumes MJL has been successfully installed. See [Installation](@ref)
if this is not the case.


```@setup doda
import Random.seed!
using MLJ
using InteractiveUtils
MLJ.color_off()
seed!(1234)
```

## Choosing and evaluating a model

To load some demonstration data, add
[RDatasets.jl](https://github.com/JuliaStats/RDatasets.jl) to your load
path and enter

```@repl doda
import RDatasets
iris = RDatasets.dataset("datasets", "iris"); # a DataFrame
```

and then split the data into input and target parts:

```@repl doda
using MLJ
y, X = unpack(iris, ==(:Species), colname -> true);
first(X, 3) |> pretty
```

To list *all* models available in MLJ's [model
registry](model_search.md) do `models()`. Listing the models
compatible with the present data:

```@repl doda
models(matching(X,y))
```

In MLJ a *model* is a struct storing the hyperparameters of the
learning algorithm indicated by the struct name (and nothing else).

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

Once loaded, a model's performance can be evaluated with the
`evaluate` method:

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
broadcast(pdf, yhat, y[test])[3:5] # predicted probability of observed class
mode.(yhat[3:5])
```

Or, one can explicitly get modes by using `predict_mode` instead of
`predict`:

```@repl doda
predict_mode(tree, X[test[3:5],:])
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
`machine` constructions look like this (see also [Constructing
machines](@ref)):

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
[MLJScientificTypes.jl](https://alan-turing-institute.github.io/MLJScientificTypes.jl/dev/)
implements the particular convention used in the MLJ universe for
assigning a specific scientific type (interpretation) to each julia
object (see the `scitype` examples below).

The basic "scalar" scientific types are `Continuous`, `Multiclass{N}`,
`OrderedFactor{N}` and `Count`. Be sure you read [Scalar
scientific types](@ref) below to be guarantee your scalar data is interpreted
correctly. Tools exist to coerce the data to have the appropriate
scientfic type; see
[MLJScientificTypes.jl](https://alan-turing-institute.github.io/MLJScientificTypes.jl/dev/)
or run `?coerce` for details.

Additionally, most data containers - such as tuples,
vectors, matrices and tables - have a scientific type.


![](img/scitypes.png)

*Figure 1. Part of the scientific type hierarchy in* [ScientificTypes.jl](https://alan-turing-institute.github.io/MLJScientificTypes.jl/dev/).

```@repl doda
scitype(4.6)
scitype(42)
x1 = categorical(["yes", "no", "yes", "maybe"]);
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
matrix, use [`MLJ.matrix`](@ref)`. 


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
tree = @load DecisionTreeClassifier
```

```@repl doda
i = info("DecisionTreeClassifier")
i.input_scitype
i.target_scitype
```

But see also [Model Search](@ref). 

### Scalar scientific types

Models in MLJ will always apply the `MLJ` convention described in
[MLJScientificTypes.jl](https://alan-turing-institute.github.io/MLJScientificTypes.jl/dev/)
to decide how to interpret the elements of your container types. Here
are the key features of that convention:

- Any `AbstractFloat` is interpreted as `Continuous`.

- Any `Integer` is interpreted as `Count`.

- Any `CategoricalValue` or `CategoricalString`, `x`, is interpreted
  as `Multiclass` or `OrderedFactor`, depending on the value of
  `x.pool.ordered`.

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
appropriate `Finite` (categorical) scitype.  See [Working with Categorical Data](@ref)).

For more on scitype coercion of arrays and tables, see [`coerce`](@ref),
[`autotype`](@ref) and [`unpack`](@ref) below and the examples at
[MLJScientificTypes.jl](https://alan-turing-institute.github.io/MLJScientificTypes.jl/dev/).



```@docs
scitype
coerce
autotype
unpack
```
