# Getting Started

### [Installation instructions](https://github.com/alan-turing-institute/MLJ.jl/blob/master/README.md)

### [Glossary](glossary.md)

### Plug-and-play model evaluation

To load some data install the
[RDatasets](https://github.com/JuliaStats/RDatasets.jl) in your load
path and enter

```@repl doda
using RDatasets
iris = dataset("datasets", "iris"); # a DataFrame
```
and then split the data into input and target parts:

```@repl doda
X = iris[:, 1:4];
y = iris[:, 5];
```

In MLJ a *model* is a struct storing the hyperparameters of the learning
algorithm indicated by the struct name.  Assuming the DecisionTree
package is in your load path, we can instantiate a
DecisionTreeClassifier model like this:

```@repl doda
using MLJ
@load DecisionTreeClassifier verbosity=1
tree_model = DecisionTreeClassifier(max_depth=2)
```

*Important:* DecisionTree and most other packages implementing machine
learning algorithms for use in MLJ are not MLJ dependencies. If such a
package is not in your load path you will receive an error explaining
how to add the package to your current environment.

Once loaded, a model is evaluated with the `evaluate` method:

```@repl doda
evaluate(tree_model, X, y, 
         resampling=CV(shuffle=true), measure=cross_entropy, verbosity=0)
```

Evaluating against multiple performance measures is also possible. See
[Evaluating model performance](evaluating_model_performance.md) for details.


### Training and testing by hand

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

One can explicitly get modes by using `predict_mode` instead of `predict`:

```@repl doda
predict_mode(tree, rows=test[3:5])
```

Machines have an internal state which allows them to avoid redundant
calculations when retrained, in certain conditions - for example when
increasing the number of trees in a random forest, or the number of
epochs in a neural network. The machine building syntax also
anticaptes a more general syntax for composing multiple models, as
explained in [Learning Networks](learning_networks.md).

There is a version of `evaluate` for machines as well as models:

```@repl doda
evaluate!(tree, resampling=Holdout(fraction_train=0.5, shuffle=true),
                measure=cross_entropy,
                verbosity=0)
```
Changing a hyperparameter and re-evaluating:

```@repl doda
tree_model.max_depth = 3
evaluate!(tree, resampling=Holdout(fraction_train=0.5, shuffle=true),
          measure=cross_entropy,
          verbosity=0)
```

### Next steps

To learn a little more about what MLJ can do, take the MLJ
[tour](https://github.com/alan-turing-institute/MLJ.jl/blob/master/examples/tour/tour.ipynb),
and then return to the manual as needed. Read at least the remainder
of this page before considering serious use of MLJ.


### Prerequisites

MLJ assumes some familiarity with the `CategoricalValue` and
`CategoricalString` types from
[CategoricalArrays.jl](https://github.com/JuliaData/CategoricalArrays.jl),
used here for representing categorical data. For probabilistic
predictors, a basic acquaintance with
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl) is
also assumed.


### Data containers and scientific types

The MLJ user should acquaint themselves with some
basic assumptions about the form of data expected by MLJ, as outlined
below. 

```
machine(model::Supervised, X, y) 
machine(model::Unsupervised, X)
```

**Multivariate input.** The input `X` in the above machine
constructors can be any table, where *table* means any data type
supporting the [Tables.jl](https://github.com/JuliaData/Tables.jl)
interface.

In particular, `DataFrame`, `JuliaDB.IndexedTable` and
`TypedTables.Table` objects are supported, as are two Julia native
formats: *column tables* (named tuples of equal length vectors) and
*row tables* (vectors of named tuples sharing the same
keys).

**Univariate input.** For models which handle only univariate inputs
(`input_is_multivariate(model)=false`) `X` cannot be a table but is
expected to be some `AbstractVector` type.

**Targets.** The target `y` in the first constructor above must be an
`AbstractVector`. A multivariate target `y` will be a vector of
*tuples*. The tuples need not have uniform length, so some forms of
sequence prediction are supported. Only the element types of `y`
matter (the types of `y[j]` for each `j`). Indeed if a machine accepts
`y` as an argument it will be just as happy with `identity.(y)`.

**Element types.** The types of input and target *elements* has strict
consequences for MLJ's behaviour. 

To articulate MLJ's conventions about data representation, MLJ
distinguishes between *machine* data types on the one hand (`Float64`,
`Bool`, `String`, etc) and *scientific data types* on the other,
represented by new Julia types: `Continuous`, `Count`,
`Multiclass{N}`, `OrderedFactor{N}` and `Unknown`, with obvious
interpretations.  These types are organized in a type
[hierarchy](scitypes.png) rooted in a new abstract type `Found`.

A *scientific type* is any subtype of
`Union{Missing,Found}`. Scientific types have no instances. (They are
used behind the scenes is values for model trait functions.) Such
types appear, for example, when querying model metadata:

```julia
julia> info("DecisionTreeClassifier")[:target_scitype_union]
```

```julia
Finite
```

```julia
subtypes(Finite)
```

```julia
2-element Array{Any,1}:
 Multiclass   
 OrderedFactor
```

This means that the scitype of all elements of `DecisionTreeClassier`
target must be `Multiclass` or `OrderedFactor`.

To see how MLJ will interpret an object `x` appearing in table or
vector input `X`, or target vector `y`, call `scitype(x)`. The fallback
this function is `scitype(::Any) = Unknown`. 

```julia
julia> (scitype(42), scitype(float(π)), scitype("Julia"))
```

```julia
(Count, Continuous, Unknown)
```
    
The table below shows machine types that have scientific types
different from `Unknown`:

`T`                         |     `scitype(x)` for `x::T`
----------------------------|:--------------------------------
`AbstractFloat`             |      `Continuous`
`Integer`                   |        `Count`
`CategoricalValue`          | `Multiclass{N}` where `N = nlevels(x)`, provided `x.pool.ordered == false` 
`CategoricalString`         | `Multiclass{N}` where `N = nlevels(x)`, provided `x.pool.ordered == false`
`CategoricalValue`          | `OrderedFactor{N}` where `N = nlevels(x)`, provided `x.pool.ordered == true` 
`CategoricalString`         | `OrderedFactor{N}` where `N = nlevels(x)` provided `x.pool.ordered == true`
`Missing`                   | `Missing`

Here `nlevels(x) = length(levels(x.pool))`.

**Special note on using integers.** According to the above, integers
cannot be used to represent `Multiclass` or `OrderedFactor` data. These can be represented by an unordered or ordered `CategoricalValue`
or `CategoricalString` (automatic if they are elements of a
`CategoricalArray`).

Methods exist to coerce the scientific type of a vector or table (see
below). [Task](working_with_tasks.md) constructors also allow one to
force the data being wrapped to have the desired scientific type.

For more about scientific types and their role, see [Adding Models for
General Use](adding_models_for_general_use.md)


```@docs
coerce
```






