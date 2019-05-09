# Getting Started

### [Installation instructions](https://github.com/alan-turing-institute/MLJ.jl/blob/master/README.md)


### Basic supervised training and testing


```julia
julia> using MLJ
julia> using RDatasets
julia> iris = dataset("datasets", "iris"); # a DataFrame
```

In MLJ one can either wrap data for supervised learning in a formal
*task* (see [Working with Tasks](working_with_tasks.md)), or work
directly with the data, split into its input and target parts:


```julia
julia> const X = iris[:, 1:4];
julia> const y = iris[:, 5];
```

A *model* is a container for hyperparameters. Assuming the
DecisionTree package is in your installation load path, we can
instantiate a DecisionTreeClassifier model like this:

```julia
julia> @load DecisionTreeClassifier
import MLJModels ✔
import DecisionTree ✔
import MLJModels.DecisionTree_.DecisionTreeClassifier ✔

julia> tree_model = DecisionTreeClassifier(target_type=String, max_depth=2)
DecisionTreeClassifier(target_type = String,
                       pruning_purity = 1.0,
                       max_depth = 2,
                       min_samples_leaf = 1,
                       min_samples_split = 2,
                       min_purity_increase = 0.0,
                       n_subfeatures = 0.0,
                       display_depth = 5,
                       post_prune = false,
                       merge_purity_threshold = 0.9,) @ 1…72
```

Wrapping the model in data creates a *machine* which will store training outcomes:

```julia
julia> tree = machine(tree_model, X, y)
Machine @ 5…78
```

Training and testing on a hold-out set:

```julia
julia> train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
julia> fit!(tree, rows=train)
julia> yhat = predict(tree, X[test,:]);
julia> misclassification_rate(yhat, y[test]);

┌ Info: Training Machine{DecisionTreeClassifier{S…} @ 1…36.
└ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:68
0.08888888888888889
```

Or, in one line:

```julia
julia> evaluate!(tree, resampling=Holdout(fraction_train=0.7, shuffle=true), measure=misclassification_rate)
0.08888888888888889
```

Changing a hyperparameter and re-evaluating:

```julia
julia> tree_model.max_depth = 3
julia> evaluate!(tree, resampling=Holdout(fraction_train=0.5, shuffle=true), measure=misclassification_rate)
0.06666666666666667
```

### Next steps

To learn a little more about what MLJ can do, take the MLJ
[tour](https://github.com/alan-turing-institute/MLJ.jl/blob/master/docs/src/tour.ipynb),
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

> At present our API is more restrictive; see this
> [issue](https://github.com/JuliaData/Tables.jl/issues/74) with
> Tables.jl. If your Tables.jl compatible format is not working in
> MLJ, please post an issue.

In particular, `DataFrame`, `JuliaDB.IndexedTable` and
`TypedTables.Table` objects are supported, as are two Julia native
formats: *column tables* (named tuples of equal length vectors) and
*row tables* (vectors of named tuples sharing the same
keys).

> Certain `JuliaDB.NDSparse` tables can be used for sparse data, but
> this is experimental and undocumented.

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

```julia ;;;
using MLJ # hide
julia> info("DecisionTreeClassifier")[:target_scitype_union]
```

This means that the union of scientific types of all elements 
a `DecisionTreeClassier` target must be a subtype of `Finite`.

The table below shows machine types that have scientific types different from `Unknown`:

`T`                         |     `scitype(x)` for `x::T`
----------------------------|:--------------------------------
`AbstractFloat`             |      `Continuous`
`Integer`                   |        `Count`
`CategoricalValue`          | `Multiclass{N}` where `N = nlevels(x)`, provided `x.pool.ordered == false` 
`CategoricalString`         | `Multiclass{N}` where `N = nlevels(x)`, provided `x.pool.ordered == false`
`CategoricalValue`          | `FiniteOrderedFactor{N}` where `N = nlevels(x)`, provided `x.pool.ordered == true` 
`CategoricalString`         | `FiniteOrderedFactor{N}` where `N = nlevels(x)` provided `x.pool.ordered == true`
`Integer`                   | `Count`
`Missing`                   |      `Missing`

Here `nlevels(x) = length(levels(x.pool))`.

You can use `scitype(x)` to determine the scientific type of a scalar
object `x`. Non-scalar objects (with the exception of tuples of
scalars) have `Unknown` scitype. 

```julia
julia> (scitype(42), scitype(float(π)), scitype("Julia"))
(Count, Continuous, Unknown)
```

**Special note on using integers.** According to the above, integers
cannot be used to represent `Multiclass` or `OrderedFactor` data. They
these can be represented by an unordered or ordered `CategoricalValue`
or `CategoricalString` (automatic if they are elements of a
`CategoricalArray`).

Methods exist to coerce the scientific type of a vector or table (see
below). [Task](working_with_tasks.md) constructors also allow one to
force the data being wrapped to have the desired scientific type.

```@docs
coerce
```






