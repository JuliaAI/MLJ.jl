# Transformers and Other Unsupervised Models

Several unsupervised models used to perform common transformations,
such as one-hot encoding, are available in MLJ out-of-the-box. These
are detailed in [Built-in transformers](@ref) below.

A transformer is *static* if it has no learned parameters. While such
a transformer is tantamount to an ordinary function, realizing it as
an MLJ static transformer (a subtype of `Static <: Unsupervised`) can be
useful, especially if the function depends on parameters the user
would like to manipulate (which become *hyper-parameters* of the
model). The necessary syntax for defining your own static transformers
is described in [Static transformers](@ref) below.

Some unsupervised models, such as clustering algorithms, have a
`predict` method in addition to a `transform` method. We give an
example of this in [Transformers that also predict](@ref)

Finally, we note that models that fit a distribution, or more generally
a sampler object, to some data, which are sometimes viewed as
unsupervised, are treated in MLJ as *supervised* models. See [Models
that learn a probability distribution](@ref) for an example.


## Built-in transformers

```@docs
MLJModels.Standardizer
MLJModels.OneHotEncoder
MLJModels.ContinuousEncoder
MLJModels.FillImputer
MLJModels.UnivariateFillImputer
MLJModels.FeatureSelector
MLJModels.UnivariateBoxCoxTransformer
MLJModels.UnivariateDiscretizer
MLJModels.UnivariateTimeTypeToContinuous
```


## Static transformers

A *static transformer* is a model for transforming data that does not generalize to new
data (does not "learn") but which nevertheless has hyperparameters. For example, the
`DBSAN` clustering model from Clustering.jl can assign labels to some collection of
observations, cannot directly assign a label to some new observation.

The general user may define their own static models. The main use-case is insertion into a
[Linear Pipelines](@ref) some parameter-dependent transformation. (If a static transformer
has no hyper-parameters, it is tantamount to an ordinary function. An ordinary function
can be inserted directly into a pipeline; the situation for learning networks is only
[slightly more complicated](@ref node_overloading).

The following example defines a new model type `Averager` to perform
the weighted average of two vectors (target predictions, for
example). We suppose the weighting is normalized, and therefore
controlled by a single hyper-parameter, `mix`.

```@setup boots
using MLJ
```

```@example boots
mutable struct Averager <: Static
    mix::Float64
end

MLJ.transform(a::Averager, _, y1, y2) = (1 - a.mix)*y1 + a.mix*y2
```

*Important.* Note the sub-typing `<: Static`.

Such static transformers with (unlearned) parameters can have
arbitrarily many inputs, but only one output. In the single input case,
an `inverse_transform` can also be defined. Since they have no real
learned parameters, you bind a static transformer to a machine without
specifying training arguments; there is no need to `fit!` the machine:

```@example boots
mach = machine(Averager(0.5))
transform(mach, [1, 2, 3], [3, 2, 1])
```

Let's see how we can include our `Averager` in a [learning network](@ref "Learning
Networks") to mix the predictions of two regressors, with one-hot encoding of the
inputs. Here's two regressors for mixing, and some dummy data for testing our learning
network:

```@example boots
ridge = (@load RidgeRegressor pkg=MultivariateStats)()
knn = (@load KNNRegressor)()

import Random.seed!
seed!(112)
X = (
    x1=coerce(rand("ab", 100), Multiclass),
    x2=rand(100),
)
y = X.x2 + 0.05*rand(100)
schema(X)
```

And the learning network:

```@example boots
Xs = source(X)
ys = source(y)

averager = Averager(0.5)

mach0 = machine(OneHotEncoder(), Xs)
W = transform(mach0, Xs) # one-hot encode the input

mach1 = machine(ridge, W, ys)
y1 = predict(mach1, W)

mach2 = machine(knn, W, ys)
y2 = predict(mach2, W)

mach4= machine(averager)
yhat = transform(mach4, y1, y2)

# test:
fit!(yhat)
Xnew = selectrows(X, 1:3)
yhat(Xnew)
```

We next "export" the learning network as a standalone composite model type. First we need
a struct for the composite model. Since we are restricting to `Deterministic` component
regressors, the composite will also make deterministic predictions, and so gets the
supertype `DeterministicNetworkComposite`:

```@example boots
mutable struct DoubleRegressor <: DeterministicNetworkComposite
    regressor1
    regressor2
    averager
end
```

As described in [Learning Networks](@ref), we next paste the learning network into a
`prefit` declaration, replace the component models with symbolic placeholders, and add a
learning network "interface":

```@example boots
import MLJBase
function MLJBase.prefit(composite::DoubleRegressor, verbosity, X, y)
    Xs = source(X)
    ys = source(y)

    mach0 = machine(OneHotEncoder(), Xs)
    W = transform(mach0, Xs) # one-hot encode the input

    mach1 = machine(:regressor1, W, ys)
    y1 = predict(mach1, W)

    mach2 = machine(:regressor2, W, ys)
    y2 = predict(mach2, W)

    mach4= machine(:averager)
    yhat = transform(mach4, y1, y2)

    # learning network interface:
    (; predict=yhat)
end
```

The new model type can be evaluated like any other supervised model:

```@example boots
X, y = @load_reduced_ames;
composite = DoubleRegressor(ridge, knn, Averager(0.5))
```

```@example boots
composite.averager.mix = 0.25 # adjust mix from default of 0.5
evaluate(composite, X, y, measure=l1)
```

A static transformer can also expose byproducts of the transform computation in the report
of any associated machine. See [Static models (models that do not generalize)](@ref) for
details.

## Transformers that also predict

Some clustering algorithms learn to label data by identifying a
collection of "centroids" in the training data. Any new input
observation is labeled with the cluster to which it is closest (this
is the output of `predict`) while the vector of all distances from the
centroids defines a lower-dimensional representation of the
observation (the output of `transform`). In the following example a
K-means clustering algorithm assigns one of three labels 1, 2, 3 to
the input features of the iris data set and compares them with the
actual species recorded in the target (not seen by the algorithm).

```julia-repl
julia> import Random.seed!
julia> seed!(123)

julia> X, y = @load_iris;
julia> KMeans = @load KMeans pkg=ParallelKMeans
julia> kmeans = KMeans()
julia> mach = machine(kmeans, X) |> fit!

julia> # transforming:
julia> Xsmall = transform(mach);
julia> selectrows(Xsmall, 1:4) |> pretty
┌─────────────────────┬────────────────────┬────────────────────┐
│ x1                  │ x2                 │ x3                 │
│ Float64             │ Float64            │ Float64            │
│ Continuous          │ Continuous         │ Continuous         │
├─────────────────────┼────────────────────┼────────────────────┤
│ 0.0215920000000267  │ 25.314260355029603 │ 11.645232464391299 │
│ 0.19199200000001326 │ 25.882721893491123 │ 11.489658693899486 │
│ 0.1699920000000077  │ 27.58656804733728  │ 12.674412792260142 │
│ 0.26919199999998966 │ 26.28656804733727  │ 11.64392098898145  │
└─────────────────────┴────────────────────┴────────────────────┘

julia> # predicting:
julia> yhat = predict(mach);
julia> compare = zip(yhat, y) |> collect;
julia> compare[1:8]
8-element Array{Tuple{CategoricalValue{Int64,UInt32},CategoricalString{UInt32}},1}:
 (1, "setosa")
 (1, "setosa")
 (1, "setosa")
 (1, "setosa")
 (1, "setosa")
 (1, "setosa")
 (1, "setosa")
 (1, "setosa")

julia> compare[51:58]
8-element Array{Tuple{CategoricalValue{Int64,UInt32},CategoricalString{UInt32}},1}:
 (2, "versicolor")
 (3, "versicolor")
 (2, "versicolor")
 (3, "versicolor")
 (3, "versicolor")
 (3, "versicolor")
 (3, "versicolor")
 (3, "versicolor")

julia> compare[101:108]
8-element Array{Tuple{CategoricalValue{Int64,UInt32},CategoricalString{UInt32}},1}:
 (2, "virginica")
 (3, "virginica")
 (2, "virginica")
 (2, "virginica")
 (2, "virginica")
 (2, "virginica")
 (3, "virginica")
 (2, "virginica")
```
