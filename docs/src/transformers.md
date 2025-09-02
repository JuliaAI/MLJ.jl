# Transformers and Other Unsupervised Models

Several unsupervised models used to perform common transformations, such as one-hot
encoding, missing value imputation, and categorical encoding, are available in MLJ
out-of-the-box (no need to load code with `@load`). They are detailed in [Built-in
transformers](@ref) below.

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

## Built-in transformers

For tutorials on the transformers below, refer to the [MLJTransforms
documentation](https://github.com/JuliaAI/MLJTransforms.jl). 

| Transformer                              | Brief Description                                                                             |
|:----------------------------------------:|:---------------------------------------------------------------------------------------------:|
| [`Standardizer`](@ref)                   | Transforming columns of numerical features by standardization                                 |
| [`UnivariateBoxCoxTransformer`](@ref)    | Apply BoxCox transformation given a single vector                                             |
| [`InteractionTransformer`](@ref)         | Transforming columns of numerical features to create new interaction features                 |
| [`UnivariateDiscretizer`](@ref)          | Discretize a continuous vector into an ordered factor                                         |
| [`FillImputer`](@ref)                    | Fill in missing values of features belonging to any scientific type                              |
| [`UnivariateFillImputer`](@ref)          | Fill in missing values in a single vector                                                        |
| [`UnivariateTimeTypeToContinuous`](@ref) | Transform a vector of time type into continuous type                                          |
| [`OneHotEncoder`](@ref)                  | Encode categorical variables into one-hot vectors                                             |
| [`ContinuousEncoder`](@ref)              | Adds type casting functionality to OnehotEncoder                                              |
| [`OrdinalEncoder`](@ref)                 | Encode categorical variables into ordered integers                                            |
| [`FrequencyEncoder`](@ref)               | Encode categorical variables into their normalized or unormalized frequencies                 |
| [`TargetEncoder`](@ref)                  | Encode categorical variables into relevant target statistics                                  |
| [`ContrastEncoder`](@ref)                | Allows defining a custom contrast encoder via a contrast matrix                               |
| [`CardinalityReducer`](@ref)             | Reduce cardinality of high cardinality categorical features by grouping infrequent categories |
| [`MissingnessEncoder`](@ref)             | Encode missing values of categorical features into new values                                 |


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
of any associated machine. See [Static transformers](@ref) for
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

```@setup predtrans
using MLJ
```

```@example predtrans
import Random.seed!
seed!(123)

X, y = @load_iris
KMeans = @load KMeans pkg=Clustering
kmeans = KMeans()
mach = machine(kmeans, X) |> fit!
nothing # hide
```

Transforming:
```@example predtrans
Xsmall = transform(mach)
selectrows(Xsmall, 1:4) |> pretty
```

Predicting:
```@example predtrans
yhat = predict(mach)
compare = zip(yhat, y) |> collect
```

```@example predtrans
compare[1:8]
```

```@example predtrans
compare[51:58]
```

```@example predtrans
compare[101:108]
```

## Reference

```@docs
MLJTransforms.Standardizer
MLJTransforms.UnivariateBoxCoxTransformer
MLJTransforms.InteractionTransformer
MLJTransforms.UnivariateDiscretizer
MLJTransforms.FillImputer
MLJTransforms.UnivariateFillImputer
MLJTransforms.UnivariateTimeTypeToContinuous
MLJTransforms.OneHotEncoder
MLJTransforms.ContinuousEncoder
MLJTransforms.OrdinalEncoder
MLJTransforms.FrequencyEncoder
MLJTransforms.TargetEncoder
MLJTransforms.ContrastEncoder
MLJTransforms.CardinalityReducer
MLJTransforms.MissingnessEncoder
```
