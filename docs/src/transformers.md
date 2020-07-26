# Transformers and other unsupervised models

Several unsupervised models used to perform common transformations,
such as one-hot encoding, are available in MLJ out-of-the-box. These
are detailed in [Built-in transformers](@ref) below.

A transformer is *static* if it has no learned parameters. While such
a transformer is tantamount to an ordinary function, realizing it as
an MLJ static transformer (subtype of `Static <: Unsupervised`) can be
useful, especially if the function depends on parameters the user
would like to manipulate (which become *hyper-parameters* of the
model). The necessary syntax for defining your own static transformers
is described in [Static transformers](@ref) below.

Some unsupervised models, such as clustering algorithms, have a
`predict` method in addition to a `transform` method. We give an
example of this in [Transformers that also predict](@ref)

Finally we note that models that fit a distribution, or more generally
a sampler object, to some data, which are sometimes viewed as
unsupervised, are treated in MLJ as *supervised* models. See [Models
that learn a probability distribution](@ref) for an example.


## Built-in transformers

```@docs
MLJModels.UnivariateStandardizer
MLJModels.Standardizer
MLJModels.OneHotEncoder
MLJModels.ContinuousEncoder
MLJModels.FeatureSelector
MLJModels.UnivariateBoxCoxTransformer
MLJModels.UnivariateDiscretizer
MLJModels.FillImputer
```


## Static transformers

The main use-case for static transformers is for insertion into a
[`@pipeline`](@ref) or other exported learning network (see [Composing
Models](@ref)). If a static transformer has no hyper-parameters, it is
tantamount to an ordinary function. An ordinary function can be
inserted directly into a `@pipeline`; the situation for learning
networks is only slightly more complicated; see [Static operations on
nodes](@ref).

The following example defines a new model type `Averager` to perform
the weighted average of two vectors (target predictions, for
example). We suppose the weighting is normalized, and therefore
controlled by a single hyper-parameter, `mix`.

```julia
mutable struct Averager <: Static
    mix::Float64
end

import MLJBase
MLJBase.transform(a::Averager, _, y1, y2) = (1 - a.mix)*y1 + a.mix*y2
```

*Important.* Note the sub-typing `<: Static`.

Such static transformers with (unlearned) parameters can have
arbitrarily many inputs, but only one output. In the single input case
an `inverse_transform` can also be defined. Since they have no real
learned parameters, you bind a static transformer to a machine without
specifying training arguments.

```julia
mach = machine(Averager(0.5)) |> fit!
transform(mach, [1, 2, 3], [3, 2, 1])
3-element Array{Float64,1}:
 2.0
 2.0
 2.0
```

Let's see how we can include our `Averager` in a learning network (see
[Composing Models](@ref)) to mix the predictions of two regressors,
with one-hot encoding of the inputs:

```julia
X = source()
y = source() #MLJ will automatically infer this a target node 

ridge = @load RidgeRegressor pkg=MultivariateStats
knn = @load KNNRegressor
averager = Averager(0.5)

hotM = machine(OneHotEncoder(), X)
W = transform(hotM, X) # one-hot encode the input

ridgeM = machine(ridge, W, y)
y1 = predict(ridgeM, W)

knnM = machine(knn, W, y)
y2 = predict(knnM, W)

averagerM= machine(averager)
yhat = transform(averagerM, y1, y2)
```

Now we export to obtain a `Deterministic` composite model and then 
instantiate composite model

```julia
learning_mach = machine(Deterministic(), X, y; predict=yhat)
Machine{DeterministicSurrogate} @772 trained 0 times.
  args: 
    1:	Source @415 ⏎ `Unknown`
    2:	Source @389 ⏎ `Unknown`


@from_network learning_mach struct DoubleRegressor
       regressor1=ridge
       regressor2=knn
       averager=averager
       end
       
composite = DoubleRegressor()
julia> composite = DoubleRegressor()
DoubleRegressor(
    regressor1 = RidgeRegressor(
            lambda = 1.0),
    regressor2 = KNNRegressor(
            K = 5,
            algorithm = :kdtree,
            metric = Distances.Euclidean(0.0),
            leafsize = 10,
            reorder = true,
            weights = :uniform),
    averager = Averager(
            mix = 0.5)) @301

```

which can be can be evaluated like any other model:

```julia
composite.averager.mix = 0.25 # adjust mix from default of 0.5
evaluate(composite, (@load_reduced_ames)..., measure=rms)
julia> evaluate(composite, (@load_reduced_ames)..., measure=rms)
Evaluating over 6 folds: 100%[=========================] Time: 0:00:00
┌───────────┬───────────────┬────────────────────────────────────────────────────────┐
│ _.measure │ _.measurement │ _.per_fold                                             │
├───────────┼───────────────┼────────────────────────────────────────────────────────┤
│ rms       │ 26800.0       │ [21400.0, 23700.0, 26800.0, 25900.0, 30800.0, 30700.0] │
└───────────┴───────────────┴────────────────────────────────────────────────────────┘
_.per_observation = [missing]
```


## Transformers that also predict

Commonly, clustering algorithms learn to label data by identifying a
collection of "centroids" in the training data. Any new input
observation is labeled with the cluster to which it is closest (this
is the output of `predict`) while the vector of all distances from the
centroids defines a lower-dimensional representation of the
observation (the output of `transform`). In the following example a
K-means clustering algorithm assigns one of three labels 1, 2, 3 to
the input features of the iris data set and compares them with the
actual species recorded in the target (not seen by the algorithm).

```julia
import Random.seed!
seed!(123)

X, y = @load_iris;
model = @load KMeans pkg=ParallelKMeans
mach = machine(model, X) |> fit!

# transforming:
Xsmall = transform(mach);
selectrows(Xsmall, 1:4) |> pretty
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

# predicting:
yhat = predict(mach);
compare = zip(yhat, y) |> collect;
compare[1:8]
8-element Array{Tuple{CategoricalValue{Int64,UInt32},CategoricalString{UInt32}},1}:
 (1, "setosa")
 (1, "setosa")
 (1, "setosa")
 (1, "setosa")
 (1, "setosa")
 (1, "setosa")
 (1, "setosa")
 (1, "setosa")

compare[51:58]
8-element Array{Tuple{CategoricalValue{Int64,UInt32},CategoricalString{UInt32}},1}:
 (2, "versicolor")
 (3, "versicolor")
 (2, "versicolor")
 (3, "versicolor")
 (3, "versicolor")
 (3, "versicolor")
 (3, "versicolor")
 (3, "versicolor")

compare[101:108]
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

