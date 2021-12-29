# Linear Pipelines

In MLJ a *pipeline* is a composite model in which models are chained
together in a linear (non-branching) chain. For other arrangements,
including custom architectures via learning networks, see [Composing
Models](@ref).

For purposes of illustration, consider a supervised learning problem
with the following toy data:

```@setup 7
using MLJ
MLJ.color_off()
```

```@example 7
using MLJ
X = (age    = [23, 45, 34, 25, 67],
     gender = categorical(['m', 'm', 'f', 'm', 'f']));
y = [67.0, 81.5, 55.6, 90.0, 61.1]
     nothing # hide
```

We would like to train using a K-nearest neighbor model, but the
model type `KNNRegressor` assumes the features are all
`Continuous`. This can be fixed by first:

- coercing the `:age` feature to have `Continuous` type by replacing
  `X` with `coerce(X, :age=>Continuous)`
- standardizing continuous features and one-hot encoding the
  `Multiclass` features using the `ContinuousEncoder` model
  
However, we can avoid separately applying these preprocessing steps
(two of which require `fit!` steps) by combining them with the
supervised `KKNRegressor` model in a new *pipeline* model, using
Julia's `|>` syntax:

```@example 7
KNNRegressor = @load KNNRegressor pkg=NearestNeighborModels
pipe = (X -> coerce(X, :age=>Continuous)) |> ContinuousEncoder() |> KNNRegressor(K=2)
```

We see above that `pipe` is a model whose hyperparameters are
themselves other models or a function. (The names of these
hyper-parameters are automatically generated. To specify your own
names, use the explicit [`Pipeline`](@ref) constructor instead.)

The `|>` syntax can also be used to extend an existing pipeline or
concatenate two existing pipelines. So, we could instead have defined:

```julia
pipe_transformer = (X -> coerce(X, :age=>Continuous)) |> ContinuousEncoder()
pipe = pipe_transformer |> KNNRegressor(K=2)
```

A pipeline is just a model like any other. For example, we can
evaluate it's performance on the data above:

```@example 7
evaluate(pipe, X, y, resampling=CV(nfolds=3), measure=mae)
```

To include target transformations in a pipeline, wrap the supervised
component using [`TransformedTargetModel`](@ref).


```@docs
Pipeline
```
