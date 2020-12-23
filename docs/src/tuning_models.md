# Tuning models

Below we illustrate tuning model hyperparameters by grid and random
searches. For a complete list of available and planned tuning
strategies, see the [MLJTuning
page](https://github.com/alan-turing-institute/MLJTuning.jl#what-is-provided-here)

In MLJ tuning is implemented as a model wrapper. After wrapping a
model in a tuning strategy and binding the wrapped model to data in a
machine, `mach`, calling `fit!(mach)` instigates a search for optimal
model hyperparameters, within a specified `range`, and then
uses all supplied data to train the best model. To predict using the
optimal model, one just calls `predict(mach, Xnew)`. In this way the
wrapped model may be viewed as a "self-tuning" version of the
unwrapped model.


## Tuning a single hyperparameter using a grid search (Regression)

```@repl goof
using MLJ
X = MLJ.table(rand(100, 10));
y = 2X.x1 - X.x2 + 0.05*rand(100);
tree_model = @load DecisionTreeRegressor;
```

Let's tune `min_purity_increase` in the model above, using a
grid-search. To do so we will use the simplest `range` object, a
one-dimensional range object constructed using the `range` method:


```@repl goof
r = range(tree_model, :min_purity_increase, lower=0.001, upper=1.0, scale=:log);
self_tuning_tree_model = TunedModel(model=tree_model,
                                    resampling = CV(nfolds=3),
                                    tuning = Grid(resolution=10),
                                    range = r,
                                    measure = rms);
```

Incidentally, a grid is generated internally "over the range" by calling the
`iterator` method with an appropriate resolution:

```@repl goof
iterator(r, 5)
```

Non-numeric hyperparameters are handled a little differently:

```@repl goof
selector_model = FeatureSelector();
r2 = range(selector_model, :features, values = [[:x1,], [:x1, :x2]]);
iterator(r2)
```

Unbounded ranges are also permitted. See the `range` and `iterator`
docstrings below for details, and the `sampler` docstring for
generating random samples from one-dimensional ranges (used internally
by the `RandomSearch` strategy).

Returning to the wrapped tree model:

```@repl goof
self_tuning_tree = machine(self_tuning_tree_model, X, y);
fit!(self_tuning_tree, verbosity=0);
```

We can inspect the detailed results of the grid search with
`report(self_tuning_tree)` or just retrieve the optimal model, as here:

```@repl goof
fitted_params(self_tuning_tree).best_model
```

Predicting on new input observations using the optimal model:

```@repl goof
Xnew  = MLJ.table(rand(3, 10));
predict(self_tuning_tree, Xnew)
```


## Tuning multiple nested hyperparameters

The following model has another model, namely a `DecisionTreeRegressor`, as a
hyperparameter:

```@setup goof
tree_model = @load DecisionTreeRegressor
forest_model = EnsembleModel(atom=tree_model);
```

```julia
julia> tree_model = DecisionTreeRegressor()
julia> forest_model = EnsembleModel(atom=tree_model);
```

Ranges for nested hyperparameters are specified using dot syntax. In
this case we will specify a `goal` for the total number of grid
points:

```@repl goof
r1 = range(forest_model, :(atom.n_subfeatures), lower=1, upper=9);
r2 = range(forest_model, :bagging_fraction, lower=0.4, upper=1.0);
self_tuning_forest_model = TunedModel(model=forest_model,
                                      tuning=Grid(goal=30),
                                      resampling=CV(nfolds=6),
                                      range=[r1, r2],
                                      measure=rms);
self_tuning_forest = machine(self_tuning_forest_model, X, y);
fit!(self_tuning_forest, verbosity=0)
```

In this two-parameter case, a plot of the grid search results is also
available:

```julia
using Plots
plot(self_tuning_forest)
```

![](img/tuning_plot.png)

Instead of specifying a `goal`, we can declare a global `resolution`,
which is overriden for a particular parameter by pairing it's range
with the resolution desired. In the next example, the default
`resolution=100` is applied to the `r2` field, but a resolution of `3`
is applied to the `r1` field. Additionally, we ask that the grid
points be randomly traversed, and the the total number of evaluations
be limited to 25.

```@repl goof
tuning = Grid(resolution=100, shuffle=true, rng=1234)
self_tuning_forest_model = TunedModel(model=forest_model,
                                      tuning=tuning,
                                      resampling=CV(nfolds=6),
                                      range=[(r1, 3), r2],
                                      measure=rms,
                                      n=25);
fit!(machine(self_tuning_forest_model, X, y), verbosity=0)
```

For more options for a grid search, see [`Grid`](@ref) below.


## Tuning using a random search

Let's attempt to tune the same hyperparameters using a `RandomSearch`
tuning strategy. By default, bounded numeric ranges like `r1` and `r2`
are sampled uniformly (before rounding, in the case of the integer
range `r1`). Positive unbounded ranges are sampled using a Gamma
distribution by default, and all others using a (truncated) normal
distribution.

```@repl goof
self_tuning_forest_model = TunedModel(model=forest_model,
                                      tuning=RandomSearch(),
                                      resampling=CV(nfolds=6),
                                      range=[r1, r2],
                                      measure=rms,
                                      n=25);
self_tuning_forest = machine(self_tuning_forest_model, X, y);
fit!(self_tuning_forest, verbosity=0)
```

```julia
using Plots
plot(self_tuning_forest)
```

![](img/random_search_tuning_plot.png)

The prior distributions used for sampling each hyperparameter can be
customized, as can the global fallbacks. See the
[`RandomSearch`](@ref) doc-string below for details.


## Tuning a single hyperparameter using a grid search (Classification)

```@repl goof
using MLJ, DataFrames
X,y = @load_iris;
X = DataFrame(MLJ.table(X));
knn = @load KNNClassifier;
```

Let's tune `K` in the model above, using a grid-search. To do so we will use the simplest `range` object, a
one-dimensional range object constructed using the `range` method:

```@repl goof
K_range = range(knn, :K, lower=5, upper=20);
self_tuning_knn = TunedModel(model=knn,
                             resampling = CV(nfolds=4),
                             tuning = Grid(resolution=5),
                             range = K_range,
                             measure=misclassification_rate,
                             check_measure=false);
                             
self_tuning_knn_machine = machine(self_tuning_knn, X, y)
```

We can create a train/test partition to evaluate fit the crossvalidation procedure  and then evaluate the best model
found during crossvalidation.

```@repl goof
using Random
Random.seed!(123)
train_ind, test_ind = partition(1:length(y), 0.7, shuffle=true)
fit!(self_tuning_knn_machine, rows=train_ind, verbosity=0)
```
The best average results achieved by the best model using crossvaludation can be found in `.best_history_entry`

```@repl goof
report(m_self_tuning_knn).best_history_entry
```

The test results of the best model fitted during crossvaludation are
```@repl goof
yhat_test = predict_mode(self_tuning_knn_machine, rows=test_ind);
misclassification_rate(yhat_test, y[test_ind])
```

### TunnedModel with custom loss/score function

Users might want to select models according to a different loss or score function during Cross Validation.
This can be achieved using the `measure` attribute in the `TunedModel`.

Let us assume we have a custom function `custom_accuracy` defined as follows:

```@repl goof
custom_accuracy(y,yhat) = mean(y .== yhat)
```

First we need to tell MLJ if the function is a loss or a score.
If the function is a score, the higher the better. If the function is a loss, the lower the better.
In this case we want our function to be treated as a score, therefore we need to set

```@repl goof
MLJ.orientation(custom_accuracy) = :score
```

Note that our `custom_accuracy` is meant to work given a pair of vectors of the same type. 
Since the `KNNClassifier` outputs vectors contain `UnivariateFinite` elements we need to convert them to classes in order to 
use our `custom_accuracy`. This is done with `predict_mode` and we can pass this function as `operation` in a `TunedModel`.


```@repl goof

m_knn = machine(knn, X, y)
self_tuning_knn = TunedModel(model=knn,
                             resampling = CV(nfolds=4),
                             tuning = Grid(resolution=5),
                             range = K_range,
                             measure = custom_accuracy,
                             operation = predict_mode);

m_self_tuning_knn = machine(self_tuning_knn, X, y)
fit!(m_self_tuning_knn, rows=train_ind, verbosity=0)
```

Now we can inspect that the measure of the tunned model was `custom_accuracy` and the model selected as best
was the one with highest accuracy.

```@repl goof
report(m_self_tuning_knn).best_history_entry
```

The objective of this section is to showcase how to use a `TunedModel` with a user provied `measure`. Nevertheless, the previous work done with our `custom_accuracy` function could be done directly using the `accuracy` provided by MLJ. MLJ already knows this is a score function. Therefore there is no need to set `MLJ.orientation(accuracy) = :score` we could do the same process as follows:

```@repl goof
m_knn = machine(knn, X, y)

self_tuning_knn = TunedModel(model=knn,
                             resampling = CV(nfolds=4),
                             tuning = Grid(resolution=5),
                             range = K_range,
                             measure = accuracy,
                             operation=predict_mode);

m_self_tuning_knn = machine(self_tuning_knn, X, y)
fit!(m_self_tuning_knn, rows=train_ind, verbosity=0)
```








## API

```@docs
MLJBase.range
MLJBase.iterator
MLJBase.sampler
Distributions.fit(::Type{D}, ::MLJBase.NumericRange) where D<:Distributions.Distribution
MLJTuning.TunedModel
MLJTuning.Grid
MLJTuning.RandomSearch
```
