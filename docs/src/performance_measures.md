# Performance Measures

In MLJ loss functions, scoring rules, sensitivities, and so on, are
collectively referred to as *measures*. These include re-exported loss
functions from the
[LossFunctions.jl](https://github.com/JuliaML/LossFunctions.jl)
library, overloaded to behave the same way as the built-in measures.

To see list all measures, run `measures()`.  Further measures for
probabilistic predictors, such as proper scoring rules, and for
constructing multi-target product measures, are planned.  If you'd like
to see measure added to MLJ, post a comment
[here](https://github.com/JuliaAI/MLJBase.jl/issues/299)

*Note for developers:* The measures interface and the built-in
measures described here are defined in MLJBase, but will ultimately live
in a separate package.


## Using built-in measures

These measures all have the common calling syntax

```julia
measure(ŷ, y)
```

or

```julia
measure(ŷ, y, w)
```

where `y` iterates over observations of some target variable, and `ŷ`
iterates over predictions (`Distribution` or `Sampler` objects in the
probabilistic case). Here `w` is an optional vector of sample weights,
or a dictionary of class weights, when these are supported by the
measure.

```@repl losses_and_scores
using MLJ
y = [1, 2, 3, 4];
ŷ = [2, 3, 3, 3];
w = [1, 2, 2, 1];
rms(ŷ, y) # reports an aggregrate loss
l2(ŷ, y, w) # reports per observation losses
y = coerce(["male", "female", "female"], Multiclass)
d = UnivariateFinite(["male", "female"], [0.55, 0.45], pool=y);
ŷ = [d, d, d];
log_loss(ŷ, y)
```

The measures `rms`, `l2` and `log_loss` illustrated here are actually
        instances of measure *types*. For, example, `l2 = LPLoss(p=2)` and
`log_loss = LogLoss() = LogLoss(tol=eps())`. Common aliases are
provided:

```@repl losses_and_scores
cross_entropy
```

## Traits and custom measures

Notice that `l1` reports per-sample evaluations, while `rms`
only reports an aggregated result. This and other behavior can be
gleaned from measure *traits* which are summarized by the `info`
method:

```@repl losses_and_scores
info(l1)
```

Query the doc-string for a measure using the name of its type:

```@repl losses_and_scores
rms
@doc RootMeanSquaredError # same as `?RootMeanSqauredError
```

Use `measures()` to list all measures, and `measures(conditions...)` to
search for measures with given traits (as you would [query
models](model_search.md)). The trait `instances` list the actual
callable instances of a given measure type (typically aliases for the
default instance).

```@docs
measures(conditions...)
```

A user-defined measure in MLJ can be passed to the `evaluate!`
method, and elsewhere in MLJ, provided it is a function or callable
object conforming to the above syntactic conventions. By default, a
custom measure is understood to:

- be a loss function (rather than a score)

- report an aggregated value (rather than per-sample evaluations)

- be feature-independent

To override this behaviour one simply overloads the appropriate trait,
as shown in the following examples:

```@repl losses_and_scores
y = [1, 2, 3, 4];
ŷ = [2, 3, 3, 3];
w = [1, 2, 2, 1];
my_loss(ŷ, y) = maximum((ŷ - y).^2);
my_loss(ŷ, y)
my_per_sample_loss(ŷ, y) = abs.(ŷ - y);
MLJ.reports_each_observation(::typeof(my_per_sample_loss)) = true;
my_per_sample_loss(ŷ, y)
my_weighted_score(ŷ, y) = 1/mean(abs.(ŷ - y));
my_weighted_score(ŷ, y, w) = 1/mean(abs.((ŷ - y).^w));
MLJ.supports_weights(::typeof(my_weighted_score)) = true;
MLJ.orientation(::typeof(my_weighted_score)) = :score;
my_weighted_score(ŷ, y)
X = (x=rand(4), penalty=[1, 2, 3, 4]);
my_feature_dependent_loss(ŷ, X, y) = sum(abs.(ŷ - y) .* X.penalty)/sum(X.penalty);
MLJ.is_feature_dependent(::typeof(my_feature_dependent_loss)) = true
my_feature_dependent_loss(ŷ, X, y)
```

The possible signatures for custom measures are: `measure(ŷ, y)`,
`measure(ŷ, y, w)`, `measure(ŷ, X, y)` and `measure(ŷ, X, y, w)`, each
measure implementing one non-weighted version, and possibly a second
weighted version.

*Implementation detail:* Internally, every measure is evaluated using
the syntax

```julia
MLJ.value(measure, ŷ, X, y, w)
```
and the traits determine what can be ignored and how `measure` is actually called. If `w=nothing` then the non-weighted form of `measure` is
dispatched.

## Using measures from LossFunctions.jl

The [LossFunctions.jl](https://github.com/JuliaML/LossFunctions.jl)
package includes "distance loss" functions for `Continuous` targets,
and "marginal loss" functions for `Finite{2}` (binary) targets. While the
LossFunctions.jl interface differs from the present one (for, example
binary observations must be +1 or -1), MLJ has overloaded instances
of the LossFunctions.jl types to behave the same as the built-in
types.

Note that the "distance losses" in the package apply to deterministic
predictions, while the "marginal losses" apply to probabilistic
predictions.


## List of measures

All measures listed below have a doc-string associated with the measure's
*type*. So, for example, do `?LPLoss` not `?l2`.

```@setup losses_and_scores
using DataFrames
```

```@example
ms = measures()
types = map(ms) do m
    m.name
end
instance = map(ms) do m m.instances end
table = (type=types, instances=instance)
DataFrame(table)
```


## Other performance related tools

In MLJ one computes a confusion matrix by calling an instance of the
`ConfusionMatrix` measure type on the data:

```@docs
ConfusionMatrix
```

```@docs
roc_curve
```
