# Performance Measures

In MLJ loss functions, scoring rules, sensitivities, and so on, are collectively referred
to as *measures*. Presently, MLJ includes a few built-in measures,
provides support for the loss functions in the
[LossFunctions.jl](https://github.com/JuliaML/LossFunctions.jl) library,
and allows for users to define their own custom measures.

Providing further measures for probabilistic predictors, such as
proper scoring rules, and for constructing multi-target product
measures, is a work in progress.

*Note for developers:* The measures interface and the built-in measures
 described here are defined in MLJBase.


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
which can be provided when the measure supports this.

```@repl losses_and_scores
using MLJ
y = [1, 2, 3, 4];
ŷ = [2, 3, 3, 3];
w = [1, 2, 2, 1];
rms(ŷ, y) # reports an aggregrate loss
l1(ŷ, y, w) # reports per observation losses
y = categorical(["male", "female", "female"])
male = y[1]; female = y[2];
d = UnivariateFinite([male, female], [0.55, 0.45]);
ŷ = [d, d, d];
cross_entropy(ŷ, y)
```

## Traits and custom measures

Notice that `l1` reports per-sample evaluations, while `rms`
only reports an aggregated result. This and other behavior can be
gleaned from measure *traits* which are summarized by the `info`
method:

```@repl losses_and_scores
info(l1)
```

Use `measures()` to list all measures and `measures(conditions...)` to
search for measures with given traits (as you would [query
models](model_search.md)).

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
and "marginal loss" functions for `Binary` targets. While the
LossFunctions,jl interface differs from the present one (for, example
`Binary` observations must be +1 or -1), one can safely pass the loss
functions defined there to any MLJ algorithm, which re-interprets it
under the hood. Note that the "distance losses" in the package apply
to deterministic predictions, while the "marginal losses" apply to
probabilistic predictions.

```@repl losses_and_scores
using LossFunctions
X = (x1=rand(5), x2=rand(5)); y = categorical(["y", "y", "y", "n", "y"]); w = [1, 2, 1, 2, 3];
mach = machine(ConstantClassifier(), X, y);
holdout = Holdout(fraction_train=0.6);
evaluate!(mach,
          measure=[ZeroOneLoss(), L1HingeLoss(), L2HingeLoss(), SigmoidLoss()],
          resampling=holdout,
          operation=predict,
          weights=w,
          verbosity=0)
```

*Note:* Although `ZeroOneLoss(ŷ, y)` makes no sense (neither `ŷ` nor
`y` have a type expected by LossFunctions.jl), one can instead use the
adaptor `MLJ.value` as discussed above:

```@repl losses_and_scores
ŷ = predict(mach, X);
loss = MLJ.value(ZeroOneLoss(), ŷ, X, y, w) # X is ignored here
mean(loss) ≈ misclassification_rate(mode.(ŷ), y, w)
```


## Built-in measures 


```@docs
area_under_curve
```

```@docs
accuracy
```

```@docs
balanced_accuracy
```

```@docs
BrierScore
```

```@docs
cross_entropy
```

```@docs
FScore
```

```@docs
false_discovery_rate
```

```@docs
false_negative
```

```@docs
false_negative_rate
```

```@docs
false_positive
```

```@docs
false_positive_rate
```

```@docs
l1
```

```@docs
l2
```

```@docs
mae
```

```@docs
matthews_correlation
```

```@docs
misclassification_rate
```

```@docs
negative_predictive_value
```

```@docs
positive_predictive_value
```

```@docs
rms
```

```@docs
rmsl
```

```@docs
rmslp1
```

```@docs
rmsp
```

```@docs
true_negative
```

```@docs
true_negative_rate
```

```@docs
true_positive
```

```@docs
true_positive_rate
```

## List of LossFunctions.jl measures

`DWDMarginLoss()`, `ExpLoss()`, `L1HingeLoss()`, `L2HingeLoss()`,
`L2MarginLoss()`, `LogitMarginLoss()`, `ModifiedHuberLoss()`,
`PerceptronLoss()`, `ScaledMarginLoss()`, `SigmoidLoss()`,
`SmoothedL1HingeLoss()`, `ZeroOneLoss()`, `HuberLoss()`,
`L1EpsilonInsLoss()`, `L2EpsilonInsLoss()`, `LPDistLoss()`,
`LogitDistLoss()`, `PeriodicLoss()`, `QuantileLoss()`,
`ScaledDistanceLoss()`.


## Other performance related tools

```@docs
confusion_matrix
```

```@docs
roc_curve
```
