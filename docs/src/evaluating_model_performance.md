# Evaluating Model Performance

MLJ allows quick evaluation of a supervised model's performance
against a battery of selected losses or scores.
For more on available performance measures, see
[Performance Measures](performance_measures.md).

In addition to hold-out and cross-validation, the user can specify
an explicit list of train/test pairs of row indices for resampling, or
define new resampling strategies.

For simultaneously evaluating *multiple* models, see [Comparing models of different type
and nested cross-validation](@ref).

For externally logging the outcomes of performance evaluation experiments, see [Logging
Workflows](@ref)

## Evaluating against a single measure

```@setup evaluation_of_supervised_models
using MLJ
MLJ.color_off()
```

```@repl evaluation_of_supervised_models
using MLJ
X = (a=rand(12), b=rand(12), c=rand(12));
y = X.a + 2X.b + 0.05*rand(12);
model = (@load RidgeRegressor pkg=MultivariateStats verbosity=0)()
cv = CV(nfolds=3)
evaluate(model, X, y, resampling=cv, measure=l2, verbosity=0)
```

Alternatively, instead of applying `evaluate` to a model + data, one
may call `evaluate!` on an existing machine wrapping the model in
data:

```@repl evaluation_of_supervised_models
mach = machine(model, X, y)
evaluate!(mach, resampling=cv, measure=l2, verbosity=0)
```

(The latter call is a mutating call as the learned parameters stored in the
machine potentially change. )

## Multiple measures

Multiple measures are specified as a vector:

```@repl evaluation_of_supervised_models
evaluate!(
    mach,
    resampling=cv,
    measures=[l1, rms, rmslp1],
    verbosity=0,
)
```

[Custom measures](@ref) can also be provided.

## Specifying weights

Per-observation weights can be passed to measures. If a measure does not support weights,
the weights are ignored:

```@repl evaluation_of_supervised_models
holdout = Holdout(fraction_train=0.8)
weights = [1, 1, 2, 1, 1, 2, 3, 1, 1, 2, 3, 1];
evaluate!(
    mach,
    resampling=CV(nfolds=3),
    measure=[l2, rsquared],
    weights=weights,
)
```

In classification problems, use `class_weights=...` to specify a class weight dictionary.

```@docs
MLJBase.evaluate!
MLJBase.evaluate
MLJBase.PerformanceEvaluation
```

## User-specified train/test sets

Users can either provide an explicit list of train/test pairs of row indices for resampling, as in this example:

```@repl evaluation_of_supervised_models
fold1 = 1:6; fold2 = 7:12;
evaluate!(
    mach,
    resampling = [(fold1, fold2), (fold2, fold1)],
    measures=[l1, l2],
    verbosity=0,
)
```

Or the user can define their own re-usable `ResamplingStrategy` objects; see [Custom
resampling strategies](@ref) below.


## Built-in resampling strategies

```@docs
MLJBase.Holdout
MLJBase.CV
MLJBase.StratifiedCV
MLJBase.TimeSeriesCV
MLJBase.InSample
```



## Custom resampling strategies

To define a new resampling strategy, make relevant parameters of
your strategy the fields of a new type `MyResamplingStrategy <:
MLJ.ResamplingStrategy`, and implement one of the following methods:

```julia
MLJ.train_test_pairs(my_strategy::MyResamplingStrategy, rows)
MLJ.train_test_pairs(my_strategy::MyResamplingStrategy, rows, y)
MLJ.train_test_pairs(my_strategy::MyResamplingStrategy, rows, X, y)
```

Each method takes a vector of indices `rows` and returns a
vector `[(t1, e1), (t2, e2), ... (tk, ek)]` of train/test pairs of row
indices selected from `rows`. Here `X`, `y` are the input and target
data (ignored in simple strategies, such as `Holdout` and `CV`).

Here is the code for the `Holdout` strategy as an example:

```julia
struct Holdout <: ResamplingStrategy
    fraction_train::Float64
    shuffle::Bool
    rng::Union{Int,AbstractRNG}

    function Holdout(fraction_train, shuffle, rng)
        0 < fraction_train < 1 ||
            error("`fraction_train` must be between 0 and 1.")
        return new(fraction_train, shuffle, rng)
    end
end

# Keyword Constructor
function Holdout(; fraction_train::Float64=0.7, shuffle=nothing, rng=nothing)
    if rng isa Integer
        rng = MersenneTwister(rng)
    end
    if shuffle === nothing
        shuffle = ifelse(rng===nothing, false, true)
    end
    if rng === nothing
        rng = Random.GLOBAL_RNG
    end
    return Holdout(fraction_train, shuffle, rng)
end

function train_test_pairs(holdout::Holdout, rows)
    train, test = partition(rows, holdout.fraction_train,
                          shuffle=holdout.shuffle, rng=holdout.rng)
    return [(train, test),]
end
```
