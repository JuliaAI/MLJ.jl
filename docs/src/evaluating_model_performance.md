# Evaluating Model Performance

MLJ allows quick evaluation of a supervised model's performance
against a battery of selected losses or scores. For more on available
performance measures, see [Performance
Measures](performance_measures.md).

In addition to hold-out and cross-validation, the user can specify
their own list of train/test pairs of row indices for resampling, or
define their own re-usable resampling strategies.

For simultaneously evaluating *multiple* models and/or data
sets, see [Benchmarking](benchmarking.md).


### Evaluating against a single measure

```@setup evaluation_of_supervised_models
import Base.eval
using MLJ
MLJ.color_off() 
```

```@repl evaluation_of_supervised_models
using MLJ
X = (a=rand(12), b=rand(12), c=rand(12));
y = X.a + 2X.b + 0.05*rand(12);
model = @load RidgeRegressor pkg=MultivariateStats
cv=CV(nfolds=3)
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

### Multiple measures

```@repl evaluation_of_supervised_models
evaluate!(mach,
          resampling=cv,
          measure=[l1, rms, rmslp1], verbosity=0)
```

### Custom measures and weighted measures

```@repl evaluation_of_supervised_models
my_loss(yhat, y) = maximum((yhat - y).^2);

my_per_observation_loss(yhat, y) = abs.(yhat - y);
MLJ.reports_each_observation(::typeof(my_per_observation_loss)) = true;

my_weighted_score(yhat, y) = 1/mean(abs.(yhat - y));
my_weighted_score(yhat, y, w) = 1/mean(abs.((yhat - y).^w));
MLJ.supports_weights(::typeof(my_weighted_score)) = true;
MLJ.orientation(::typeof(my_weighted_score)) = :score;

holdout = Holdout(fraction_train=0.8)
weights = [1, 1, 2, 1, 1, 2, 3, 1, 1, 2, 3, 1];
evaluate!(mach,
          resampling=CV(nfolds=3),
          measure=[my_loss, my_per_observation_loss, my_weighted_score, l1],
          weights=weights, verbosity=0)
```

### User-specified train/test sets

Users can either provide their own list of train/test pairs of row indices for resampling, as in this example:

```@repl evaluation_of_supervised_models
fold1 = 1:6; fold2 = 7:12;
evaluate!(mach,
          resampling = [(fold1, fold2), (fold2, fold1)],
          measure=[l1, l2], verbosity=0)
```

Or define their own re-usable `ResamplingStrategy` objects, - see
[Custom resampling strategies](@ref) below.


### Resampling strategies

`Holdout` and `CV` (cross-validation) resampling strategies are
available:


```@docs
Holdout
```

```@docs
CV
```


### Custom resampling strategies

To define your own resampling strategy, make relevant parameters of
your strategy the fields of a new type `MyResamplingStrategy <:
MLJ.ResamplingStrategy`, and implement
`MLJ.train_test_pairs(my_strategy::MyResamplingStrategy, rows, X, y)`,
a method which will take a vector of indices `rows` and return a
vector `[(t1, e1), (t2, e2), ... (tk, ek)]` of train/test pairs of row
indices selected from `rows`. Here `X`, `y` are the input and target
data (ignored in simple strategies, such as `Holdout` and `CV`).

Here is the code for the `Holdout` strategy as an example:

```julia
struct Holdout <: MLJ.ResamplingStrategy
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
function Holdout(; fraction_train::Float64=0.7,
                   shuffle::Bool=false,
                   rng::Union{Int,AbstractRNG}=Random.GLOBAL_RNG)
    Holdout(fraction_train, shuffle, rng)
end

function MLJ.train_test_pairs(holdout::Holdout, rows, X, y)
    if holdout.rng isa Integer
        rng = MersenneTwister(holdout.rng)
    else
        rng = holdout.rng
    end
    train, evalu = partition(rows, holdout.fraction_train,
                             shuffle=holdout.shuffle, rng=rng)
    return [(train, evalu),]
end
```


### API

```@docs
evaluate!
```

```@docs
evaluate
```
