# Simple User Defined Models


To quickly implement a new supervised model in MLJ, it suffices to:

- Define a `mutable struct` to store hyperparameters. This is either a subtype
  of `Probabilistic` or `Deterministic`, depending on
  whether probabilistic or ordinary point predictions are
  intended. This `struct` is the *model*.

- Define a `fit` method, dispatched on the model, returning
  learned parameters, also known as the *fitresult*.

- Define a `predict` method, dispatched on the model, and the
  fitresult, to return predictions on new patterns.

In the examples below, the training input `X` of `fit`, and the new
input `Xnew` passed to `predict`, are tables. Each training target `y`
is a `AbstractVector`.

The predictions returned by `predict` have the same form as `y` for
deterministic models, but are `Vector`s of distributions for
probabilistic models.

Advanced model functionality not addressed here includes: (i) optional
`update` method to avoid redundant calculations when calling `fit!` on
machines a second time; (ii) reporting extra training-related
statistics; (iii) exposing model-specific functionality; (iv) checking
the scientific type of data passed to your model in `machine`
construction; and (iv) checking validity of hyperparameter values. All
this is described in [Adding Models for General
Use](adding_models_for_general_use.md).

For an unsupervised model, implement `transform` and, optionally,
`inverse_transform` using the same signature at `predict` below.

## A simple deterministic regressor

Here's a quick-and-dirty implementation of a ridge regressor with no intercept:

```julia
import MLJBase
using LinearAlgebra

mutable struct MyRegressor <: MLJBase.Deterministic
    lambda::Float64
end
MyRegressor(; lambda=0.1) = MyRegressor(lambda)

# fit returns coefficients minimizing a penalized rms loss function:
function MLJBase.fit(model::MyRegressor, verbosity, X, y)
    x = MLJBase.matrix(X)                     # convert table to matrix
    fitresult = (x'x + model.lambda*I)\(x'y)  # the coefficients
    cache=nothing
    report=nothing
    return fitresult, cache, report
end

# predict uses coefficients to make new prediction:
MLJBase.predict(::MyRegressor, fitresult, Xnew) = MLJBase.matrix(Xnew) * fitresult
```

``` @setup regressor_example
using MLJ
import MLJBase
using LinearAlgebra
MLJBase.color_off()
mutable struct MyRegressor <: MLJBase.Deterministic
    lambda::Float64
end
MyRegressor(; lambda=0.1) = MyRegressor(lambda)
function MLJBase.fit(model::MyRegressor, verbosity, X, y)
    x = MLJBase.matrix(X)
    fitresult = (x'x + model.lambda*I)\(x'y)
    cache=nothing
    report=nothing
    return fitresult, cache, report
end
MLJBase.predict(::MyRegressor, fitresult, Xnew) = MLJBase.matrix(Xnew) * fitresult
```

After loading this code, all MLJ's basic meta-algorithms can be applied to `MyRegressor`:

```@repl regressor_example
X, y = @load_boston;
model = MyRegressor(lambda=1.0)
regressor = machine(model, X, y)
evaluate!(regressor, resampling=CV(), measure=rms, verbosity=0)

```

## A simple probabilistic classifier

The following probabilistic model simply fits a probability
distribution to the `MultiClass` training target (i.e., ignores `X`)
and returns this pdf for any new pattern:

```julia
import MLJBase
import Distributions

struct MyClassifier <: MLJBase.Probabilistic
end

# `fit` ignores the inputs X and returns the training target y
# probability distribution:
function MLJBase.fit(model::MyClassifier, verbosity, X, y)
    fitresult = Distributions.fit(MLJBase.UnivariateFinite, y)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

# `predict` returns the passed fitresult (pdf) for all new patterns:
MLJBase.predict(model::MyClassifier, fitresult, Xnew) =
    [fitresult for r in 1:nrows(Xnew)]
```

```julia
julia> X, y = @load_iris
julia> mach = fit!(machine(MyClassifier(), X, y))
julia> predict(mach, selectrows(X, 1:2))
2-element Array{UnivariateFinite{String,UInt32,Float64},1}:
 UnivariateFinite(setosa=>0.333, versicolor=>0.333, virginica=>0.333)
 UnivariateFinite(setosa=>0.333, versicolor=>0.333, virginica=>0.333)
```
