# Simple User Defined Models

To quickly implement a new supervised model in MLJ, it suffices to:

- Define a `mutable struct` to store hyperparameters. This is either a subtype
  of `Probabilistic` or `Deterministic`, depending on
  whether probabilistic or ordinary point predictions are
  intended. This `struct` is the *model*.
  
- Define a `fit` method, dispatched on the model, returning
  learned parameters, also known as the *fit-result*.
  
- Define a `predict` method, dispatched on the model, and passed the
  fit-result, to return predictions on new patterns.
  
In the examples below, the training input `X` of `fit`, and the new
input `Xnew` passed to `predict`, are tables. Each training target `y`
is a `AbstractVector`.

The predicitions returned by `predict` have the same form as `y` for
deterministic models, but are `Vector`s of distibutions for
probabilistic models.

For your models to implement an optional `update` method, to buy into the
MLJ logging protocol, or report training statistics or other
model-specific functionality, a `fit` method with a slightly different
signature and output is required. To enable checks of the scientific
type of data passed to your model by MLJ's meta-algorithms, one needs
to implement additional traits. A `clean!` method can be defined to
check that hyperparameter values are within normal ranges. For details, see
[Adding Models for General Use](adding_models_for_general_use.md).

For an unsupervised model, implement `transform` and, optionally,
`inverse_transform` using the same signature at `predict below.


### A simple deterministic regressor

Here's a quick-and-dirty implementation of a ridge regressor with no intercept:

````julia
import MLJBase
using LinearAlgebra

mutable struct MyRegressor <: MLJBase.Deterministic
    lambda::Float64
end
MyRegressor(; lambda=0.1) = MyRegressor(lambda)

# fit returns coefficients minimizing a penalized rms loss function:
function MLJBase.fit(model::MyRegressor, X, y)
    x = MLJBase.matrix(X)                     # convert table to matrix
    fitresult = (x'x - model.lambda*I)\(x'y)  # the coefficients
    return fitresult
end

# predict uses coefficients to make new prediction:
MLJBase.predict(model::MyRegressor, fitresult, Xnew) = MLJBase.matrix(Xnew)fitresult
````

After loading this code, all MLJ's basic meta-algorithms can be applied to `MyRegressor`:

````julia
julia> using MLJ
julia> task = load_boston()
julia> model = MyRegressor(lambda=1.0)
julia> regressor = machine(model, task)
julia> evaluate!(regressor, resampling=CV(), measure=rms) |> mean
7.434221318358656

````

### A simple probabilistic classifier

The following probabilistic model simply fits a probability
distribution to the `MultiClass` training target (i.e., ignores `X`)
and returns this pdf for any new pattern:

````julia
import MLJBase
import Distributions

struct MyClassifier <: MLJBase.Probabilistic
end

# `fit` ignores the inputs X and returns the training target y
# probability distribution:
function MLJBase.fit(model::MyClassifier, X, y)
    fitresult = Distributions.fit(MLJBase.UnivariateFinite, y)
    return fitresult
end

# `predict` retunrs the passed fitresult (pdf) for all new patterns:
MLJBase.predict(model::MyClassifier, fitresult, Xnew) = 
    [fitresult for r in 1:nrows(Xnew)]
````

For more details on the `UnivariateFinite` distribution, query
`MLJBase.UnivariateFinite`.
