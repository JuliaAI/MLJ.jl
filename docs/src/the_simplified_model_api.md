# The Simplified Model API

To quickly implement a new supervised model in MLJ, it suffices to:

- Define a `mutable struct` to store hyperparameters. This is either a subtype
  of `Probabilistic{Any}` or `Deterministic{Any}`, depending on
  whether probabilistic or ordinary point predictions are
  intended. This `struct` is the *model*.
  
- Define a `fit` method, dispatched on the model, returning
  learned parameters, also known as the *fit-result*.
  
- Define a `predict` method, dispatched on the model, and passed the
  fit-result, to return predictions on new patterns.
  
In the examples below, the training input `X` of `fit`, and the new
input `Xnew` passed to `predict`, are tables implementing the
[Tables.jl](https://github.com/JuliaData/Tables.jl) interface. Each
training target `y` is a `Vector` for regressors, and a 
`CategoricalVector` for classifiers. The predicitions returned by
`predict` have the same form as `y` for deterministic models, but are
`Vector`s of distibutions for probabilistic models.

For your models to implement an optional `update` method, buy into the
MLJ logging protocol, or report training statistics or other
model-specific functionality, a `fit` method with a slightly different
signature and output is required. To enable checks of the scientific
type of data passed to your model by MLJ's meta-algorithms, one needs
to implement additional traits. A `clean!` method can be defined to
check that hyperparameter values are within normal ranges. For details, see
[Adding New Models](adding_new_models.md).


### A simple deterministic regressor

Here's a quick-and-dirty implementation of a ridge regressor with no intercept:

````julia
import MLJBase
using LinearAlgebra

mutable struct MyRegressor <: MLJBase.Deterministic{Any}
    lambda::Float64
end

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
julia> model = MyRegressor(1.0)
julia> regressor = machine(model, task)
julia> evaluate!(regressor, resampling=CV(), measure=rms) |> mean
7.434221318358656

````

### A simple probabilistic classifier

The following probabilistic model simply fits a probability
distribution to a `MultiClass`training target and returns this pdf for
any new pattern:

````julia
import MLJBase
import Tables
import Distributions

struct MyClassifier <: MLJBase.Probabilistic{Any}
end

# `fit` ignores the inputs X and returns the training target y
# probability distribution:
function MLJBase.fit(model::MyClassifier, X, y)
    fitresult = Distributions.fit(MLJBase.UnivariateNominal, y)
    return fitresult
end

# `predict` retunrs the passed fitresult (pdf) for all new patterns:
function MLJBase.predict(model::MyClassifier, fitresult, Xnew)
    row_iterator = Tables.rows(Xnew)
    return [fitresult for r in row_iterator]
end
````

For more details on `UnivariateNominal`, query `MLJBase.UnivariateNominal`. 
