# Model Search

MLJ has a model registry, allowing the user to search models and their
properties, without loading all the packages containing model code. In
turn, this allows one to efficiently find all models solving a given
machine learning task. The task itself is specified with the help of
the `matching` method, and the search executed with the `models`
methods, as detailed below. 

A table of all models is also given at [List of Supported
Models](@ref).


## Model metadata

*Terminology.* In this section the word "model" refers to a metadata
entry in the model registry, as opposed to an actual model `struct`
that such an entry represents. One can obtain such an entry with the
`info` command:

```@setup tokai
using MLJ
MLJ.color_off()
```

```@repl tokai
info("PCA")
```

So a "model" in the present context is just a named tuple containing
metadata, and not an actual model type or instance. If two models with
the same name occur in different packages, the package name must be
specified, as in `info("LinearRegressor", pkg="GLM")`.


## General model queries

We list all models (named tuples) using `models()`, and list the models for which code is  already loaded with `localmodels()`:

```@repl tokai
localmodels()
localmodels()[2]
```

One can search for models containing specified strings or regular expressions in their `docstring` attributes, as in

```@repl tokai 
models("forest")
```

or by specifying a filter (`Bool`-valued function):

```@repl tokai
filter(model) = model.is_supervised &&
                model.input_scitype >: MLJ.Table(Continuous) &&
                model.target_scitype >: AbstractVector{<:Multiclass{3}} &&
                model.prediction_type == :deterministic
models(filter)
```

Multiple test arguments may be passed to `models`, which are applied
conjunctively.


## Matching models to data

Common searches are streamlined with the help of the `matching`
command, defined as follows:

- `matching(model, X, y) == true` exactly when `model` is supervised
   and admits inputs and targets with the scientific types of `X` and
   `y`, respectively

- `matching(model, X) == true` exactly when `model` is unsupervised
   and admits inputs with the scientific types of `X`.

So, to search for all supervised probabilistic models handling input
`X` and target `y`, one can define the testing function `task` by

```julia
task(model) = matching(model, X, y) && model.prediction_type == :probabilistic
```

And execute the search with

```julia
models(task)
```

Also defined are `Bool`-valued callable objects `matching(model)`,
`matching(X, y)` and `matching(X)`, with obvious behaviour. For example,
`matching(X, y)(model) = matching(model, X, y)`.

So, to search for all models compatible with input `X` and target `y`,
for example, one executes

```julia
models(matching(X, y))
```

while the preceding search can also be written

```julia
models() do model
    matching(model, X, y) &&
    model.prediction_type == :probabilistic
end
```

## API

```@docs
models
localmodels
```
