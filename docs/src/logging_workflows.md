# Logging Workflows

Currently the following workflows can log their outcomes to an external machine learning
tracking platform, such as [MLflow](https://mlflow.org) (see [MLflow](@ref) below):

- Estimating model performance using [`evaluate`](@ref)/[`evaluate!`](@ref).

- Model tuning, using the `TunedModel` wrapper, as described under [Tuning Models](@ref).

To enable logging one must create a `logger` object for the relevant tracking platform,
and either:

- Provide `logger` as an explicit keyword argument in the workflow, as in `evaluate(...;
  logger=...)` or `TunedModel(...; logger=...)`; or

- Set a global default logger with the call [`default_logger(logger)`](@ref).

MLJ logging examples are given in the [MLJFlow.jl](https://github.com/JuliaAI/MLJFlow.jl)
documentation.
x

### Supported tracking platforms

- [MLflow](@ref) (natively supported: MLJ re-exports `MLJFlow.Logger(...)`)


!!! warning

    MLJFlow.jl is a new package still under active development and should be regarded as experimental. At this time, breaking changes to MLJFlow.jl will not necessarily trigger new breaking releases of MLJ.jl.

