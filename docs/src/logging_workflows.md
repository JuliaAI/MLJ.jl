!!! note

	Starting with MLJ 0.23.0, MLJFlow.jl methods are not immediately available, as the package has been removed as a direct dependency. Previous behaviour will require `using MLJFlow`. See also the "Warning" below.

# Logging Workflows

In principle, the following workflows can log their outcomes to an external machine learning
tracking platform, such as [mlflow](https://mlflow.org):

- Estimating model performance using [`evaluate`](@ref)/[`evaluate!`](@ref).

- Model tuning, using the `TunedModel` wrapper, as described under [Tuning Models](@ref).

To enable logging one must create a `logger` object for the relevant tracking platform,
and either:

- Provide `logger` as an explicit keyword argument in the workflow, as in `evaluate(...;
  logger=...)` or `TunedModel(...; logger=...)`; or

- Set a global default logger with the call [`default_logger(logger)`](@ref).

MLJ logging examples are given in the [MLJFlow.jl](https://github.com/JuliaAI/MLJFlow.jl)
documentation.


### Supported tracking platforms

!!! warning

    Due to issues with the mlflow REST API, the current model for MLJ-mlflow integration is being reassessed. Use the existing tools at your own risk.

- To use [mlflow](https://mlflow.org) with MLJ you will need to add MLJFlow to your
  package environment and call `using MLJFlow`. You additionally need to install
  mlflow itself, and separately launch an mlflow service; see the [mlflow
  docs](https://mlflow.org) on how to do this. The service can immediately be wrapped to
  create a `logger` object, as demonstrated in the [MLJFlow.jl
  documentation](https://github.com/JuliaAI/MLJFlow.jl).
