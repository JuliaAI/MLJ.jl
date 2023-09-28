# Logging Workflows

## MLflow integration

[MLflow](https://mlflow.org) is a popular, language-agnostic, tool for externally logging
the outcomes of machine learning experiments, including those carried out using MLJ.

MLJ logging examples are given in the [MLJFlow.jl](https://github.com/JuliaAI/MLJFlow.jl)
documentation. MLJ includes and re-exports all the methods of MLJFlow.jl, so there is no
need to import MLJFlow.jl if `using MLJ`.

!!! warning

    MLJFlow.jl is a new package still under active development and should be regarded as experimental. At this time, breaking changes to MLJFlow.jl will not necessarily trigger new breaking releases of MLJ.jl.

