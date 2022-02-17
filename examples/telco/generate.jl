# Execute this julia file to generate the notebooks from ../notebook.jl

joinpath(@__DIR__, "..", "generate.jl") |> include
generate(@__DIR__, pluto=false, execute=false)

# Execution has been failing with a an issue with deserializing the
# final model.
# Executing the notebook in Juptyer is fine however.

#https://discourse.julialang.org/t/execution-of-notebook-in-literate-jl-not-working-but-notebook-executes-fine-in-jupyter-serialization-issue/76387/4

