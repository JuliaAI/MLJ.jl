# # Code used in "Using MLJ. Lesson 2: Model Composition"

# Ensure your current directory contains this file and change `@__DIR__` to `pwd()` below
# to activate a Julia package environment that has been tested with this notebook:

using Pkg
Pkg.activate(@__DIR__)
# Pkg.activate(pwd())
Pkg.instantiate()

# Presentationi starts with evaluation of this cell:

Pkg.status()

#-

using MLJ

# load some model code:
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels

# load some data and inspect schema:
data = load_reduced_ames();
schema(data)

# horizontally split with observation shuffling:
y, X = unpack(data, ==(:target); rng=123);
schema(X)

# defined a pipeline model:
pipe = ContinuousEncoder() |> Standardizer() |> RidgeRegressor()

# accessing a nested hyperparameter:
pipe.ridge_regressor.fit_intercept

# changing it:
pipe.ridge_regressor.fit_intercept = false

# evaluate the pipeline:
evaluate(pipe, X, y; resampling=CV(nfolds=4, rng=123), repeats=2, measure=mav)

# look at the target:
@show mean(y) std(y)

# wrap in target normalization:
norm_pipe = TransformedTargetModel(pipe, transformer=Standardizer())

# evaluate performance:
evaluate(norm_pipe, X, y; resampling=CV(nfolds=4, rng=123), repeats=2, measure=mav)

# horizontally split with observation shuffling:
y, X = unpack(data, ==(:target); rng=123)
schema(X)
