# # Code used in "Using MLJ. Lesson 3: Model Tuning"

# Ensure your current directory contains this file and change `@__DIR__` to `pwd()` below
# to activate a Julia package environment that has been tested with this notebook:

using Pkg
Pkg.activate(@__DIR__)
# Pkg.activate(pwd())
Pkg.instantiate()

# Presentation starts with evaluation of this cell:

Pkg.status()

# ## PART I. LEARNING CURVES

using MLJ, Plots
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree

X, price = @load_reduced_ames;
schema(X)
y = price/100_000; # price in multiples of $100,000
first(y, 5)

# define a model:
model = RandomForestRegressor(
    rng=123,
    n_subfeatures=1,
)
pipe =  ContinuousEncoder() |> model
baseline = ContinuousEncoder() |>  ConstantRegressor()

# sanity check:
options = (; resampling=CV(nfolds=3), measure = mae, acceleration=CPUThreads())
e0 = evaluate(pipe, X, y; options...)
ebase = evaluate(baseline, X, y; options...)

# inspect hyper-parameters:
pipe

# define a 1D hyper-parameter range:
r = range(pipe, :(random_forest_regressor.n_trees), lower=10, upper=500)

# same arguments as `evaluate` except but with extra `range` option:
curve = learning_curve(
    pipe, X, y;
    range=r,
    resampling=Holdout(fraction_train=0.8),
    measure=mae,
)
plot(curve.parameter_values, curve.measurements)

pipe.random_forest_regressor.n_trees = 250


# ## PART 2. THE `TunedModel` WRAPPER

r1 = range(pipe, :(random_forest_regressor.n_subfeatures), lower=1, upper=12)
r2 = range(pipe, :(random_forest_regressor.min_samples_split), lower=2, upper=10)

tuned_pipe = TunedModel(
    pipe,
    range=[r1, r2],
    tuning=Grid(goal=40),
    resampling=CV(nfolds=4),
    measures=mae,
)

mach = machine(tuned_pipe, X, y) |> fit!
plot(mach)
keys(report(mach))
report(mach).best_model
report(mach).best_history_entry.evaluation

tuned_pipe = TunedModel(
    pipe,
    range=[r1, r2],
    tuning=RandomSearch(rng=123),
    resampling=CV(nfolds=4),
    measures=mae,
    n=40,
)

mach = machine(tuned_pipe, X, y) |> fit!
plot(mach)

# in-sample predictions based on optimized parameters and retraining on *all* data:
predict(mach, X)

e1 = evaluate(tuned_pipe, X, y; options...)

@show ebase e0 e1;

# Can use tuned_model to compare models of different types:

tuned_model = TunedModel(models=[pipe, baseline], resampling=CV(nfolds=4), measure=l1)
mach = machine(tuned_model, X, y) |> fit!
report(mach).best_model
