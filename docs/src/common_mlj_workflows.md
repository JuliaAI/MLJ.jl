# Common MLJ Workflows

## Data ingestion

```@setup workflows
# to avoid RDatasets as a doc dependency:
using MLJ; color_off()
import DataFrames
channing = (Sex = rand(["Male","Female"], 462),
            Entry = rand(Int, 462),
            Exit = rand(Int, 462),
            Time = rand(Int, 462),
            Cens = rand(Int, 462)) |> DataFrames.DataFrame
coerce!(channing, :Sex => Multiclass)
```

```julia
import RDatasets
channing = RDatasets.dataset("boot", "channing")

julia> first(channing, 4)
4×5 DataFrame
 Row │ Sex   Entry  Exit   Time   Cens
     │ Cat…  Int32  Int32  Int32  Int32
─────┼──────────────────────────────────
   1 │ Male    782    909    127      1
   2 │ Male   1020   1128    108      1
   3 │ Male    856    969    113      1
   4 │ Male    915    957     42      1
```

Inspecting metadata, including column scientific types:

```@example workflows
schema(channing)
```

Unpacking data and shuffling rows:

```@example workflows
y, X =  unpack(channing,
               ==(:Exit),            # y is the :Exit column
               !=(:Time));           # X is the rest, except :Time
nothing # hide
```

*Note:* Before julia 1.2, replace `!=(:Time)` with `col -> col != :Time`.

```julia workflows
julia> y[1:4]
4-element Vector{Int32}:
  909
 1128
  969
  957
```

Fixing wrong scientfic types in `X`:

```julia
X = coerce(X, :Exit=>Continuous, :Entry=>Continuous, :Cens=>Multiclass)

julia> schema(X)
┌─────────┬─────────────────────────────────┬───────────────┐
│ _.names │ _.types                         │ _.scitypes    │
├─────────┼─────────────────────────────────┼───────────────┤
│ Sex     │ CategoricalValue{String, UInt8} │ Multiclass{2} │
│ Entry   │ Float64                         │ Continuous    │
│ Cens    │ CategoricalValue{Int32, UInt32} │ Multiclass{2} │
└─────────┴─────────────────────────────────┴───────────────┘
_.nrows = 462
```


Loading a built-in supervised dataset:

```@example workflows
table = load_iris()
schema(table)
```

Loading a built-in data set already split into `X` and `y`:

```@example workflows
X, y = @load_iris;
selectrows(X, 1:4) # selectrows works for any Tables.jl table
```

```@example workflows
y[1:4]
```

## Model search

*Reference:*   [Model Search](model_search.md)

Searching for a supervised model:

```@example workflows
X, y = @load_boston
models(matching(X, y))
```

```@example workflows
models(matching(X, y))[6]
```

More refined searches:

```@example workflows
models() do model
    matching(model, X, y) &&
    model.prediction_type == :deterministic &&
    model.is_pure_julia
end
```

Searching for an unsupervised model:

```@example workflows
models(matching(X))
```

Getting the metadata entry for a given model type:

```@example workflows
info("PCA")
info("RidgeRegressor", pkg="MultivariateStats") # a model type in multiple packages
```

## Instantiating a model

*Reference:*   [Getting Started](@ref), [Loading Model Code](@ref)

```@example workflows
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree(min_samples_split=5, max_depth=4)
```

or

```@julia
tree = (@load DecisionTreeClassifier)()
tree.min_samples_split = 5
tree.max_depth = 4
```

## Evaluating a model

*Reference:*   [Evaluating Model Performance](evaluating_model_performance.md)


```@example workflows
X, y = @load_boston
KNN = @load KNNRegressor
knn = KNN()
evaluate(knn, X, y, resampling=CV(nfolds=5), measure=[RootMeanSquaredError(), MeanAbsoluteError()])
```

Note `RootMeanSquaredError()` has alias `rms` and `MeanAbsoluteError()` has alias `mae`.

Do `measures()` to list all losses and scores and their aliases.


##  Basic fit/evaluate/predict by hand:

*Reference:*   [Getting Started](index.md), [Machines](machines.md),
[Evaluating Model Performance](evaluating_model_performance.md), [Performance Measures](performance_measures.md)

```@example workflows
crabs = load_crabs() |> DataFrames.DataFrame
schema(crabs)
```

```@example workflows
y, X = unpack(crabs, ==(:sp), !in([:index, :sex]); rng=123)


Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree(max_depth=2) # hide
```

Bind the model and data together in a *machine* , which will
additionally store the learned parameters (*fitresults*) when fit:

```@example workflows
mach = machine(tree, X, y)
```

Split row indices into training and evaluation rows:

```@example workflows
train, test = partition(eachindex(y), 0.7); # 70:30 split
```

Fit on train and evaluate on test:

```@example workflows
fit!(mach, rows=train)
yhat = predict(mach, X[test,:])
mean(LogLoss(tol=1e-4)(yhat, y[test]))
```

Note `LogLoss()` has aliases `log_loss` and `cross_entropy`.

Run `measures()` to list all losses and scores and their aliases ("instances").

Predict on new data:

```@example workflows
Xnew = (FL = rand(3), RW = rand(3), CL = rand(3), CW = rand(3), BD =rand(3))
predict(mach, Xnew)      # a vector of distributions
```

```@example workflows
predict_mode(mach, Xnew) # a vector of point-predictions
```

## More performance evaluation examples

Evaluating model + data directly:

```@example workflows
evaluate(tree, X, y,
         resampling=Holdout(fraction_train=0.7, shuffle=true, rng=1234),
         measure=[LogLoss(), Accuracy()])
```

If a machine is already defined, as above:

```@example workflows
evaluate!(mach,
          resampling=Holdout(fraction_train=0.7, shuffle=true, rng=1234),
          measure=[LogLoss(), Accuracy()])
```

Using cross-validation:

```@example workflows
evaluate!(mach, resampling=CV(nfolds=5, shuffle=true, rng=1234),
          measure=[LogLoss(), Accuracy()])
```

With user-specified train/test pairs of row indices:

```@example workflows
f1, f2, f3 = 1:13, 14:26, 27:36
pairs = [(f1, vcat(f2, f3)), (f2, vcat(f3, f1)), (f3, vcat(f1, f2))];
evaluate!(mach,
          resampling=pairs,
          measure=[LogLoss(), Accuracy()])
```

Changing a hyperparameter and re-evaluating:

```@example workflows
tree.max_depth = 3
evaluate!(mach,
          resampling=CV(nfolds=5, shuffle=true, rng=1234),
          measure=[LogLoss(), Accuracy()])
```

##  Inspecting training results

Fit a ordinary least square model to some synthetic data:

```@example workflows
x1 = rand(100)
x2 = rand(100)

X = (x1=x1, x2=x2)
y = x1 - 2x2 + 0.1*rand(100);

OLS = @load LinearRegressor pkg=GLM
ols = OLS()
mach =  machine(ols, X, y) |> fit!
```

Get a named tuple representing the learned parameters,
human-readable if appropriate:

```@example workflows
fitted_params(mach)
```

Get other training-related information:

```@example workflows
report(mach)
```

##  Basic fit/transform for unsupervised models

Load data:

```@example workflows
X, y = @load_iris
train, test = partition(eachindex(y), 0.97, shuffle=true, rng=123)
```

Instantiate and fit the model/machine:

```@example workflows
PCA = @load PCA
pca = PCA(maxoutdim=2)
mach = machine(pca, X)
fit!(mach, rows=train)
```

Transform selected data bound to the machine:

```@example workflows
transform(mach, rows=test);
```

Transform new data:

```@example workflows
Xnew = (sepal_length=rand(3), sepal_width=rand(3),
        petal_length=rand(3), petal_width=rand(3));
transform(mach, Xnew)
```

##  Inverting learned transformations

```@example workflows
y = rand(100);
stand = Standardizer()
mach = machine(stand, y)
fit!(mach)
z = transform(mach, y);
@assert inverse_transform(mach, z) ≈ y # true
```

## Nested hyperparameter tuning

*Reference:*   [Tuning Models](tuning_models.md)

```@example workflows
X, y = @load_iris; nothing # hide
```

Define a model with nested hyperparameters:

```@example workflows
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()
forest = EnsembleModel(atom=tree, n=300)
```

Define ranges for hyperparameters to be tuned:

```@example workflows
r1 = range(forest, :bagging_fraction, lower=0.5, upper=1.0, scale=:log10)
```

```@example workflows
r2 = range(forest, :(atom.n_subfeatures), lower=1, upper=4) # nested
```

Wrap the model in a tuning strategy:

```@example workflows
tuned_forest = TunedModel(model=forest,
                          tuning=Grid(resolution=12),
                          resampling=CV(nfolds=6),
                          ranges=[r1, r2],
                          measure=BrierScore())
```

Bound the wrapped model to data:

```@example workflows
mach = machine(tuned_forest, X, y)
```

Fitting the resultant machine optimizes the hyperparameters specified
in `range`, using the specified `tuning` and `resampling` strategies
and performance `measure` (possibly a vector of measures), and
retrains on all data bound to the machine:

```@example workflows
fit!(mach)
```

Inspecting the optimal model:

```@example workflows
F = fitted_params(mach)
```

```@example workflows
F.best_model
```

Inspecting details of tuning procedure:

```@example workflows
r = report(mach);
keys(r)
```

```@example workflows
r.history[[1,end]]
```

Visualizing these results:

```julia
using Plots
plot(mach)
```

![](img/workflows_tuning_plot.png)

Predicting on new data using the optimized model:

```@example workflows
predict(mach, Xnew)
```

## Constructing a linear pipeline

*Reference:*   [Composing Models](composing_models.md)

Constructing a linear (unbranching) pipeline with a *learned* target
transformation/inverse transformation:

```@example workflows
X, y = @load_reduced_ames
KNN = @load KNNRegressor
pipe = @pipeline(X -> coerce(X, :age=>Continuous),
                 OneHotEncoder,
                 KNN(K=3),
                 target = Standardizer)
```

Evaluating the pipeline (just as you would any other model):

```@example workflows
pipe.knn_regressor.K = 2
pipe.one_hot_encoder.drop_last = true
evaluate(pipe, X, y, resampling=Holdout(), measure=RootMeanSquaredError(), verbosity=2)
```

Inspecting the learned parameters in a pipeline:

```@example workflows
mach = machine(pipe, X, y) |> fit!
F = fitted_params(mach)
F.one_hot_encoder
```

Constructing a linear (unbranching) pipeline with a *static* (unlearned)
target transformation/inverse transformation:

```@example workflows
Tree = @load DecisionTreeRegressor pkg=DecisionTree
pipe2 = @pipeline(X -> coerce(X, :age=>Continuous),
                  OneHotEncoder,
                  Tree(max_depth=4),
                  target = y -> log.(y),
                  inverse = z -> exp.(z))
```

## Creating a homogeneous ensemble of models

*Reference:* [Homogeneous Ensembles](homogeneous_ensembles.md)

```@example workflows
X, y = @load_iris
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree()
forest = EnsembleModel(atom=tree, bagging_fraction=0.8, n=300)
mach = machine(forest, X, y)
evaluate!(mach, measure=LogLoss())
```

## Performance curves

Generate a plot of performance, as a function of some hyperparameter
(building on the preceding example)

Single performance curve:

```@example workflows
r = range(forest, :n, lower=1, upper=1000, scale=:log10)
curve = learning_curve(mach,
                       range=r,
                       resampling=Holdout(),
                       resolution=50,
                       measure=LogLoss(),
                       verbosity=0)
```

```julia
using Plots
plot(curve.parameter_values, curve.measurements, xlab=curve.parameter_name, xscale=curve.parameter_scale)
```

![](img/workflows_learning_curve.png)

Multiple curves:

```@example workflows
curve = learning_curve(mach,
                       range=r,
                       resampling=Holdout(),
                       measure=LogLoss(),
                       resolution=50,
                       rng_name=:rng,
                       rngs=4,
                       verbosity=0)
```

```julia
plot(curve.parameter_values, curve.measurements,
xlab=curve.parameter_name, xscale=curve.parameter_scale)
```

![](img/workflows_learning_curves.png)
