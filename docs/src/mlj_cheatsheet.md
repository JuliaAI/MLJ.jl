# MLJ Cheatsheet

#### Tasks 

`task = supervised(data=…, types=…, target=…, ignore=…, is_probabilistic=false, verbosity=1)`
 
`task = unsupervised(data=…, types=…, ignore=…, verbosity=1)`   


`task.X` for inputs (a vector or table)

`task.y` for target (a vector)

`shuffle(task)`

`task[1:10]`

`nrows(task)`

`models(task)` is all models adapted to the task


#### Data and scitypes

`Finite{N}` has subtypes `Multiclass{N}` and `OrderedFactor{N}`.
 
`Infinite` has subtypes `Continuous` and `Count`

`scitype(x)` is the scientific type of `x`
 
Use `schema(X)` to get the column scitypes of a table `X`
 
`coerce(Multiclass, y)` attempts coercion of all elements of `y` into scientific type `Multiclass`

`coerce(Dict(:x1 => Continuous, :x2 => OrderedFactor), X)` to coerce columns `:x1` and `:x2` of `X`.


#### Machine construction

Supervised case:
 
`model = KNNRegressor(K=1)` and `mach = machine(model, X, y)` or `mach = machine(model, task)`
 
Unsupervised case:

`model = OneHotEncoder()` and `mach = machine(model, X)` or `mach = machine(model, task)`

#### Inspect all model hyperparameters, even nested ones:

`params(model)`

#### Fitting

`fit!(mach, rows=:, verbosity=1, force=false)`


#### Inspecting results of fitting

`fitted_params(mach)` for learned parameters

`report(mach)` for other results (e.g. feature rankings)


#### Prediction

Supervised case: `predict(mach, Xnew)` or `predict(mach, rows=:)` or `predict(mach, task)`
  
Similarly, for probabilistic models: `predict_mode`, `predict_mean` and `predict_median`.

Unsupervised case: `transform(mach, rows=:)` or `inverse_transform(mach, rows)`, etc.

#### Resampling strategies
    
`Holdout(fraction_train=…, shuffle=false)` for simple holdout
 
`CV(nfolds=6, parallel=true, shuffle=false)` for cross-validation

or a list of pairs of row indices:

`[(train1, eval1), (train2, eval2), ... (traink, evalk)]` 


#### Performance estimation

`evaluate(model, X, y, resampling=CV(), measure=..., operation=predict, weights=..., verbosity=1)`
`evaluate!(mach, resampling=Holdout(), measure=..., operation=predict, weights=..., verbosity=1)`

#### Ranges for tuning

If `r = range(KNNRegressor(), :K, lower=1, upper = 20, scale=:log)` then `iterator(r, 6) = [1, 2, 3, 6, 11, 20]`

Non-numeric ranges: `r = range(model, :parameter, values=…)`.

Nested ranges: Use dot syntax, as in `r = range(EnsembleModel(atom=tree), :(atom.max_depth), ...)`

#### Tuning strategies

`Grid(resolution=10, parallel=true)` for grid search


#### Tuning model wrapper

`tuned_model = TunedModel(model=…, tuning=Grid(), resampling=Holdout(), measure=…, operation=predict, ranges=…, minimize=true, full_report=true)`

Use `params(model)` to get pattern to match in specifying `nested_ranges`.


#### Learning curves

`curve = learning_curve!(mach, resolution=30, resampling=Holdout(), measure=…, operation=predict, range=…,, n=1)`
 

If using Plots.jl:


`plot(curve.parameter_values, curve.measurements, xlab=curve.parameter_name, xscale=curve.parameter_scale)` 


#### Ensemble model wrapper

`EnsembleModel(atom=…, weights=Float64[], bagging_fraction=0.8, rng=GLOBAL_RNG, n=100, parallel=true, out_of_bag_measure=[])`


#### Built-in measures

`l1`, `l2`, `mav`, `rms`, `rmsl`, `rmslp1`, `rmsp`, `misclassification_rate`, `cross_entropy`


#### Transformers 

Built-ins include: `Standardizer`, `OneHotEncoder`, `UnivariateBoxCoxTransformer`, `FeatureSelector`, `UnivariateStandardizer`

Externals include: `PCA` (in MultivariateStats), `KMeans`, `KMedoids` (in Clustering).

Full list: do `models(m -> !m[:is_supervised])`
