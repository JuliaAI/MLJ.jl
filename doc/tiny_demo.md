### Basic training and testing

Let's load data and define train and test rows:


```julia
using MLJ
using DataFrames

X, y = X_and_y(load_boston())

train, test = partition(eachindex(y), 0.70); # 70:10:10 split
```

A *model* is a container for hyperparameters:


```julia
knn_model=KNNRegressor(K=10)
```

    # KNNRegressor @ 6…89: 
    K                       =>   10
    metric                  =>   euclidean (generic function with 1 method)
    kernel                  =>   reciprocal (generic function with 1 method)
    
Wrapping the model in data creates a *machine* which will store training outcomes (called *fit-results*):


```julia
knn = machine(knn_model, X, y)
```

    # Machine{KNNRegressor} @ 1…96: 
    model                   =>   KNNRegressor @ 6…89
    fitresult               =>   (undefined)
    cache                   =>   (undefined)
    args                    =>   (omitted Tuple{DataFrame,Array{Float64,1}} of length 2)
    report                  =>   empty Dict{Symbol,Any}
    rows                    =>   (undefined)
    
Training on the training rows and evaluating on the test rows:

```julia
fit!(knn, rows=train)
yhat = predict(knn, X[test,:])
rms(y[test], yhat)
```

    ┌ Info: Training Machine{KNNRegressor} @ 1…96.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:69

    8.090639098853249


Changing a hyperparameter and re-evaluating:

```julia
knn_model.K = 20
fit!(knn)
yhat = predict(knn, X[test,:])
rms(y[test], yhat)
```

    ┌ Info: Training Machine{KNNRegressor} @ 1…96.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:69

    6.253838532302258


### Systematic tuning as a model wrapper

A simple example of a composite model is a homogeneous ensemble. Here's a bagged ensemble model for 20 K-nearest neighbour regressors:

```julia
ensemble_model = EnsembleModel(atom=knn_model, n=20) 
```

    # DeterministicEnsembleModel @ 1…59: 
    atom                    =>   KNNRegressor @ 6…89
    weights                 =>   0-element Array{Float64,1}
    bagging_fraction        =>   0.8
    rng_seed                =>   0
    n                       =>   20
    parallel                =>   true
    
Let's simultaneously tune the ensemble's `bagging_fraction` and the K-nearest neighbour hyperparameter `K`. Since one of these models is a field of the other, we have nested hyperparameters:

```julia
params(ensemble_model)
```

    Params(:atom => Params(:K => 20, :metric => MLJ.KNN.euclidean, :kernel => MLJ.KNN.reciprocal), :weights => Float64[], :bagging_fraction => 0.8, :rng_seed => 0, :n => 20, :parallel => true)

To define a tuning grid, we construct ranges for the two parameters and collate these ranges following the same pattern above (omitting parameters that don't change):

```julia
B_range = range(ensemble_model, :bagging_fraction, lower= 0.5, upper=1.0, scale = :linear)
K_range = range(knn_model, :K, lower=1, upper=100, scale=:log10)
nested_ranges = Params(:atom => Params(:K => K_range), :bagging_fraction => B_range)
```

    Params(:atom => Params(:K => NumericRange @ 1…22), :bagging_fraction => NumericRange @ 1…24)

Now we choose a tuning strategy, and a resampling strategy (for estimating performance), and wrap these strategies around our ensemble model to obtain a new model:

```julia
tuning = Grid(resolution=12)
resampling = Holdout(fraction_train=0.8)

tuned_ensemble_model = TunedModel(model=ensemble_model, 
    tuning_strategy=tuning, resampling_strategy=resampling, nested_ranges=nested_ranges)
```

    # TunedModel @ 6…74: 
    model                   =>   DeterministicEnsembleModel @ 1…59
    tuning_strategy         =>   Grid @ 1…83
    resampling_strategy     =>   Holdout @ 1…58
    measure                 =>   rms (generic function with 5 methods)
    operation               =>   predict (generic function with 19 methods)
    nested_ranges           =>   Params(:atom => Params(:K => NumericRange @ 1…22), :bagging_fraction => NumericRange @ 1…24)
    report_measurements     =>   true

Fitting the corresponding machine tunes the underlying model (in this case an ensemble) and retrains on all supplied data:

```julia
tuned_ensemble = machine(tuned_ensemble_model, X[train,:], y[train])
fit!(tuned_ensemble);
```

    ┌ Info: Training Machine{TunedModel{Grid,Determin…} @ 1…91.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:69
    Searching a 132-point grid for best model: 100%[==================================================] Time: 0:00:16
    ┌ Info: Training best model on all supplied data.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/tuning.jl:107

```julia
tuned_ensemble.report
```

    Dict{Symbol,Any} with 4 entries:
      :measurements     => [7.03102, 6.09291, 6.05707, 5.93617, 5.86848, 5.73299, 5…
      :models           => DeterministicEnsembleModel{Tuple{Array{Float64,2},Array{…
      :best_model       => DeterministicEnsembleModel @ 3…49
      :best_measurement => 5.46102

```julia
best_model = tuned_ensemble.report[:best_model]
@show best_model.bagging_fraction
@show best_model.atom.K
```

    best_model.bagging_fraction = 0.6363636363636364
    (best_model.atom).K = 43



