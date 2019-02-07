## MLJ

A pure Julia machine learning framework.

[![Build Status](https://travis-ci.com/alan-turing-institute/MLJ.jl.svg?branch=master)](https://travis-ci.com/alan-turing-institute/MLJ.jl)
[![Slack Channel mlj](https://img.shields.io/badge/chat-on%20slack-yellow.svg)](https://slackinvite.julialang.org/)

![](doc/two_model_stack.png)

MLJ aims to be a flexible framework for combining and tuning machine
learning models, written in the high performance, rapid development,
scientific programming language, [Julia](https://julialang.org). MLJ
is work in progress and new collaborators are being sought. 

Click [here](CONTRIBUTE.md) if your are interested in contributing.

The MLJ project is partly inspired by [MLR](https://mlr.mlr-org.com/index.html) ([recent
slides 7/18](https://github.com/mlr-org/mlr-outreach).) For an earlier proof-of-concept, see
[this branch](https://github.com/alan-turing-institute/MLJ.jl/tree/poc)
and [this poster summary](material/MLJ-JuliaCon2018-poster.pdf).

Packages wishing to implement the MLJ interface for their algorithms should
import [MLJBase](https://github.com/alan-turing-institute/MLJBase.jl).

### Features to include:

- **Automated tuning** of hyperparameters, including
  composite models with *nested parameters*. Tuning implemented as a
  wrapper, allowing composition with other meta-algorithms. &#10004;

- Option to tune hyperparameters using gradient descent and **automatic
	differentiation** (for learning algorithms written in Julia).

- **Data agnostic**: Train models on any data supported by the Tables.jl 
[interface](https://github.com/JuliaData/Tables.jl). &#10004;

- Intuitive syntax for building arbitrarily complicated
  **learning networks** .&#10004;
  
- Learning networks can be exported as self-contained **composite models** &#10004;, but
  common networks (e.g., linear pipelines, stacks) come ready to plug-and-play.

- Performant parallel implementation of large homogeneous **ensembles**
  of arbitrary models (e.g., random forests). &#10004;

- **Task** interface matches machine learning problem to available models.

- **Benchmarking** a battery of assorted models for a given task.

- Automated estimates of cpu and memory requirements for given task/model.


### Requirements

* Julia 1.0 or higher


### Installation

In the Julia REPL:

````julia
]add https://github.com/alan-turing-institute/MLJBase.jl
add https://github.com/alan-turing-institute/MLJ.jl
````

### Basic training and testing

For more detail and features, see the evolving MLJ [tour](doc/tour.ipynb).

```julia
using MLJ
using DataFrames

task = load_boston()
X, y = X_and_y(task);

X = DataFrame(X) # or any other tabular format supported by Tables.jl 

train, test = partition(eachindex(y), 0.7); # 70:30 split
```

A *model* is a container for hyperparameters:

```julia
knn_model=KNNRegressor(K=10)
```

    # KNNRegressor{Float64} @ 1…90: 
    target_type             =>   Float64
    K                       =>   10
    metric                  =>   euclidean (generic function with 1 method)
    kernel                  =>   reciprocal (generic function with 1 method)

Wrapping the model in data creates a *machine* which will store training outcomes (called *fit-results*):

```julia
knn = machine(knn_model, X, y)
```

    # Machine{KNNRegressor{Float64}} @ 9…72: 
    model                   =>   KNNRegressor{Float64} @ 1…90
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

    ┌ Info: Training Machine{KNNRegressor{Float64}} @ 9…72.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:69
    8.090639098853249

Or, in one line:

```julia
evaluate!(knn, resampling=Holdout(fraction_train=0.7))
```

    8.090639098853249

Changing a hyperparameter and re-evaluating:

```julia
knn_model.K = 20
evaluate!(knn, resampling=Holdout(fraction_train=0.7))
```

    8.41003854724935

### Systematic tuning as a model wrapper

A simple example of a composite model is a homogeneous ensemble. Here's a bagged ensemble model for 20 K-nearest neighbour regressors:

```julia
ensemble_model = EnsembleModel(atom=knn_model, n=20) 
```

    # DeterministicEnsembleModel @ 5…24: 
    atom                    =>   KNNRegressor{Float64} @ 1…90
    weights                 =>   0-element Array{Float64,1}
    bagging_fraction        =>   0.8
    rng_seed                =>   0
    n                       =>   20
    parallel                =>   true
    
Let's simultaneously tune the ensemble's `bagging_fraction` and the K-nearest neighbour hyperparameter `K`. Since one of these models is a field of the other, we have nested hyperparameters:

```julia
params(ensemble_model)
```

    Params(:atom => Params(:target_type => Float64, :K => 20, :metric => MLJ.KNN.euclidean, :kernel => MLJ.KNN.reciprocal), :weights => Float64[], :bagging_fraction => 0.8, :rng_seed => 0, :n => 20, :parallel => true)

To define a tuning grid, we construct ranges for the two parameters and collate these ranges following the same pattern above (omitting parameters that don't change):

```julia
B_range = range(ensemble_model, :bagging_fraction, lower= 0.5, upper=1.0, scale = :linear)
K_range = range(knn_model, :K, lower=1, upper=100, scale=:log10)
nested_ranges = Params(:atom => Params(:K => K_range), :bagging_fraction => B_range)
```

    Params(:atom => Params(:K => NumericRange @ 1…75), :bagging_fraction => NumericRange @ 1…56)

Now we choose a tuning strategy, and a resampling strategy (for estimating performance), and wrap these strategies around our ensemble model to obtain a new model:

```julia
tuning = Grid(resolution=12)
resampling = CV(nfolds=6)

tuned_ensemble_model = TunedModel(model=ensemble_model, 
    tuning=tuning, resampling=resampling, nested_ranges=nested_ranges)
```

    # DeterministicTunedModel @ 1…93: 
    model                   =>   DeterministicEnsembleModel @ 5…24
    tuning                  =>   Grid @ 1…37
    resampling              =>   CV @ 6…31
    measure                 =>   nothing
    operation               =>   predict (generic function with 19 methods)
    nested_ranges           =>   Params(:atom => Params(:K => NumericRange @ 1…75), :bagging_fraction => NumericRange @ 1…56)
    report_measurements     =>   true
    
Fitting the corresponding machine tunes the underlying model (in this case an ensemble) and retrains on all supplied data:

```julia
tuned_ensemble = machine(tuned_ensemble_model, X[train,:], y[train])
fit!(tuned_ensemble);
```

    ┌ Info: Training Machine{MLJ.DeterministicTunedMo…} @ 1…05.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:69
    Searching a 132-point grid for best model: 100%[=========================] Time: 0:01:20
    ┌ Info: Training best model on all supplied data.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/tuning.jl:130

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

    best_model.bagging_fraction = 0.7272727272727273
    (best_model.atom).K = 100


### History

Predecessors of the current package are
[AnalyticalEngine.jl](https://github.com/tlienart/AnalyticalEngine.jl)
and [Orchestra.jl](https://github.com/svs14/Orchestra.jl), and
[Koala.jl](https://github.com/ablaom/Koala.jl). Work
continued as a research study group at the University of Warwick,
beginning with a review of existing ML Modules that were available in
Julia at the time ([in-depth](https://github.com/dominusmi/Julia-Machine-Learning-Review/tree/master/Educational),
[overview](https://github.com/dominusmi/Julia-Machine-Learning-Review/tree/master/Package%20Review)).

![alt text](material/packages.jpg)

Further work culminated in the first MLJ
[proof-of-concept](https://github.com/alan-turing-institute/MLJ.jl/tree/poc)



