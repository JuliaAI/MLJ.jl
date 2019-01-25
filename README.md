## MLJ

A pure Julia machine learning framework.

[![Build Status](https://travis-ci.com/alan-turing-institute/MLJ.jl.svg?branch=master)](https://travis-ci.com/alan-turing-institute/MLJ.jl)
[![Slack Channel mlj](https://img.shields.io/badge/chat-on%20slack-yellow.svg)](https://slackinvite.julialang.org/)

MLJ aims to be a flexible framework for combining and tuning machine
learning models, written in the high performance, rapid
development, scientific programming language, Julia. This is a work in
progress and new [collaborators](#collaborators) are being sought.

![](doc/two_model_stack.png)


In large measure the MLJ project is inspired by [mlR](https://pat-s.github.io/mlr/index.html) ([recent
slides 7/18](https://github.com/mlr-org/mlr-outreach).) For an earlier proof-of-concept, see
[this branch](https://github.com/alan-turing-institute/MLJ.jl/tree/poc)
and [this poster summary](material/MLJ-JuliaCon2018-poster.pdf).

Packages wishing to implement the MLJ interface for their algorithms should
import MLJBase, which has its own
[repository](https://github.com/alan-turing-institute/MLJBase.jl).

### Features to include:

- Automated tuning of hyperparameters, including
  composite models with nested parameters. Tuning implemented as a
  wrapper, allowing composition with other meta-algorithms. &#10004;

- Option to tune hyperparameters using gradient descent and automatic
  differentiation (for learning algorithms written in Julia).

- Data agnostic: Train models on any data supported by the Queryverse
[iterable tables
interface](https://github.com/queryverse/IterableTables.jl). &#10004;

- Intuitive syntax for building arbitrarily complicated
  learning networks. &#10004;
  
- Learning networks can be exported as self-contained composite models &#10004;, but
  common networks (e.g., linear pipelines, stacks) come ready to plug-and-play.

- Performant parallel implementation of large homogeneous ensembles
  of arbitrary models (e.g., random forests).

- "Task" interface matches machine learning problem to available models.

- Benchmarking a battery of assorted models for a given task.

- Automated estimates of cpu and memory requirements for given task/model.


### Requirements

* Julia 0.7 or higher


### Installation

In the Julia REPL run `]add https://github.com/alan-turing-institute/MLJBase.jl` first and `]add https://github.com/alan-turing-institute/MLJ.jl.git` afterwards.


### Basic training and testing

See also the MLJ [tour](doc/tour.ipynb).

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


### Collaborators

Diego Arenas, Edoardo Barp, Anthony Blaom, Gergö Bohner, Valentin
Churvay, Harvey Devereux, Thibaut Lienart, Franz J Király, Mohammed
Nook, Annika Stechemesser, Yiannis Simillides, Sebastian Vollmer; Mike
Innes in partnership with Julia Computing

We are looking for new collaborators @ the Alan Turing Institute! 
  * Implementation of unsupported learners
  * Backend improvement! (Scheduling, Dagger, JuliaDB, Queryverse)
  * Store learner meta info in METADATA.JL fashion (ideally open.ml compatible)
  * Feature Improvement 
  * Bootstrapping from Sklearn and mlr by wrapping with task info
  

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



