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

Packages wishing to implement the MLJ API for their algorithms should
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

Julia 0.7


### Installation

In the Julia REPL run  `]add https://github.com/alan-turing-institute/MLJ.jl.git`


### Basic train and test

```julia
using MLJ

X, y = datanow(); # boston dataset
train, test = partition(eachindex(y), 0.7); # 70:30 split
```

A *model* is a container for hyperparameters:

```julia
knn_model=KNNRegressor(K=10)

# KNNRegressor @ 1…94: 
K                       =>   10
metric                  =>   euclidean (generic function with 1 method)
kernel                  =>   reciprocal (generic function with 1 method)
```
Wrapping the model in data creates a *machine*, which is what stores the results of training, on which predictions on new data are based:


```julia
knn = machine(knn_model, X, y);
fit!(knn, rows=train);
yhat = predict(knn, X[test,:])
rms(y[test], yhat)

┌ Info: Training Machine{KNNRegressor} @ 5…64.
└ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:98

5.114498666132261
```
Changing a hyper-parameter and re-evaluating:

```julia
knn_model.K = 20
fit!(knn, rows=train)
yhat = predict(knn, X[test,:])
rms(y[test], yhat)

┌ Info: Training Machine{KNNRegressor} @ 5…64.
└ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:98

4.884523266056419
```

### Systematic tuning

```julia
K_range = range(knn_model, :K, lower=1, upper=100, scale=:log10)

# NumericRange @ 1…77: 
lower                   =>   1
upper                   =>   100
scale                   =>   :log10
```
A *tuned model* is just a regular model wrapped in a resampling strategy and a tuning strategy:

```julia
param_ranges = Params(:K => K_range)
tuning = Grid(resolution=8)
resampling = Holdout(fraction_train=0.8)
tuned_knn_model = TunedModel(model=knn_model, 
    tuning_stragegy=tuning, resampling_stragegy=resampling, param_ranges=param_ranges);
```
Fitting the corresponding machine tunes the underlying model and retrains on all supplied data:

```julia
tuned_knn = machine(tuned_knn_model, X[train,:], y[train])
fit!(tuned_knn); # fit using all provided data

┌ Info: Training Machine{TunedModel{Grid,KNNRegre…} @ 1…27.
└ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:98

Searching for best model...
model number: 1	 measurement: 2.0303940504246962    
model number: 2	 measurement: 1.9828439251201737    
model number: 3	 measurement: 2.6425280736693972    
model number: 4	 measurement: 2.973368220376769    
model number: 5	 measurement: 3.1908319369192526    
model number: 6	 measurement: 4.175863415495205    
model number: 7	 measurement: 4.731343943808259    
model number: 8	 measurement: 4.731343943808259    
    
Training best model on all supplied data...

┌ Info: Training Machine{KNNRegressor} @ 1…48.
└ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:98

tuned_knn.report[:best_model]

# KNNRegressor @ 1…28: 
K                       =>   2
metric                  =>   euclidean (generic function with 1 method)
kernel                  =>   reciprocal (generic function with 1 method)
    
yhat = predict(tuned_knn, X[test,:])
rms(yhat, y[test])

7.506195536032624
```


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
[AnalyticalEngine.jl](https://github.com/tlienart/AnalyticalEngine.jl),
[Orchestra.jl](https://github.com/svs14/Orchestra.jl), and
[Koala.jl](https://github.com/ablaom/Koala.jl). Work continued as a
research study group at the Univeristy of Warwick, beginning with a
review of existing ML Modules that are available in Julia
([in-depth](https://github.com/dominusmi/Julia-Machine-Learning-Review/tree/master/Educational),
[overview](https://github.com/dominusmi/Julia-Machine-Learning-Review/tree/master/Package%20Review)).

![alt text](material/packages.jpg)

Further work culminated in the first MLJ
[proof-of-concept](https://github.com/alan-turing-institute/MLJ.jl/tree/poc)



