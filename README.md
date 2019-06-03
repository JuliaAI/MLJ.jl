## MLJ

A pure Julia machine learning framework.

[MLJ News](https://alan-turing-institute.github.io/MLJ.jl/dev/NEWS/)


## `join!(MLJ, YourModel)`

**Call for help.** MLJ is [getting
attention](https://github.com/trending/julia?since=monthly) but its
small project team needs help to ensure its success. This depends
crucially on:

- Existing and developing ML algorithms implementing the MLJ model interface

- Improvements to existing but poorly maintained Julia ML algorithms 

The MLJ model interface is now relatively stable and
[well-documented](https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/),
and the core team is happy to respond to [issue requests](https://github.com/alan-turing-institute/MLJ.jl/issues) for
assistance. Please click [here](CONTRIBUTE.md) for more details on
contributing.

MLJ is presently supported by a small Alan Turing Institute grant and is looking for new funding sources to grow the project.

[![Build Status](https://travis-ci.com/alan-turing-institute/MLJ.jl.svg?branch=master)](https://travis-ci.com/alan-turing-institute/MLJ.jl)
[![Slack Channel mlj](https://img.shields.io/badge/chat-on%20slack-yellow.svg)](https://slackinvite.julialang.org/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://alan-turing-institute.github.io/MLJ.jl/dev/)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://alan-turing-institute.github.io/MLJ.jl/stable/)
[![Coverage Status](https://coveralls.io/repos/github/alan-turing-institute/MLJ.jl/badge.svg?branch=master)](https://coveralls.io/github/alan-turing-institute/MLJ.jl?branch=master)

![](docs/src/two_model_stack.png)

MLJ aims to be a flexible framework for combining and tuning machine
learning models, written in the high performance, rapid development,
scientific programming language, [Julia](https://julialang.org). 


The MLJ project is partly inspired by [MLR](https://mlr.mlr-org.com/index.html).

A list of models implementing the MLJ interface:
[MLJRegistry](https://github.com/alan-turing-institute/MLJRegistry.jl/blob/dev/Models.toml)





### Installation

In the Julia REPL:

````julia
]add MLJ
add MLJModels
````

A docker image with installation [instructions](https://github.com/ysimillides/mlj-docker) is also available.


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

- **Task** interface matches machine learning problem to available models. &#10004; 

- **Benchmarking** a battery of assorted models for a given task.

- Automated estimates of cpu and memory requirements for given task/model.


### Frequently Asked Questions

See [here](docs/src/frequently_asked_questions.md).


### Known issues

- The ScikitLearn SVM models will not work under Julia 1.0.3 but do work under Julia 1.1 due to [Issue #29208](https://github.com/JuliaLang/julia/issues/29208)

- When MLJRegistry is updated with new models you may need to force a new
  precompilation of MLJ to make new models available.
  

### Getting started

Get started
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/),
or take the MLJ [tour](/examples/tour/tour.ipynb).


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



