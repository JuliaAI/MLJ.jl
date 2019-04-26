## MLJ

A pure Julia machine learning framework.

[![Build Status](https://travis-ci.com/alan-turing-institute/MLJ.jl.svg?branch=master)](https://travis-ci.com/alan-turing-institute/MLJ.jl)
[![Slack Channel mlj](https://img.shields.io/badge/chat-on%20slack-yellow.svg)](https://slackinvite.julialang.org/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://alan-turing-institute.github.io/MLJ.jl/dev/)

![](docs/src/two_model_stack.png)

MLJ aims to be a flexible framework for combining and tuning machine
learning models, written in the high performance, rapid development,
scientific programming language, [Julia](https://julialang.org). 

MLJ is in a relatively early stage of development and welcomes new
collaborators. Click [here](CONTRIBUTE.md) if you are interested in
contributing, or if you are interested in implementing the MLJ
interface for an existing Julia machine learning algorithm.

The MLJ project is partly inspired by [MLR](https://mlr.mlr-org.com/index.html).

A list of models implementing the MLJ interface:
[MLJRegistry](https://github.com/alan-turing-institute/MLJRegistry.jl/blob/master/Models.toml)


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

- **Task** interface matches machine learning problem to available models. &#10004; (mostly)

- **Benchmarking** a battery of assorted models for a given task.

- Automated estimates of cpu and memory requirements for given task/model.


### Frequently Asked Questions

See [here](docs/src/frequently_asked_questions.md).


### Known issues

- The ScikitLearn SVM models will not work under Julia 1.0.3 but do work under Julia 1.1 due to [Issue #29208](https://github.com/JuliaLang/julia/issues/29208)


### Getting started

Get started
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/),
or take the MLJ [tour](docs/src/tour.ipynb).


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



