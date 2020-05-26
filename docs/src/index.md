```@raw html
<span style="color:darkslateblue;font-size:2.25em;font-style:italic;">
A Machine Learning Framework for Julia
</span>
<br>
<br>
<div style="font-size:1.25em;font-weight:bold;">
  <a href="#Installation-1">Installation</a>    &nbsp;|&nbsp;
  <a href="mlj_cheatsheet">Cheatsheet</a>       &nbsp;|&nbsp;
  <a href="common_mlj_workflows">Workflows</a>  &nbsp;|&nbsp;
  <a href="https://alan-turing-institute.github.io/DataScienceTutorials.jl/">Tutorials</a>       &nbsp;|&nbsp;
  <a href="https://github.com/alan-turing-institute/MLJ.jl/">Repository</a>
</div>
```

# Introduction

MLJ (Machine Learning in Julia) is a toolbox written in Julia
providing a common interface and meta-algorithms for selecting,
tuning, evaluating, composing and comparing machine learning models
written in Julia and other languages.  MLJ is released
under the MIT licensed and sponsored by the [Alan Turing
Institute](https://www.turing.ac.uk/).


## Key goals

* Offer a consistent way to use, compose and tune machine learning
  models in Julia,

* Promote the improvement of the Julia ML/Stats ecosystem by making it
  easier to use models from a wide range of packages,

* Unlock performance gains by exploiting Julia's support for
  parallelism, automatic differentiation, GPU, optimisation etc.


## Key features

* Data agnostic, train models on any data supported by the
  [Tables.jl](https://github.com/JuliaData/Tables.jl) interface,

* Extensive support for model composition (*pipelines* and *learning
  networks*),

* Convenient syntax to tune and evaluate (composite) models.

* Consistent interface to handle probabilistic predictions.

* Extensible [tuning
  interface](https://github.com/alan-turing-institute/MLJTuning.jl),
  to support growing number of optimization strategies, and designed
  to play well with model composition.


More information is available from the [MLJ design paper](https://github.com/alan-turing-institute/MLJ.jl/blob/master/paper/paper.md)


## Reporting problems

Users are encouraged to provide feedback on their experience using MLJ
and to report issues. You can do so
[here](https://github.com/alan-turing-institute/MLJ.jl/issues) or on
the `#mlj` Julia slack channel.

For known issues that are not strictly MLJ bugs, see
[here](https://github.com/alan-turing-institute/MLJ.jl#known-issues)


## Installation

Initially it is recommended that MLJ and associated packages be
installed in a new
[environment](https://julialang.github.io/Pkg.jl/v1/environments/) to
avoid package conflicts. You can do this with

```julia
julia> using Pkg; Pkg.activate("My_MLJ_env", shared=true)
```

Installing MLJ is also done with the package manager:

```julia
julia> Pkg.add("MLJ")
```

It is important to note that MLJ is essentially a big wrapper
providing a unified access to _model providing packages_ and so you
will also need to make sure these packages are available in your
environment.  For instance, if you want to use a **Decision Tree
Classifier**, you need to have
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl)
installed:

```julia
julia> Pkg.add("DecisionTree");
julia> using MLJ;
julia> @load DecisionTreeClassifier
```

For a list of models and their packages run

```julia
using MLJ
models()
```
or refer to [this table](https://github.com/alan-turing-institute/MLJ.jl#list-of-wrapped-models).

It is recommended that you start with models marked as coming from mature
packages such as DecisionTree.jl, ScikitLearn.jl or XGBoost.jl.

MLJ is supported by a number of satelite packages (MLJTuning,
MLJModelInterface, etc) which the general user is *not* required to
install directly. Developers can learn more about these
[here](https://github.com/alan-turing-institute/MLJ.jl/blob/master/ORGANIZATION.md)


## Learning to use MLJ

The present document, although littered with examples, is primarily
intended as a complete reference. For a lightning introduction to MLJ
read the [Getting Started](@ref) section of this manual. For more
leisurely and extensive tutorials, we highly recommend the [MLJ
Tutorials](https://alan-turing-institute.github.io/MLJTutorials/)
website.  Each tutorial can be downloaded as a notebook or Julia
script to facilitate experimentation.

Users are also welcome to join the `#mlj` Julia slack channel to ask
questions and make suggestions.


## Citing MLJ

An MLJ [design
paper](https://github.com/alan-turing-institute/MLJ.jl/blob/master/paper/paper.md)
is under review. In the meantime, please cite the software using one
of the following:

[https://doi.org/10.5281/zenodo.3541506](https://doi.org/10.5281/zenodo.3541506)

```bibtex
@software{anthony_blaom_2019_3541506,
  author       = {Anthony Blaom and
                  Franz Kiraly and
                  Thibaut Lienart and
                  Sebastian Vollmer},
  title        = {alan-turing-institute/MLJ.jl: v0.5.3},
  month        = nov,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {v0.5.3},
  doi          = {10.5281/zenodo.3541506},
  url          = {https://doi.org/10.5281/zenodo.3541506}
}
```
