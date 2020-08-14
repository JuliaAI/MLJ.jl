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
  <a href="https://github.com/alan-turing-institute/MLJ.jl/">Repository</a>       &nbsp;|&nbsp;
  <a href="https://mybinder.org/v2/gh/alan-turing-institute/MLJ.jl/master?filepath=binder%2FMLJ_demo.ipynb">Binder</a>
</div>
```

# Introduction

MLJ (Machine Learning in Julia) is a toolbox written in Julia
providing a common interface and meta-algorithms for selecting,
tuning, evaluating, composing and comparing machine learning models
written in Julia and other languages. In particular MLJ wraps a large
number of [scikit-learn](https://scikit-learn.org/stable/) models. 

MLJ is released under the MIT licensed and sponsored by the [Alan
Turing Institute](https://www.turing.ac.uk/).


Try out MLJ in the following
[notebook](https://mybinder.org/v2/gh/alan-turing-institute/MLJ.jl/master?filepath=binder%2FMLJ_demo.ipynb)
on Binder. No installation required. 


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

* Extensive, state-of-the art, support for model composition
  (*pipelines* and *learning networks*) (see more
  [below](#model-composability)),

* Convenient syntax to tune and evaluate (composite) models.

* Consistent interface to handle probabilistic predictions.

* Extensible [tuning
  interface](https://github.com/alan-turing-institute/MLJTuning.jl),
  to support growing number of optimization strategies, and designed
  to play well with model composition.


## Model composability

The generic model composition API's provided by other toolboxes we
have surveyed share one or more of the following shortcomings, which
do not exist in MLJ:

- Composite models do not inherit all the behavior of ordinary
  models.

- Composition is limited to linear (non-branching) pipelines.

- Supervised components in a linear pipeline can only occur at the
  end of the pipeline.

- Only static (unlearned) target transformations/inverse
  transformations are supported.

- Hyper-parameters in homogeneous model ensembles cannot be coupled.

- Model stacking, with out-of-sample predictions for base learners,
  cannot be implemented (using the generic API alone).

- Hyper-parameters and/or learned parameters of component models are
  not easily inspected or manipulated (by tuning algorithms, for
  example)
  
- Composite models cannot implement multiple opertations, for example,
  both a `predict` and `transform` method (as in clustering models) or
  both a `transform` and `inverse_transform` method.
  
Some of these features are demonstrated in [this
notebook](https://github.com/ablaom/MachineLearningInJulia2020/blob/master/wow.ipynb)

For more information see the [MLJ design
paper](https://github.com/alan-turing-institute/MLJ.jl/blob/master/paper/paper.md)


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


## Learning Julia

If you have experience in programming in another language but are new
to Julia, then we highly recommend Aaron Christinson's tutorial
[Dispatching Design
Patterns](https://github.com/ninjaaron/dispatching-design-patterns)
which is nicely compressed in his [half-hour
video presentation](https://live.juliacon.org/talk/JYNERU).

However, one doesn't need to be able to program in Julia to start
using MLJ.

## Learning to use MLJ

The present document, although littered with examples, is primarily
intended as a complete reference. For a lightning introduction to MLJ
read the [Getting Started](@ref) section of this manual. For more
leisurely and extensive tutorials, we highly recommend the [MLJ
Tutorials](https://alan-turing-institute.github.io/DataScienceTutorials.jl/)
website.  Each tutorial can be downloaded as a notebook or Julia
script to facilitate experimentation. Finally, you may like to
checkout the [JuliaCon2020
Workshop](https://github.com/ablaom/MachineLearningInJulia2020) on MLJ
(recorded
[here](https://www.youtube.com/watch?time_continue=27&v=qSWbCn170HU&feature=emb_title)).

You can try also MLJ out in the following
[notebook](https://mybinder.org/v2/gh/alan-turing-institute/MLJ.jl/master?filepath=binder%2FMLJ_demo.ipynb)
on Binder, without installing Julia or MLJ.

Users are also welcome to join the `#mlj` Julia slack channel to ask
questions and make suggestions.


## Citing MLJ

When presenting work that uses MLJ, please cite the [MLJ design
paper](https://arxiv.org/abs/2007.12285). Here is the relevant bibtex entry:

```bitex
@misc{blaom2020mlj,
    title={MLJ: A Julia package for composable machine learning},
    author={Anthony D. Blaom and Franz Kiraly and Thibaut Lienart and Yiannis Simillides and Diego Arenas and Sebastian J. Vollmer},
    year={2020},
    eprint={2007.12285},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
