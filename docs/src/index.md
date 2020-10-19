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
  <a href="https://github.com/alan-turing-institute/MLJ.jl/">For Developers</a> &nbsp;|&nbsp;
  <a href="https://mybinder.org/v2/gh/alan-turing-institute/MLJ.jl/master?filepath=binder%2FMLJ_demo.ipynb">Live Demo</a>
</div>
```

# Introduction

MLJ (Machine Learning in Julia) is a toolbox written in Julia
providing a common interface and meta-algorithms for selecting,
tuning, evaluating, composing and comparing [over 150 machine learning
models](@ref model_list) written in Julia and other languages. In
particular MLJ wraps a large number of
[scikit-learn](https://scikit-learn.org/stable/) models.

MLJ is released under the MIT licensed and sponsored by the [Alan
Turing Institute](https://www.turing.ac.uk/).


## Lightning tour

*For a more elementary introduction to MLJ usage see [Getting Started](@ref).*

The first code snippet below creates a new Julia environment
`MLJ_tour` and installs just those packages needed for the tour. See
[Installation](@ref) for more on creating a Julia environment for use
with MLJ.

Julia installation instructions are
[here](https://julialang.org/downloads/).

```julia
using Pkg
Pkg.activate("MLJ_tour", shared=true)
Pkg.add("MLJ")
Pkg.add("MLJModels")
Pkg.add("EvoTrees")
```

Load a selection of features and labels from the Ames House Price dataset:

```julia
using MLJ
X, y = @load_reduced_ames;
```

Load and instantiate a gradient tree-boosting model:

```julia
booster = @load EvoTreeRegressor
booster.max_depth = 2
booster.nrounds=50
```

Combine the model with categorical feature encoding:

```julia
pipe = @pipeline ContinuousEncoder booster
```

Define a hyper-parameter range for optimization:

```julia
max_depth_range = range(pipe,
                        :(evo_tree_regressor.max_depth),
                        lower = 1,
                        upper = 10)
```

Wrap the pipeline model in an optimization strategy:

```julia
self_tuning_pipe = TunedModel(model=pipe,
                              tuning=RandomSearch(),
                              ranges = max_depth_range,
                              resampling=CV(nfolds=3, rng=456),
                              measure=l1,
                              acceleration=CPUThreads(),
                              n=50)
```

Bind the "self-tuning" pipeline model (just a container for
hyper-parameters) to data in a *machine* (which will additionally
store *learned* parameters):

```julia
mach = machine(self_tuning_pipe, X, y)
```

Evaluate the "self-tuning" pipeline model's performance (implies nested resampling):

```julia
julia> evaluate!(mach,
                measures=[l1, l2],
                resampling=CV(nfolds=6, rng=123),
                acceleration=CPUProcesses(), verbosity=2)
┌───────────┬───────────────┬────────────────────────────────────────────────────────┐
│ _.measure │ _.measurement │ _.per_fold                                             │
├───────────┼───────────────┼────────────────────────────────────────────────────────┤
│ l1        │ 16700.0       │ [16100.0, 16400.0, 14500.0, 17000.0, 16400.0, 19500.0] │
│ l2        │ 6.43e8        │ [5.88e8, 6.81e8, 4.35e8, 6.35e8, 5.98e8, 9.18e8]       │
└───────────┴───────────────┴────────────────────────────────────────────────────────┘
_.per_observation = [[[29100.0, 9990.0, ..., 103.0], [12100.0, 1330.0, ..., 13200.0], [6490.0, 22000.0, ..., 13800.0], [9090.0, 9530.0, ..., 13900.0], [50800.0, 22700.0, ..., 1550.0], [32800.0, 4940.0, ..., 1110.0]], [[8.45e8, 9.98e7, ..., 10500.0], [1.46e8, 1.77e6, ..., 1.73e8], [4.22e7, 4.86e8, ..., 1.9e8], [8.26e7, 9.09e7, ..., 1.93e8], [2.58e9, 5.13e8, ..., 2.42e6], [1.07e9, 2.44e7, ..., 1.24e6]]]
_.fitted_params_per_fold = [ … ]
_.report_per_fold = [ … ]

```

Try out MLJ yourself in the following batteries-included Binder
[notebook](https://mybinder.org/v2/gh/alan-turing-institute/MLJ.jl/master?filepath=binder%2FMLJ_demo.ipynb). No
installation required.


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
julia> using Pkg; Pkg.activate("my_MLJ_env", shared=true)
```

Installing MLJ is also done with the package manager:

```julia
julia> Pkg.add("MLJ")
```

**Optional:** To test your installation, run

```julia
julia> Pkg.test("MLJ")
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

or refer to [List of Supported Models](@ref model_list)

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
intended as a complete reference. For a short introduction to basic
MLJ functionality, read the [Getting Started](@ref) section of this
manual. For extensive tutorials, we recommend the [MLJ
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
