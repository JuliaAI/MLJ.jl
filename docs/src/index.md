```@raw html
<script async defer src="https://buttons.github.io/buttons.js"></script>

<span style="color:darkslateblue;font-size:2.25em;font-style:italic;">
A Machine Learning Framework for Julia
</span>  &nbsp; &nbsp; &nbsp; &nbsp;
<a class="github-button" href="https://github.com/alan-turing-institute/MLJ.jl" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star alan-turing-institute/MLJ.jl on GitHub">Star</a>

<br>
<br>
<div style="font-size:1.25em;font-weight:bold;">
  <a href="#Installation-1" style="color: orange;">Install</a>         &nbsp;|&nbsp;
  <a href="#Learning-to-use-MLJ-1" style="color: orange;">Learn</a>    &nbsp;|&nbsp;
  <a href="mlj_cheatsheet">Cheatsheet</a>       &nbsp;|&nbsp;
  <a href="common_mlj_workflows">Workflows</a>  &nbsp;|&nbsp;
  <a href="https://github.com/alan-turing-institute/MLJ.jl/">For Developers</a> &nbsp;|&nbsp;
  <a href="https://mybinder.org/v2/gh/alan-turing-institute/MLJ.jl/master?filepath=binder%2FMLJ_demo.ipynb">Live Demo</a> &nbsp;|&nbsp;
  <a href="third_party_packages">3rd Party Packages</a>
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

*For more elementary introductions to MLJ usage see [Basic
introductions](@ref) below.*

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
Pkg.add("MLJIteration")
Pkg.add("EvoTrees")
```

In MLJ a *model* is just a container for hyper-parameters, and that's
all. Here we will apply several kinds of model composition before
binding the resulting "meta-model" to data in a *machine* for
evaluation using cross-validation.

Loading and instantiating a gradient tree-boosting model:

```julia
using MLJ
Booster = @load EvoTreeRegressor # loads code defining a model type
booster = Booster(max_depth=2)   # specify hyper-parameter at construction
booster.nrounds=50               # or mutate post facto
```

This model is an example of an iterative model. As is stands, the
number of iterations `nrounds` is fixed.


#### Composition 1: Wrapping the model to make it "self-iterating"

Let's create a new model that automatically learns the number of iterations,
using the `NumberSinceBest(3)` criterion, as applied to an
out-of-sample `l1` loss:

```julia
using MLJIteration
iterated_booster = IteratedModel(model=booster,
                                 resampling=Holdout(fraction_train=0.8),
                                 controls=[Step(2), NumberSinceBest(3), NumberLimit(300)],
                                 measure=l1,
                                 retrain=true)
```

#### Composition 2: Preprocess the input features

Combining the model with categorical feature encoding:

```julia
pipe = @pipeline ContinuousEncoder iterated_booster
```

#### Composition 3: Wrapping the model to make it "self-tuning"

First, we define a hyper-parameter range for optimization of a
(nested) hyper-parameter:

```julia
max_depth_range = range(pipe,
                        :(deterministic_iterated_model.model.max_depth),
                        lower = 1,
                        upper = 10)
```

Now we can wrap the pipeline model in an optimization strategy to make
it "self-tuning":

```julia
self_tuning_pipe = TunedModel(model=pipe,
                              tuning=RandomSearch(),
                              ranges = max_depth_range,
                              resampling=CV(nfolds=3, rng=456),
                              measure=l1,
                              acceleration=CPUThreads(),
                              n=50)
```

#### Binding to data and evaluating performance

Loading a selection of features and labels from the Ames
House Price dataset:

```julia
X, y = @load_reduced_ames;
```

Binding the "self-tuning" pipeline model to data in a *machine* (which
will additionally store *learned* parameters):

```julia
mach = machine(self_tuning_pipe, X, y)
```

Evaluating the "self-tuning" pipeline model's performance using 5-fold
cross-validation (implies multiple layers of nested resampling):

```julia
julia> evaluate!(mach,
                 measures=[l1, l2],
                 resampling=CV(nfolds=5, rng=123),
                 acceleration=CPUThreads(),
                 verbosity=2)
┌────────────────────┬───────────────┬───────────────────────────────────────────────┐
│ _.measure          │ _.measurement │ _.per_fold                                    │
├────────────────────┼───────────────┼───────────────────────────────────────────────┤
│ LPLoss{Int64} @410 │ 16900.0       │ [17000.0, 16200.0, 16200.0, 16400.0, 18600.0] │
│ LPLoss{Int64} @632 │ 6.57e8        │ [6.38e8, 6.19e8, 5.92e8, 5.67e8, 8.7e8]       │
└────────────────────┴───────────────┴───────────────────────────────────────────────┘
_.per_observation = [[[20300.0, 21800.0, ..., 7910.0], [4300.0, 31900.0, ..., 12600.0], [22000.0, 91600.0, ..., 35500.0], [2980.0, 35700.0, ..., 6240.0], [9140.0, 30000.0, ..., 3050.0]], [[4.13e8, 4.74e8, ..., 6.26e7], [1.85e7, 1.02e9, ..., 1.59e8], [4.83e8, 8.38e9, ..., 1.26e9], [8.86e6, 1.28e9, ..., 3.89e7], [8.35e7, 9.01e8, ..., 9.31e6]]]
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
  parallelism, automatic differentiation, GPU, optimization etc.


## Key features

* Data agnostic, train models on any data supported by the
  [Tables.jl](https://github.com/JuliaData/Tables.jl) interface,

* Extensive, state-of-the art, support for model composition
  (*pipelines* and *learning networks*) (see more
  [below](#model-composability)),

* Convenient syntax to tune and evaluate (composite) models.

* Consistent interface to handle probabilistic predictions.

* Extensible [tuning
  interface](https://github.com/JuliaAI/MLJTuning.jl),
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

- Composite models cannot implement multiple operations, for example,
  both a `predict` and `transform` method (as in clustering models) or
  both a `transform` and `inverse_transform` method.

Some of these features are demonstrated in [this
notebook](https://github.com/ablaom/MachineLearningInJulia2020/blob/master/wow.ipynb)

For more information see the [MLJ design
paper](https://doi.org/10.21105/joss.02704) or our detailed
[paper](https://arxiv.org/abs/2012.15505) on the composition
interface.


## Getting help and reporting problems

Users are encouraged to provide feedback on their experience using MLJ
and to report issues.

For a query to have maximum exposure to maintainers and users, start a
discussion thread at [Julia Discourse Machine
Learning](https://github.com/alan-turing-institute/MLJ.jl) and tag
your issue "mlj". Queries can also be posted as
[issues](https://github.com/alan-turing-institute/MLJ.jl/issues), or
on the `#mlj` slack workspace in the Julia Slack channel.

Bugs, suggestions, and feature requests can be posted
[here](https://github.com/alan-turing-institute/MLJ.jl/issues).

See also, [Known Issues](@ref)


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
providing unified access to _model providing packages_. For this
reason, one generally needs to add further packages to your
environment to make model-specific code available. This
happens automatically when you use MLJ's interactive load command
`@iload`, as in

```julia
julia> Tree = @iload DecisionTreeClassifier # load type
julia> tree = Tree() # instance
```

where you will also be asked to choose a providing package, for more
than one provide a `DecisionTreeClassifier` model. For more on
identifying the name of an applicable model, see [Model Search](@ref model_search). 
For non-interactive loading of code (e.g., from a
module or function) see [Loading Model Code](@ref).

It is recommended that you start with models from more mature
packages such as DecisionTree.jl, ScikitLearn.jl or XGBoost.jl.

MLJ is supported by a number of satellite packages (MLJTuning,
MLJModelInterface, etc) which the general user is *not* required to
install directly. Developers can learn more about these
[here](https://github.com/alan-turing-institute/MLJ.jl/blob/master/ORGANIZATION.md).

See also the alternative instalation instructions for [Customizing
Behavior](@ref).


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
intended as a complete reference. Resources for learning MLJ are:

### Basic introductions

- the [Getting Started](@ref) section of this manual

- an introductory [binder notebook](https://mybinder.org/v2/gh/alan-turing-institute/MLJ.jl/master?filepath=binder%2FMLJ_demo.ipynb) (no Julia/MLJ installation required)

To get direct help from maintainers and other users, see [Getting help
and reporting problems](@ref).


### In depth

- the MLJ JuliaCon2020 Workshop [materials](https://github.com/ablaom/MachineLearningInJulia2020) and [video recording](https://www.youtube.com/watch?time_continue=27&v=qSWbCn170HU&feature=emb_title)

- [Data Science Tutorials in Julia](https://alan-turing-institute.github.io/DataScienceTutorials.jl/)

Users are also welcome to join the `#mlj` Julia slack channel to ask
questions and make suggestions.


## Citing MLJ

When presenting work that uses MLJ, please cite the MLJ design
paper:

[![DOI](https://joss.theoj.org/papers/10.21105/joss.02704/status.svg)](https://doi.org/10.21105/joss.02704)

```bibtex
@article{Blaom2020,
  doi = {10.21105/joss.02704},
  url = {https://doi.org/10.21105/joss.02704},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {55},
  pages = {2704},
  author = {Anthony D. Blaom and Franz Kiraly and Thibaut Lienart and Yiannis Simillides and Diego Arenas and Sebastian J. Vollmer},
  title = {{MLJ}: A Julia package for composable machine learning},
  journal = {Journal of Open Source Software}
}
```

If using the model composition features of MLJ (learning networks)
please additionally cite

```bitex
@misc{blaom2020flexible,
  title={{Flexible model composition in machine learning and its implementation in MLJ}},
  author={Anthony D. Blaom and Sebastian J. Vollmer},
  year={2020},
  eprint={2012.15505},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
