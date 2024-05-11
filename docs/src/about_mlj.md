# About MLJ

MLJ (Machine Learning in Julia) is a toolbox written in Julia 
providing a common interface and meta-algorithms for selecting,
tuning, evaluating, composing and comparing [over 180 machine learning
models](@ref model_list) written in Julia and other languages. In
particular MLJ wraps a large number of
[scikit-learn](https://scikit-learn.org/stable/) models.

MLJ is released under the MIT license.

## Lightning tour

*For help learning to use MLJ, see [Learning MLJ](@ref)*.

A self-contained notebook and julia script of this demonstration is
also available
[here](https://github.com/JuliaAI/MLJ.jl/tree/dev/examples/lightning_tour).

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
booster.nrounds=50               # or mutate afterwards
```

This model is an example of an iterative model. As it stands, the
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
pipe = ContinuousEncoder() |> iterated_booster
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
Evaluating the "self-tuning" pipeline model's performance using 5-fold
cross-validation (implies multiple layers of nested resampling):

```julia-repl
julia> evaluate(self_tuning_pipe, X, y,
                measures=[l1, l2],
                resampling=CV(nfolds=5, rng=123),
                acceleration=CPUThreads(),
                verbosity=2)
PerformanceEvaluation object with these fields:
  measure, measurement, operation, per_fold,
  per_observation, fitted_params_per_fold,
  report_per_fold, train_test_pairs
Extract:
┌───────────────┬─────────────┬───────────┬───────────────────────────────────────────────┐
│ measure       │ measurement │ operation │ per_fold                                      │
├───────────────┼─────────────┼───────────┼───────────────────────────────────────────────┤
│ LPLoss(p = 1) │ 17200.0     │ predict   │ [16500.0, 17100.0, 16300.0, 17500.0, 18900.0] │
│ LPLoss(p = 2) │ 6.83e8      │ predict   │ [6.14e8, 6.64e8, 5.98e8, 6.37e8, 9.03e8]      │
└───────────────┴─────────────┴───────────┴───────────────────────────────────────────────┘
```

## Key goals

* Offer a consistent way to use, compose and tune machine learning
  models in Julia,

* Promote the improvement of the Julia ML/Stats ecosystem by making it
  easier to use models from a wide range of packages,

* Unlock performance gains by exploiting Julia's support for
  parallelism, automatic differentiation, GPU, optimization etc.


## Key features

* Data agnostic, train most models on any data `X` supported by the
  [Tables.jl](https://github.com/JuliaData/Tables.jl) interface (needs `Tables.istable(X)
  == true`).

* Extensive, state-of-the-art, support for model composition
  (*pipelines*, *stacks* and, more generally, *learning networks*). See more
  [below](#model-composability).

* Convenient syntax to tune and evaluate (composite) models.

* Consistent interface to handle probabilistic predictions.

* Extensible [tuning
  interface](https://github.com/JuliaAI/MLJTuning.jl),
  to support a growing number of optimization strategies, and designed
  to play well with model composition.

* Options to accelerate model evaluation and tuning with
  multithreading and/or distributed processing.


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
Learning](https://github.com/JuliaAI/MLJ.jl) and tag
your issue "mlj". Queries can also be posted as
[issues](https://github.com/JuliaAI/MLJ.jl/issues), or
on the `#mlj` slack workspace in the Julia Slack channel.

Bugs, suggestions, and feature requests can be posted
[here](https://github.com/JuliaAI/MLJ.jl/issues).

Users are also welcome to join the `#mlj` Julia slack channel to ask
questions and make suggestions.


## Installation

Initially, it is recommended that MLJ and associated packages be
installed in a new
[environment](https://julialang.github.io/Pkg.jl/v1/environments/) to
avoid package conflicts. You can do this with

```julia-repl
julia> using Pkg; Pkg.activate("my_MLJ_env", shared=true)
```

Installing MLJ is also done with the package manager:

```julia-repl
julia> Pkg.add("MLJ")
```

**Optional:** To test your installation, run

```julia-repl
julia> Pkg.test("MLJ")
```

It is important to note that MLJ is essentially a big wrapper
providing unified access to _model-providing packages_. For this
reason, one generally needs to add further packages to your
environment to make model-specific code available. This
happens automatically when you use MLJ's interactive load command
`@iload`, as in

```julia-repl
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

MLJ is supported by several satellite packages (MLJTuning,
MLJModelInterface, etc) which the general user is *not* required to
install directly. Developers can learn more about these
[here](https://github.com/JuliaAI/MLJ.jl/blob/master/ORGANIZATION.md).

See also the alternative installation instructions for [Modifying Behavior](@ref).


## Funding

MLJ was initially created as a Tools, Practices and Systems project at
the [Alan Turing Institute](https://www.turing.ac.uk/)
in 2019. Current funding is provided by a [New Zealand Strategic
Science Investment
Fund](https://www.mbie.govt.nz/science-and-technology/science-and-innovation/funding-information-and-opportunities/investment-funds/strategic-science-investment-fund/ssif-funded-programmes/university-of-auckland/)
awarded to the University of Auckland.


## Citing MLJ

An overview of MLJ design:


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

An in-depth view of MLJ's model composition design:

[![arXiv](https://img.shields.io/badge/arXiv-2012.15505-<COLOR>.svg)](https://arxiv.org/abs/2012.15505)

```bibtex
@misc{blaom2020flexible,
      title={Flexible model composition in machine learning and its implementation in {MLJ}},
      author={Anthony D. Blaom and Sebastian J. Vollmer},
      year={2020},
      eprint={2012.15505},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
