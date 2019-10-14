## Contributing to the MLJ machine learning project

Contributions to MLJ are most welcome. Queries can be made through
issues or the Julia [slack
channel](https://slackinvite.julialang.org), #MLJ.


- [List of presently implemented
  models](https://github.com/alan-turing-institute/MLJModels.jl/tree/master/src/registry/Models.toml). Or, do `using MLJ; models()`.

- [Enhancement requests](https://github.com/alan-turing-institute/MLJ.jl/issues?utf8=✓&q=is%3Aissue+is%3Aopen+label%3A%22enhancement%22)

- [Nice overview of Julia ML packages](https://www.simonwenkel.com/2018/10/05/Julia-for-datascience-machine-learning-and-artificial-intelligence.html)

While new model implementations are a priority at present, help adding
core functionality to MLJ is also welcome. If you are interested in
contributing, please read the this rest of this document. A guide to
implementing the MLJ interface for new models is
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/).


### Brief design overview

MLJ has a basement level *model* interface, which must be implemented
for each new learning algorithm. Formally, each model is a `mutable
struct` storing hyperparameters and the implementer defines
model-dispatched `fit` and `predict` methods; for details, see
[here](docs/src/adding_models_for_general_use.md). The general user
interacts using a *machine* interface using `fit!` and `predict`
methods, dispatched on machines. A machine wraps a model in data (or a
*task*) and the results of training. The model interface has a
functional style, the machine interface is more "object-oriented".

A generalization of machine, called a *nodal* machine, is the key
element of *learning networks* which combine several models
together. See the [tour](docs/src/tour.ipynb) for more on these.

The core MLJ ecosystem is currently spread across three repositories:

- [MLJ](https://github.com/alan-turing-institute/MLJ.jl) is the
  ordinary user's point-of-entry. It implements the meta-algorithms
  (resampling, tuning, learning networks, etc).
  
- [MLJBase](https://github.com/alan-turing-institute/MLJBase.jl)
  defines the model interface which new algorithms must implement to
  participate in MLJ. 
    
- [MLJModels](https://github.com/alan-turing-institute/MLJModels.jl)
  contains the implementation code for models in external packages
  that do not natively support the MLJ interface.



### Road map (early August 2019)

The MLJ model interface is now reasonably stable and well documented,
and core functionality is now also in place. The success of the
project currently **depends crucially on new Julia ML algorithms
implementing the MLJ interface**, which in large part will depend on
contributions from the larger community. While ideally external
packages natively implement the MLJ interface, implementation code
PR's to MLJModels is also welcome; see the end of
[here](docs/src/adding_models_for_general_use.md) for more on these
two options.

#### Enhancing functionality: Adding models

-  Wrap the scit-learn (python/C) models (WIP: Z.~Nugent, D.~Arenas)
-  Flux.jl deep learning (WIP: A.~Shridhar)
-  Turing.jl probabilistic programming (WIP: M.~Trapp) or other PP pkgs
-  Geostats.jl (WIP: J.~Hoffimann)
-  Time series models
-  Data imputation (LowRankModels.jl?)
-  Feature engineering (featuretools?)

#### Enhancing core functionality

-  Systematic benchmarking
-  More comprehensive performance evaluation
-  Tuning using Bayesian optimization
-  Tuning using gradient descent and AD
-  Iterative model control
-  Serialization and deserialization of trained models

#### Broadening scope

-  Extend or supplement LossFunctions.jl to support probabilistic losses
-  Add sparse data support (NLP)

#### Improving scalability

-  Online learning support and distributed data
-  DAG scheduling (WIP: J.~Samaroo)
-  Automated estimates of cpu/memory requirements



See also the long wish list of [Feature enhancement
requests](https://github.com/alan-turing-institute/MLJ.jl/issues?utf8=✓&q=is%3Aissue+is%3Aopen+label%3A%22enhancement%22).



### Funding

Development of MLJ is currently sponsored through the Alan Turing
Institute “Machine Learning in Julia" project, in cooperation with
Julia Computing and New Zealand eScience Infrastructure.


### Authors

**Core design.** Anthony Blaom, Franz Kiraly, Sebastian Vollmer

**Lead programmer.** Anthony Blaom

**Julia language consultants.** Mike Innes, Avik Sengupta

**Other contributors, past and present.** Dilum Aluthge, Diego
    Arenas, Edoardo Barp, Gergö Bohner, Michael K. Borregaard,
    Valentin Churavy, Harvey Devereux, Mosè Giordano, Thibaut Lienart,
    Mohammed Nook, Piotr Oleśkiewicz, Julian Samaroo, Ayush Shridar,
    Yiannis Simillides, Annika Stechemesser






