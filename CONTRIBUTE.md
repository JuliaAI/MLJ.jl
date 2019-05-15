## Contributing to the MLJ machine learning project

Contributions to MLJ are most welcome. Queries can be made through
issues or the Julia [slack
channel](https://slackinvite.julialang.org), #MLJ.


- [List of presently implemented models](https://github.com/alan-turing-institute/MLJRegistry.jl/blob/dev/Models.toml), excluding built-ins (or run `models()` in MLJ)

- [Enhancement requests](https://github.com/alan-turing-institute/MLJ.jl/issues?utf8=✓&q=is%3Aissue+is%3Aopen+label%3A%22enhancement%22)

- [Nice overview of Julia ML packages](https://www.simonwenkel.com/2018/10/05/Julia-for-datascience-machine-learning-and-artificial-intelligence.html)

While new model implementations are a priority at present, help adding
core functionality to MLJ is also welcome. If you are interested in
contributing, please read the this rest of this document. A guide to
implementing the MLJ interface for new models is
[here](docs/src/adding_models_for_general_use.md).


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

The MLJ ecosystem is currently spread across four repositories:

- [MLJ](https://github.com/alan-turing-institute/MLJ.jl) is the
  ordinary user's point-of-entry. It implements the meta-algorithms
  (resampling, tuning, learning networks, etc).
  
- [MLJBase](https://github.com/alan-turing-institute/MLJBase.jl)
  defines the model interface which new algorithms must implement to
  participate in MLJ. 
  
- [MLJRegistry](https://github.com/alan-turing-institute/MLJRegistry.jl)
  stores metadata on all models "registered" with MLJ, which become
  available to the MLJ user through the task interface, before external
  packages implementing the models need be loaded. This is for MLJ
  administrators.
  
- [MLJModels](https://github.com/alan-turing-institute/MLJModels.jl)
  contains the implementation code for models in external packages
  that do not natively support the MLJ interface.


### Current state of the project and directions (mid May 2019)

The MLJ model interface is now reasonably stable and well documented,
and core functionality is now also in place. The success of the
project currently **depends crucially on new Julia ML algorithms
implementing the MLJ interface**, which in large part will depend on
contributions from the larger community. While ideally external
packages natively implement the MLJ interface, implementation code
PR's to MLJModels is also welcome; see the end of
[here](docs/src/adding_models_for_general_use.md) for more on these
two options.

#### Short-term goals for improving functionality

- add benchmarking

- add common learning network architectures as stand-alone models with
  macros for quick instantiation
  
- replace existing metrics with LossFunctions with an enhanced
  probabilistic API (e.g., proper scoring rules)
  
Longer term goals, are likely to be driven by end-user feedback; refer or add to the [Enhancement
requests](https://github.com/alan-turing-institute/MLJ.jl/issues?utf8=✓&q=is%3Aissue+is%3Aopen+label%3A%22enhancement%22).


#### Funding

Development of MLJ is currently sponsored through the Alan Turing
Institute “Machine Learning in Julia" project, in cooperation with
Julia Computing. 


#### The team at Turing

- Sebastian Vollmer (director, consultant)
- Franz Kiraly (director, consultant)
- Anthony Blaom, through an arrangement with NeSI (lead contributor, coordinator) 
- Yiannis Simillides (contributor)
- Ayush Shridar, UCL summer intern (contributor)

#### At Julia Computing

- Mike Innes

#### Other contributors, past and present

Diego Arenas, Edoardo Barp, Gergö Bohner, Michael K. Borregaard,
Valentin Churavy, Harvey Devereux, Mosè Giordano, Thibaut Lienart,
Mohammed Nook, Annika Stechemesser, Ayush Shridar, Yiannis Simillides


