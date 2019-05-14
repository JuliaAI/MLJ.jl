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
implementing the MLJ inteface for new models is
[here](docs/src/adding_models_for_general_use.md).


### Brief design overview

MLJ has a basement level *model* interface, which must be implemented
for each new learning algorithm. Formally, each model is a `mutable
struct` storing hyperparameters and the implementer defines
model-dispatched `fit` and `predict` methods; for details, see
[here](docs/src/adding_models_for_general_use.md). The user interacts through a *task*
interface (work-in-progress) and a *machine* interface using `fit!`
and `predict` methods, dispatched on machines. A machine wraps a model
in data and the results of training. The model interface has a
functional style, the machine interface is more object-oriented.

A generalization of machine, called a *nodal* machine, is the key
element of *learning networks* which combine several models
together. See the [tour](docs/src/tour.ipynb) for more on these.

The MLJ ecosystem is currently spread across four repostitories:

- [MLJ](https://github.com/alan-turing-institute/MLJ.jl) is the
  ordinary user's point-of-entry. It implements the meta-algorithms
  (resampling, tuning, learing networks, etc).
  
- [MLJBase](https://github.com/alan-turing-institute/MLJBase.jl)
  defines the model interface which new algorithims must implement to
  participate in MLJ.
  
- [MLJRegistry](https://github.com/alan-turing-institute/MLJRegistry.jl)
  stores metadata on all models "registered" with MLJ, which become
  available to the MLJ user through the task inteface, before external
  packages implementing the models need be loaded.
  
- [MLJModels](https://github.com/alan-turing-institute/MLJModels.jl)
  contains the implementation code for models in external packages
  that do not natively support the MLJ interface.


### Current state of the project and directions (early Feb 2019)

The long-term goal is for external packages to natively implement the
MLJ interface for their models (and in principle they can already do
so). Currently, model interface implementations are either built in
(see [src/builtin](src/builtin)) or lazy-loaded from MLJModels using
Requires.jl.

The project is presently in the transition stage: The model API is
mostly fixed, but small changes are still being driven by experiences
implementing new models.

#### Funding

Development of MLJ is currently sponsored through the Alan Turing
Institute “Machine Learning in Julia" project, in cooperation with
Julia Computing. 


#### The team at Turing

- Sebastian Vollmer (director, consultant)
- Franz Kiraly (director, consultant)
- Anthony Blaom (lead contributor, coordinator)
- Yiannis Simillides (contributor)


#### At Julia Computing

- Mike Innes

#### Other contributors, past and present

Diego Arenas, Edoardo Barp, Gergö Bohner, Michael K. Borregaard,
Valentin Churavy, Harvey Devereux, Mosè Giordano, Thibaut Lienart,
Mohammed Nook, Annika Stechemesser, Ayush Shridar, Yiannis Simillides


