## Contributing to the MLJ machine learning project

Contributions to MLJ are most welcome. Queries can be made through
issues or the Julia [slack
channel](https://slackinvite.julialang.org), #MLJ.


There is a [list]() of presently implemented models/packages. Please
add to the new [model wish list](), preferably including a brief plug
for the model. Even better, write a pull-request for an implementation
yourself!

While new model implementations are a priority, help adding core
functionality to MLJ is also welcome. If you are interested in
contributing, please read the "Design Overview" below.


### Feature wish list (in no particular order)

- Add visualisation tools

- Add random search tuning strategy

- Add genetic algorithm tuning strategy 

- Explore possibilities for architecture search

- Integrate with Flux (both "high" level for the Flux expert and
  plug-and-play "low" level options). Recurrent networks, adversarial networks, etc

- Add model selection tools, benchmarking

- Add DAG scheduling, eg Dagger.jl, for training learning networks.

- Add tools to estimate resource requirements (like MLR "learning curves")

- Integrate interpretable machine learning tools, such as Shapley values

- Implement the task interface for matching models to tasks

- Implement a gradient descent tuning strategy using automatic
  differentiation (for cure Julie algorithms).

- Bootstrapping from Sklearn and mlr by wrapping with task info

Development of MLJ is currently sponsored through the Alan Turing
Institute “Machine Learning in Julia" project, in cooperation with Julia Computing.


### Brief design overview

MLJ has a basement level *model* interface, which must be implemented
for each new learning algorithm. Formally, each model is a `mutable
struct` storing hyperparameters and the implementer defines
model-dispatched `fit` and `predict` methods; for details, see
[here](doc/adding_new_models.md). The user interacts through a
*machine* interface using `fit!` and `predict` methods, dispatched on
machines. A machine wraps a model in data and the results of training. The
model interface has a functional style, the machine interface is more
object-oriented.

A generalization of machine, called a *nodal* machine, is the key
element of *learning networks* which combine several models
together. See the [tour](doc/tour.ipynb) for more on these.


### Current state of the project and directions (early Feb 2019)

The long-term goal is for external packages to natively implement the
MLJ interface for their models (and in principle they can already do
so). Currently, model interface implementations are either built in
(see [src/builtin](src/builtin)) or lazy-loaded using Requires.jl (see
[src/interfaces](src/interfaces)). The lazy-loaded implementations
will soon move to a new repository, MLJModels.

The project is presently in the transition stage: The model API is
mostly fixed, but small changes are still being driven by experiences
implementing new models.


#### The team at Turing

- Sebastian Vollmer (director, consultant)
- Franz Kiraly (director, consultant)
- Anthony Blaom (coordinating contributor)
- Yiannis Simillides (contributor)

#### At Julia Computing

- Mike Innes

#### Other contributors, past and present

Diego Arenas, Edoardo Barp, Gergö Bohner, Valentin Churvay, Harvey
Devereux, Thibaut Lienart, Mohammed Nook, Annika Stechemesser, Ayush
Shridar, Yiannis Simillides


