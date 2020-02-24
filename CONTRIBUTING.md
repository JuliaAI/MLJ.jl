## Contributing to the MLJ machine learning project

Contributions to MLJ are most welcome. Queries can be made through
issues or the Julia [slack
channel](https://slackinvite.julialang.org), #MLJ. 

- [Road map](ROADMAP.md)

- [Code organization](ORGANIZATION.md)


### Git work-flow

**Important.** Please make pull requests to the `dev` branch
(**not** `master`) of the appropriate repo. 

All pull requests onto `master` come from `dev` and generally precede a new release. 

We follow [this git
work-flow](https://nvie.com/posts/a-successful-git-branching-model/).


### Very brief design overview

MLJ has a basement level *model* interface, which must be implemented
for each new learning algorithm. Formally, each model is a `mutable
struct` storing hyperparameters and the implementer defines
model-dispatched `fit` and `predict` methods; for details, see
[here](docs/src/adding_models_for_general_use.md). The general user
interacts using *machines* which bind models with data and have an
internal state reflecting the outcomes of applying `fit!` and
`predict` methods on them. The model interface is pure "functional";
the machine interface more "object-oriented".

A generalization of machine, called a *nodal* machine, is a key
element of *learning networks* which combine several models
together. See
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/)
for more on these.

MLJ code is now spread over [multiple repositories](ORGANIZATION.md).


