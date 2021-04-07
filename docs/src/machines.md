# Machines

Recall from [Getting Started](@ref) that a machine binds a model
(i.e., a choice of algorithm + hyperparameters) to data (see more at
[Constructing machines](@ref) below). A machine is also the object
storing *learned* parameters.  Under the hood, calling `fit!` on a
machine calls either `MLJBase.fit` or `MLJBase.update`, depending on
the machine's internal state (as recorded in private fields
`old_model` and `old_rows`). These lower-level `fit` and `update`
methods, which are not ordinarily called directly by the user,
dispatch on the model and a view of the data defined by the optional
`rows` keyword argument of `fit!` (all rows by default). 

# Warm restarts

If a model `update` method has been implemented for the model, calls
to `fit!` will avoid redundant calculations for certain kinds of model
mutations. The main use-case is increasing an iteration parameter,
such as the number of epochs in a neural network. To test if
`SomeIterativeModel` supports this feature, check
`iteration_parameter(SomeIterativeModel)` is different from `nothing`.

```@example machines
using MLJ; color_off() # hide
tree = (@load DecisionTreeClassifier pkg=DecisionTree verbosity=0)()
forest = EnsembleModel(atom=tree, n=10);
X, y = @load_iris;
mach = machine(forest, X, y)
fit!(mach, verbosity=2);
```

Generally, changing a hyperparameter triggers retraining on calls to
subsequent `fit!`:

```@repl machines
forest.bagging_fraction=0.5
fit!(mach, verbosity=2);
```

However, for this iterative model, increasing the iteration parameter
only adds models to the existing ensemble:

```@repl machines
forest.n=15
fit!(mach, verbosity=2);
```

Call `fit!` again without making a change and no retraining occurs:

```@repl machines
fit!(mach);
```

However, retraining can be forced:

```@repl machines
fit!(mach, force=true);
```

And is re-triggered if the view of the data changes:

```@repl machines
fit!(mach, rows=1:100);
```

```@repl machines
fit!(mach, rows=1:100);
```

If an iterative model exposes it's iteration parameter as a
hyper-parameter, and it implements the warm restart behaviour above,
then it can be wrapped in a "control strategy", like an early stopping
critetion. See [Controlling Iterative Models](@ref) for details.


## Inspecting machines

There are two methods for inspecting the outcomes of training in
MLJ. To obtain a named-tuple describing the learned parameters (in a
user-friendly way where possible) use `fitted_params(mach)`. All other
training-related outcomes are inspected with `report(mach)`.

```@example machines
X, y = @load_iris
pca = (@load PCA verbosity=0)()
mach = machine(pca, X)
fit!(mach)
```

```@repl machines
fitted_params(mach)
report(mach)
```

```@docs
fitted_params
report
```


## Constructing machines

A machine is constructed with the syntax `machine(model, args...)`
where the possibilities for `args` (called *training arguments*) are
summarized in table below. Here `X` and `y` represent inputs and
target, respectively, and `Xout` the output of a `transform` call.
Machines for supervised models may have additional training arguments,
such as a vector of per-observation weights (in which case
`supports_weights(model) == true`).

`model` supertype   | `machine` constructor calls | operation calls (first compulsory)
--------------------|-----------------------------|--------------------------------------
`Deterministic <: Supervised`    | `machine(model, X, y, extras...)` | `predict(mach, Xnew)`, `transform(mach, Xnew)`, `inverse_transform(mach, Xout)`
`Probabilistic <: Supervised`    | `machine(model, X, y, extras...)` | `predict(mach, Xnew)`, `predict_mean(mach, Xnew)`, `predict_median(mach, Xnew)`, `predict_mode(mach, Xnew)`, `transform(mach, Xnew)`, `inverse_transform(mach, Xout)`
`Unsupervised` (except `Static`) | `machine(model, X)` | `transform(mach, Xnew)`, `inverse_transform(mach, Xout)`, `predict(mach, Xnew)`
`Static`                        | `machine(model)`    | `transform(mach, Xnews...)`, `inverse_transform(mach, Xout)`

All operations on machines (`predict`, `transform`, etc) have exactly
one argument (`Xnew` or `Xout` above) after `mach`, the machine
instance. An exception is a machine bound to a `Static` model, which
can have any number of arguments after `mach`. For more on `Static`
transformers (which have no *training* arguments) see [Static
transformers](@ref).

A machine is reconstructed from a file using the syntax
`machine("my_machine.jlso")`, or `machine("my_machine.jlso", args...)`
if retraining using new data. See [Saving machines](@ref) below.


## Lowering memory demands

For large data sets you may be able to save memory by suppressing data
caching that some models perform to increase speed. To do this,
specify `cache=false`, as in

```julia
machine(model, X, y, cache=false)
```


### Constructing machines in learning networks

Instead of data `X`, `y`, etc,  the `machine` constructor is provided
`Node` or `Source` objects ("dynamic data") when building a learning
network. See [Composing Models](composing_models.md) for more on this
advanced feature. One also uses `machine` to wrap a machine
around a whole learning network; see [Learning network
machines](@ref).


## Saving machines

To save a machine to file, use the `MLJ.save` command:

```julia
tree = (@load DecisionTreeClassifier pkg=DecisionTree verbosity=0)()
mach = fit!(machine(tree, X, y))
MLJ.save("my_machine.jlso", mach)
```

To de-serialize, one uses the `machine` constructor:

```julia
mach2 = machine("my_machine.jlso")
predict(mach2, Xnew);
```

The machine `mach2` cannot be retrained; however, by providing data to
the constructor one can enable retraining using the saved model
hyperparameters (which overwrites the saved learned parameters):

```julia
mach3 = machine("my_machine.jlso", Xnew, ynew)
fit!(mach3)
```


## Internals

For a supervised machine the `predict` method calls a lower-level
`MLJBase.predict` method, dispatched on the underlying model and the
`fitresult` (see below). To see `predict` in action, as well as its
unsupervised cousins `transform` and `inverse_transform`, see
[Getting Started](index.md).

The fields of a `Machine` instance (which should not generally be
accessed by the user) are:

- `model` - the struct containing the hyperparameters to be used in
  calls to `fit!`

- `fitresult` - the learned parameters in a raw form, initially undefined

- `args` - a tuple of the data, each element wrapped in a source node;
  see [Learning Networks](@ref) (in the supervised learning example
  above, `args = (source(X), source(y))`)

- `report` - outputs of training not encoded in `fitresult` (eg, feature rankings)

- `old_model` - a deep copy of the model used in the last call to `fit!`

- `old_rows` -  a copy of the row indices used in last call to `fit!`

- `cache`

The interested reader can learn more on machine internals by examining
the simplified code excerpt in [Internals](internals.md).


## API Reference

```@docs
MLJBase.machine
fit!
fit_only!
MLJSerialization.save
```
