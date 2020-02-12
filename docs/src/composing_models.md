# Composing Models

MLJ has a flexible interface for composing multiple machine learning
elements to form a *learning network*, whose complexity can extend
beyond the "pipelines" of other machine learning toolboxes. While
these learning networks can be applied directly to learning tasks,
they are more commonly used to specify new re-usable, stand-alone,
composite model types, that behave like any other model type. The main
novelty of composite models is that they include other models as
hyper-parameters.

That said, MLJ also provides dedicated syntax for the most common
composition use-cases, which are described first below. A description
of the general framework begins at [Learning Networks](@ref).

## Linear pipelines

In MLJ a *pipeline* is a composite model in which models are chained
together in a linear (non-branching) chain. Pipelines can include
learned or static target transformations, if one of the models is
supervised.

To illustrate basic construction of a pipeline, consider the following
toy data:

```@setup 7
using MLJ
MLJ.color_off()
```

```@example 7
using MLJ
X = (age    = [23, 45, 34, 25, 67],
     gender = categorical(['m', 'm', 'f', 'm', 'f']));
height = [67.0, 81.5, 55.6, 90.0, 61.1];
```

The code below creates a new pipeline model type called `MyPipe` for
performing the following operations:

- standardize the target variable `:height` to have mean zero and
  standard deviation one
- coerce the `:age` field to have `Continuous` scitype
- one-hot encode the categorical feature `:gender`
- train a K-nearest neighbor model on the transformed inputs and
  transformed target
- restore the predictions of the KNN model to the original `:height`
  scale (i.e., invert the standardization)

The code also creates an instance of the new pipeline model type,
called `pipe`, whose hyperparameters `hot`, `knn`, and `stand` are the
component model instances specified in the macro expression:

```@example 7
@load KNNRegressor # hide
```

```julia
julia> pipe = @pipeline MyPipe(X -> coerce(X, :age=>Continuous),
                               hot = OneHotEncoder(),
                               regressor = KNNRegressor(K=3),
                               target = UnivariateStandardizer())

MyPipe(hot = OneHotEncoder(features = Symbol[],
                           drop_last = false,
                           ordered_factor = true,),
       regressor = KNNRegressor(K = 3,
                          metric = MLJModels.KNN.euclidean,
                          kernel = MLJModels.KNN.reciprocal,),
       target = UnivariateStandardizer(),) @ 1…16
```

We can, for example, evaluate the pipeline like we would any other model:

```julia
julia> pipe.regressor.K = 2
julia> pipe.hot.drop_last = true
julia> evaluate(pipe, X, height, resampling=Holdout(), measure=rms, verbosity=2)

[ Info: Training Machine{MyPipe} @ 4…44.
[ Info: Training NodalMachine{OneHotEncoder} @ 1…16.
[ Info: Spawning 1 sub-features to one-hot encode feature :gender.
[ Info: Training NodalMachine{UnivariateStandardizer} @ 5…65.
[ Info: Training NodalMachine{KNNRegressor} @ 1…49.
(measure = MLJBase.RMS[rms],
 measurement = [10.0336],
 per_fold = Array{Float64,1}[[10.0336]],
 per_observation = Missing[missing],)
```

Incidentally, there is nothing preventing the user from replacing the
regressor component in this pipeline with different deterministic
regressor:

```julia
julia> pipe.regressor = @load RidgeRegressor pkg=MultivariateStats
julia> pipe
```

```julia
MyPipe(hot = OneHotEncoder(features = Symbol[],
                           drop_last = false,
                           ordered_factor = true,),
       regressor = RidgeRegressor(lambda = 1.0,),
       target = UnivariateStandardizer(),) @ 1…16
```

For important details on including target transformations, see below.

```@docs
@pipeline
```

## Homogeneous Ensembles

For performance reasons, creating a large ensemble of models sharing a
common set of hyperparameters is achieved in MLJ through a model
wrapper, rather than through the learning networks API. See the
separate [Homogeneous Ensembles](homogeneous_ensembles.md) section for
details.


## Learning Networks

Hand-crafting a learning network, as outlined below, is a relatively
advanced MLJ feature, assuming familiarity with the basics outlined in
[Getting Started](index.md). The syntax for building a learning
network is essentially an extension of the basic syntax but with data
containers replaced with nodes ("dynamic data").

In MLJ, a *learning network* is a directed acyclic graph whose nodes
apply an operation, such as `predict` or `transform`, using a fixed
machine (requiring training) - or which, alternatively, applies a
regular (untrained) mathematical operation, such as `+`, `log` or
`vcat`, to its input(s). In practice, a learning network works with
fixed sources for its training/evaluation data, but can be built and
tested in stages. By contrast, an *exported learning network* is a
learning network exported as a stand-alone, re-usable `Model` object,
to which all the MLJ `Model` meta-algorithms can be applied
(ensembling, systematic tuning, etc).

Different nodes can point to the same machine (i.e., can access a
common set of learned parameters) and different machines can wrap a
common model (allowing for hyperparameters in different machines to be
coupled).

By specifying data at the source nodes of a learning network, one can
use and test the learning network as it is defined, which is also a
good way to understand how learning networks work under the hood. This
data, if specified, is ignored in the export process, for the exported
composite model, like any other model, is not associated with any data
until wrapped in a machine.

In MLJ learning networks treat the flow of information during training
and predicting separately. Also, different nodes may use the same
parameters (fitresult) learned during the training of some model (that
is, point to a common *nodal machine*; see below). For these reasons,
simple examples may appear more slightly more complicated than in
other frameworks. However, in more sophisticated applications, the
extra flexibility is essential.


### Building a simple learning network

![](img/wrapped_ridge.png)

The diagram above depicts a learning network which standardizes the
input data `X`, learns an optimal Box-Cox transformation for the
target `y`, predicts new target values using ridge regression, and
then inverse-transforms those predictions, for later comparison with
the original test data. The machines, labeled in yellow, are where
data to be used for training enters a node, and where training
outcomes are stored, as in the basic fit/predict scenario.

Looking ahead, we note that the new composite model type we will
create later will be assigned a single hyperparameter `regressor`, and the
learning network model `RidgeRegressor(lambda=0.1)` will become this
parameter's default value. Since model hyperparameters are mutable,
this regressor can be changed to a different one (e.g.,
`HuberRegressor()`).

For testing purposes, we'll use a small synthetic data set:

```@example 7
using Statistics, DataFrames
@load RidgeRegressor pkg=MultivariateStats # hide
```

```julia
x1 = rand(300)
x2 = rand(300)
x3 = rand(300)
y = exp.(x1 - x2 -2x3 + 0.1*rand(300))
X = DataFrame(x1=x1, x2=x2, x3=x3)

train, test  = partition(eachindex(y), 0.8)

Xs = source(X)
ys = source(y, kind=:target)
```

```julia
Source @ 3…40
```

*Note.* One can omit the specification of data at the source nodes (by
writing instead `Xs = source()` and `ys = source(kind=:target)`) and
still export the resulting network as a stand-alone model using the
@from_network macro described later; see the example under [Static
operations on nodes](@ref). However, one will be unable to fit
or call network nodes, as illustrated below.

We label the nodes that we will define according to their outputs in
the diagram. Notice that the nodes `z` and `yhat` use the same
machine, namely `box`, for different operations.

To construct the `W` node we first need to define the machine `stand`
that it will use to transform inputs.

```julia
stand_model = Standardizer()
stand = machine(stand_model, Xs)
```

```julia
NodalMachine @ 6…82 = machine(Standardizer{} @ 1…82, 3…40)
```

Because `Xs` is a node, instead of concrete data, we can call
`transform` on the machine without first training it, and the result
is the new node `W`, instead of concrete transformed data:

```julia
W = transform(stand, Xs)
```

```julia
Node @ 1…67 = transform(6…82, 3…40)
```

To get actual transformed data we *call* the node appropriately, which
will require we first train the node. Training a node, rather than a
machine, triggers training of *all* necessary machines in the network.


```julia
fit!(W, rows=train)
W()           # transform all data
W(rows=test ) # transform only test data
W(X[3:4,:])   # transform any data, new or old
```

```julia
2×3 DataFrame
│ Row │ x1        │ x2       │ x3        │
│     │ Float64   │ Float64  │ Float64   │
├─────┼───────────┼──────────┼───────────┤
│ 1   │ -0.516373 │ 0.675257 │ 1.27734   │
│ 2   │ 0.63249   │ -1.70306 │ 0.0479891 │
```

If you like, you can think of `W` (and the other nodes we will define)
as "dynamic data": `W` is *data*, in the sense that it an be called
("indexed") on rows, but *dynamic*, in the sense the result depends on
the outcome of training events.

The other nodes of our network are defined similarly:

```julia
box_model = UnivariateBoxCoxTransformer()  # for making data look normally-distributed
box = machine(box_model, ys)
z = transform(box, ys)

ridge_model = RidgeRegressor(lambda=0.1)
ridge =machine(ridge_model, W, z)
zhat = predict(ridge, W)

yhat = inverse_transform(box, zhat)
```

```julia
Node @ 1…07 = inverse_transform(1…09, predict(2…66, transform(6…82, 3…40)))
```

We are ready to train and evaluate the completed network. Notice that
the standardizer, `stand`, is *not* retrained, as MLJ remembers that
it was trained earlier:


```julia
fit!(yhat, rows=train)
```

```julia
[ Info: Not retraining NodalMachine{Standardizer} @ 6…82. It is up-to-date.
[ Info: Training NodalMachine{UnivariateBoxCoxTransformer} @ 1…09.
[ Info: Training NodalMachine{RidgeRegressor} @ 2…66.
Node @ 1…07 = inverse_transform(1…09, predict(2…66, transform(6…82, 3…40)))
```

```julia
rms(y[test], yhat(rows=test)) # evaluate
```

```julia
0.022837595088079567
```

We can change a hyperparameters and retrain:

```julia
ridge_model.lambda = 0.01
fit!(yhat, rows=train)
```

```julia
[ Info: Not retraining NodalMachine{UnivariateBoxCoxTransformer} @ 1…09. It is up-to-date.
[ Info: Not retraining NodalMachine{Standardizer} @ 6…82. It is up-to-date.
[ Info: Updating NodalMachine{RidgeRegressor} @ 2…66.
Node @ 1…07 = inverse_transform(1…09, predict(2…66, transform(6…82, 3…40)))
```

And re-evaluate:

```julia
rms(y[test], yhat(rows=test))
0.039410306910269116
```

> **Notable feature.** The machine, `ridge::NodalMachine{RidgeRegressor}`, is retrained, because its underlying model has been mutated. However, since the outcome of this training has no effect on the training inputs of the machines `stand` and `box`, these transformers are left untouched. (During construction, each node and machine in a learning network determines and records all machines on which it depends.) This behavior, which extends to exported learning networks, means we can tune our wrapped regressor (using a holdout set) without re-computing transformations each time the hyperparameter is changed.

### Learning networks with sample weights

To build an exportable learning network supporting sample weights,
create a source node with `ws = source(w; kind=:weights)` or `ws =
source(; kind=weights)`.


## Exporting a learning network as a stand-alone model

Having satisfied that our learning network works on the synthetic
data, we are ready to export it as a stand-alone model.


### Method I: The @from_network macro

The following call simultaneously defines a new model subtype
`WrappedRegressor <: Supervised` and returns an instance of this type,
bound to `wrapped_regressor`:

```julia
wrapped_regressor = @from_network WrappedRegressor(regressor=ridge_model) <= yhat
```

```julia
WrappedRegressor(regressor = RidgeRegressor(lambda = 1.0,),) @ 2…63
```

Any MLJ work-flow can be applied to this composite model:

```julia
X, y = @load_boston
evaluate(wrapped_regressor, X, y, resampling=CV(), measure=rms, verbosity=0)
```

```julia
(measure = MLJBase.RMS[rms],
 measurement = [5.26949],
 per_fold = Array{Float64,1}[[3.02163, 4.75385, 5.01146, 4.22582, 8.93383, 3.47707]],
 per_observation = Missing[missing],)
```

*Notes:*

- A deep copy of the original learning network `ridge_model` has
  become the default value for the field `regressor` of the new
  `WrappedRegressor` struct.

- It is important to have labeled the target source, as in `ys = source(y,
  kind=:target)`, to ensure the network is exported as a *supervised*
  model.

- One can can also use the `@from_network` to export unsupervised
  learning networks and the syntax is the same. For example:

```julia
langs_composite = @from_network LangsComposite(pca=network_pca) <= Xout
```

- For a supervised network making *probabilistic* predictions, one
must add `prediction_type=:probabilistic` to the end of the `@from
network` call. For example:

```julia
petes_composite = @from_network PetesComposite(tree_classifier=network_tree) prediction_type=:probabilistic
```

Returning to the `WrappedRegressor` model, we can change the regressor
being wrapped if so desired:

```julia
wrapped_rgs.regressor = KNNRegressor(K=7)
wrapped_rgs
```

```julia
WrappedRegressor(regressor = KNNRegressor(K = 7,
                                          algorithm = :kdtree,
                                          metric = Distances.Euclidean(0.0),
                                          leafsize = 10,
                                          reorder = true,
                                          weights = :uniform,),) @ 2…63
```


### Method II: Finer control (advanced)

This section described an advanced feature that can be
skipped on a first reading.

In Method I above, only models appearing in the network will appear as
hyperparameters of the exported composite model. There is a second
more flexible method for exporting the network, which allows finer
control over the exported `Model` struct, and which also avoids
macros. The two steps required are:

- Define a new `mutable struct` model type.

- Wrap the learning network code in a model `fit` method.

We now demonstrate this second method to the preceding example. To
see how to use the method to expose user-specified hyperparameters
that are not component models, see
[here](https://alan-turing-institute.github.io/MLJTutorials/pub/end-to-end/AMES.html#tuning_the_model).

All learning networks that make deterministic (respectively,
probabilistic) predictions export to models of subtype
`DeterministicNetwork` (respectively, `ProbabilisticNetwork`),
Unsupervised learning networks export to `UnsupervisedNetwork` model
subtypes. So our `mutable struct` definition looks like this:

```@example 7
mutable struct WrappedRegressor2 <: DeterministicNetwork
    regressor
end

# keyword constructor
WrappedRegressor2(; regressor=RidgeRegressor()) = WrappedRegressor2(regressor)
nothing #hide
```

We now simply cut and paste the code defining the learning network
into a model `fit` method (as opposed to machine `fit!` methods, which
internally dispatch model `fit` methods on the data bound to the
machine):

```@example 7
import MLJBase
function MLJBase.fit(model::WrappedRegressor2, verbosity::Integer, X, y)
    Xs = source(X)
    ys = source(y, kind=:target)

    stand_model = Standardizer()
    stand = machine(stand_model, Xs)
    W = transform(stand, Xs)

    box_model = UnivariateBoxCoxTransformer()
    box = machine(box_model, ys)
    z = transform(box, ys)

    ridge_model = model.regressor        ###
    ridge =machine(ridge_model, W, z)
    zhat = predict(ridge, W)

    yhat = inverse_transform(box, zhat)
    fit!(yhat, verbosity=0)

    return fitresults(yhat)
end
```

The line marked `###`, where the new exported model's hyperparameter
`regressor` is spliced into the network, is the only modification. This
completes the export process.

> **What's going on here?** MLJ's machine interface is built atop a more primitive *[model](simple_user_defined_models.md)* interface, implemented for each algorithm. Each supervised model type (eg, `RidgeRegressor`) requires model `fit` and `predict` methods, which are called by the corresponding *machine* `fit!` and `predict` methods. We don't need to define a  model `predict` method here because MLJ provides a fallback which simply calls the terminating node of the network built in `fit` on the data supplied. The expression `fitresults(yhat)` bundles the terminal node `yhat` with reports (one for each machine in the network) and moves training data out to a bundled cache object. This ensures machines wrapping exported model instances do not contain actual training data in their `fitresult` fields.

```julia
X, y = @load_boston
wrapped_regressor2 = WrappedRegressor2()
evaluate(wrapped_regressor2, X, y, resampling=CV(), measure=rms, verbosity=0)
```

```julia
(measure = MLJBase.RMS[rms],
 measurement = [5.26287],
 per_fold = Array{Float64,1}[[3.01228, 4.73544, 5.01316, 4.21653, 8.9335, 3.45975]],
 per_observation = Missing[missing],)
```

## Static operations on nodes

Continuing to view nodes as "dynamic data", we can, in addition to
applying "dynamic" operations like `predict` and `transform` to nodes,
overload ordinary "static" (unlearned) operations as well. Common
operations, like addition, scalar multiplication, `exp`, `log`,
`vcat`, `hcat`, tabularization (`MLJ.table`) and matrixification
(`MLJ.matrix`) work out-of-the box.

As a demonstration, consider the learning network below that: (i)
One-hot encodes the input table `X`; (ii) Log transforms the
continuous target `y`; (iii) Fits specified K-nearest neighbour and
ridge regressor models to the data; (iv) Computes an average
of the individual model predictions; and (v) Inverse transforms
(exponentiates) the blended predictions.

Note, in particular, the lines defining `zhat` and `yhat`, which
combine several static node operations.

```julia
@load RidgeRegressor pkg=MultivariateStats
@load KNNRegressor

Xs = source()
ys = source(kind=:target)

hot = machine(OneHotEncoder(), Xs)

# W, z, zhat and yhat are nodes in the network:

W = transform(hot, Xs) # one-hot encode the input
z = log(ys)            # transform the target

model1 = RidgeRegressor(lambda=0.1)
model2 = KNNRegressor(K=7)

mach1 = machine(model1, W, z)
mach2 = machine(model2, W, z)

# average the predictions of the KNN and ridge models:
zhat = 0.5*predict(mach1, W) + 0.5*predict(mach2, W)

# inverse the target transformation
yhat = exp(zhat)
```

Exporting this learning network as a stand-alone model:

```julia
julia> @from_network DoubleRegressor1(regressor1=model1, regressor2=model2) <= yhat
DoubleRegressor1(regressor1 = RidgeRegressor(lambda = 0.1,),
                 regressor2 = KNNRegressor(K = 7,
                                           algorithm = :kdtree,
                                           metric = Distances.Euclidean(0.0),
                                           leafsize = 10,
                                           reorder = true,
                                           weights = :uniform,),) @ 1…93
```

To deal with operations on nodes not supported out-of-the box, one
uses the `nodes` method. Supposing, in the preceding example, we
wanted the geometric mean rather than arithmetic mean. Then, the
definition of `zhat` above can be replaced with

```julia
zhat = node((y1, y2)->sqrt.(y1.*y2), predict(mach1, W), predict(mach2, W))
```

Finally, suppose we want a *weighted* average of the two models, with
the weighting controlled by a user-specified parameter `mix` (the
weights being `(1 - mix)` and `mix` respectively). We can either use
the advanced export Method II above to arrange for our exported model
to include `mix` as a hyperparameter (because `@from_network` can
only expose component models as hyperparameters of the composite) or
we can encode the weighting operation in a new custom "static" model
type defined in the following way:

```julia
mutable struct Averager <: Static
    mix::Float64
end

import MLJBase
MLJBase.transform(a::Averager, _, y1, y2) = (1 - a.mix)*y1 + a.mix*y2
```

Such static transformers with (unlearned) parameters can have
arbitrarily many inputs, but only one output. In the single input
case an `inverse_transform` can also be defined.

Now that the static transformer `Averager` is defined, our new definition of
`zhat` and `yhat` become:

```julia
averager_model = Averager(0.5)
y1 = predict(mach1, W)
y2 = predict(mach2, W)
averager = machine(averager_model, y1, y2)
zhat = transform(averager, y1, y1)
yhat = exp(zhat)
```

Exporting to obtain the composite model instance:

```julia
composite = @from_network(DoubleRegressor3(regressor1=model1,
                                           regressor2=model2,
                                           averager=averager_model) <= yhat)

```

Training on some data, using the default regressors and `mix=0.2`:

```julia
julia> composite.averager.mix = 0.2
julia> evaluate(composite, X, y, resampling=Holdout(fraction_train=0.7), measure=rmsl)
```

```julia
Evaluating over 1 folds: 100%[=========================] Time: 0:00:09
(measure = MLJBase.RMSL[rmsl],
 measurement = [0.546889],
 per_fold = Array{Float64,1}[[0.546889]],
 per_observation = Missing[missing],)
```


### More `node` examples

A `node` method allows us to overload a given function to
node arguments.  Here are some examples taken from MLJ source
(at work in the example above):

```julia
Base.log(v::Vector{<:Number}) = log.(v)
Base.log(X::AbstractNode) = node(log, X)

import Base.+
+(y1::AbstractNode, y2::AbstractNode) = node(+, y1, y2)
+(y1, y2::AbstractNode) = node(+, y1, y2)
+(y1::AbstractNode, y2) = node(+, y1, y2)
```

Here `AbstractNode` is the common super-type of `Node` and `Source`.

As a final example, here's how to extend row shuffling to nodes:

```julia
using Random
Random.shuffle(X::AbstractNode) = node(Y -> MLJ.selectrows(Y, Random.shuffle(1:nrows(Y))), X)
X = (x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     x2 = [:one, :two, :three, :four, :five, :six, :seven, :eight, :nine, :ten])
Xs = source(X)
W = shuffle(Xs)
```

```julia
Node @ 9…86 = #4(6…62)
```

```julia
W()
```

```julia
(x1 = [1, 4, 3, 6, 8, 5, 7, 2, 9, 10],
 x2 = Symbol[:one, :four, :three, :six, :eight, :five, :seven, :two, :nine, :ten],)
```


## The learning network API

Three julia types are part of learning networks: `Source`, `Node` and
`NodalMachine`. A `NodalMachine` is returned by the `machine`
constructor when given nodal arguments instead of concrete data.

The definitions of `Node` and `NodalMachine` are coupled because every
`NodalMachine` has `Node` objects in its `args` field (the *training
arguments* specified in the constructor) and every `Node` must specify
a `NodalMachine`, unless it is static (see below).

Formally, a learning network defines *two* labeled directed acyclic
graphs (DAG's) whose nodes are `Node` or `Source` objects, and whose
labels are `NodalMachine` objects. We obtain the first DAG from
directed edges of the form $N1 -> N2$ whenever $N1$ is an *argument*
of $N2$ (see below). Only this DAG is relevant when calling a node, as
discussed in examples above and below. To form the second DAG
(relevant when calling or calling `fit!` on a node) one adds edges for
which $N1$ is *training argument* of the the machine which labels
$N1$. We call the second, larger DAG, the *complete learning network*
below (but note only edges of the smaller network are explicitly drawn
in diagrams, for simplicity).

### Source nodes

Only source nodes reference concrete data. A `Source` object has a
single field, `data`.

```@docs
source(X)
rebind!
sources
origins
```

### Nodal machines

The key components of a `NodalMachine` object are:

- A *model*,  specifying a learning algorithm and hyperparameters.

- Training *arguments*, which specify the nodes acting as proxies for
  training data on calls to `fit!`.

- A *fitresult*, for storing the outcomes of calls to `fit!`.

A nodal machine is trained in the same way as a regular machine with
one difference: Instead of training the model on the wrapped data
*indexed* on `rows`, it is trained on the wrapped nodes *called* on
`rows`, with calling being a recursive operation on nodes within a
learning network (see below).


### Nodes

The key components of a `Node` are:

- An *operation*, which will either be *static* (a fixed function) or
  *dynamic* (such as `predict` or `transform`, dispatched on a nodal
  machine `NodalMachine`).

- A nodal *machine* on which to dispatch the operation (void if the
  operation is static).

- Upstream connections to other nodes (including source nodes)
  specified by *arguments* (one for each argument of the operation).

- A dependency *tape*, listing of all upstream nodes in the complete
  learning network, with an order consistent with the learning network
  as a DAG.

```@docs
node
```

```@docs
fit!(N::Node; rows=nothing, verbosity=1, force=false)
```

```@docs
fit!(mach::MLJ.AbstractMachine; rows=nothing, verbosity=1, force=false)
```

```@docs
@from_network
```
