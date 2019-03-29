# Learning Networks

MLJ has a flexible interface for building networks from multiple
machine learning elements, whose complexity extend beyond the
"pipelines" of other machine learning toolboxes. 


### Overview

In the future the casual MLJ user will be able to build common
pipeline architetures, such as linear compositites and stacks, with
simple macro invocations. Handcrafting a learning network, as outlined
below, is an advanced MLJ feature, assuming familiarity with the
basics outlined in [Getting Started](index.md). The syntax
for building a learning network is essentially an extension of the
basic syntax but with data objects replaced with nodes ("dynamic
data").

In MLJ, a *learning network* is a graph whose nodes apply an
operation, such as `predict` or `transform`, using a fixed machine
(requiring training) - or which, alternatively, applies a regular
(untrained) mathematical operation to its input(s). In practice, a
learning network works with *fixed* sources for its
training/evaluation data, but can be built and tested in stages. By
contrast, an *exported learning network* is a learning network
exported as a stand-alone, re-usable `Model` object, to which all the
MLJ `Model` meta-algorthims can be applied (ensembling, systematic
tuning, etc).

As we shall see, exporting a learning network as a reusable model, is
quite simple. While one can entirely skip the build-and-train steps,
experimenting with raw learning networks may be the best way to
understand how the stand-alone models work.

In MLJ learning networks treat the flow of information during training
and predicting separately. For this reason, simpler examples may
appear more a little more complicated than in other
approaches. However, in more sophisticated examples, such as stacking,
this separation is essential.


### Building a simple learning network

![](wrapped_ridge.png)

The diagram above depicts a learning network which standardises the
input data `X`, learns an optimal Box-Cox transformation for the
target `y`, predicts new target values using ridge regression, and
then inverse-transforms those predictions, for later comparison with
the original test data. The machines are labelled yellow.

To implement the network, we begin by loading data needed for training
and evaluation into *source nodes*. For testing purposes, we use
synthetic data:

```@example 1
using MLJ # hide
using DataFrames, Statistics # hide
Xraw = rand(300,3)
y = exp(Xraw[:,1] - Xraw[:,2] - 2Xraw[:,3] + 0.1*rand(300))
X = DataFrame(Xraw)
ys = source(y)
Xs = source(X)
```

We label nodes we will construct according to their outputs in the
diagram. Notice that the nodes `z` and `yhat` use the same machine,
namely `box`, for different operations.

To construct the `W` node we first need to define the machine `stand`
that it will use to transform inputs.

```@example 1
stand_model = Standardizer()
stand = machine(stand_model, Xs)
```

Because `Xs` is a node, instead of concrete data, we can call
`transform` on the machine without first training it, and the result
is the new node `W`, instead of concrete transformed data:

```@example 1
W = transform(stand, Xs)
```

To get actual transformed data we *call* the node appropriately, which
will require we first train the node. Training a node, rather than a
machine, triggers training of *all* necessary machines in the network.


```@example 1
test, train = partition(eachindex(y), 0.8)
fit!(W, rows=train)
W()           # transform all data
W(rows=test ) # transform only test data
W(X[3:4,:])   # transform any data, new or old
```

If you like, you can think of `W` (and the other nodes we will define)
as "dynamic data": `W` is *data*, in the sense that it an be called
("indexed") on rows, but *dynamic*, in the sense the result depends on
the outcome of training events.

The other nodes of our network are defined similarly:

```@example 1
box_model = UnivariateBoxCoxTransformer()  # for making data look normally-distributed
box = machine(box_model, ys)
z = transform(box, ys)

ridge_model = RidgeRegressor(lambda=0.1)
ridge =machine(ridge_model, W, z)
zhat = predict(ridge, W)

yhat = inverse_transform(box, zhat)
```

We are ready to train and evaluate the completed network. Notice that
the standardizer, `stand`, is *not* retrained, as MLJ remembers that
it was trained earlier:


```@example 1
fit!(yhat, rows=train)
rms(y[test], yhat(rows=test)) # evaluate
```

We can change a hyperparameters and retrain:

```@example 1
ridge_model.lambda = 0.01
fit!(yhat, rows=train) 
rms(y[test], yhat(rows=test))
```

> **Notable feature.** The machine, `ridge::NodalMachine{RidgeRegressor}`, is retrained, because its underlying model has been mutated. However, since the outcome of this training has no effect on the training inputs of the machines `stand` and `box`, these transformers are left untouched. (During construction, each node and machine in a learning network determines and records all machines on which it depends.) This behaviour, which extends to exported learning networks, means we can tune our wrapped regressor without re-computing transformations each time the hyperparameter is changed. 


### Exporting a learning network as a stand-alone model

To export a learning network:
- Define a new `mutable struct` model type.
- Wrap the learning network code in a model `fit` method.

All learning networks that make determinisic (or, probabilistic)
predictions export as models of subtype `Deterministic{Node}`
(respectively, `Probabilistic{Node}`):

```@example 1
mutable struct WrappedRidge <: Deterministic{Node}
    ridge_model
end

WrappedRidge(; ridge_model=RidgeRegressor) = WrappedRidge(ridge_model); # keyword constructor
```

Now satisfied that our wrapped Ridge Regression learning network
works, we simply cut and paste its defining code into a `fit` method:


```@example 1
function MLJ.fit(model::WrappedRidge, X, y)
    Xs = source(X)
    ys = source(y)

    stand_model = Standardizer()
    stand = machine(stand_model, Xs)
    W = transform(stand, Xs)

    box_model = UnivariateBoxCoxTransformer()  # for making data look normally-distributed
    box = machine(box_model, ys)
    z = transform(box, ys)

    ridge_model = model.ridge_model ###
    ridge =machine(ridge_model, W, z)
    zhat = predict(ridge, W)

    yhat = inverse_transform(box, zhat)
    fit!(yhat, verbosity=0)
    
    return yhat
end
```

The line marked `###`, where the new exported model's hyperparameter `ridge_model` is spliced into the network, is the only modification.

> **What's going on here?** MLJ's machine interface is built atop a more primitive *[model](adding_new_models.md)* interface, implemented for each algorithm. Each supervised model type (eg, `RidgeRegressor`) requires model `fit` and `predict` methods, which are called by the corresponding machine `fit!` and `predict` methods. We don't need to define a  model `predict` method here because MLJ provides a fallback which simply calls the node returned by `fit` on the data supplied: `MLJ.predict(model::Supervised{Node}, Xnew) = yhat(Xnew)`.

The export process is complete and we can wrap our exported model
around any data or task we like, and evaluate like any other model:

```@example 1
task = load_boston()
wrapped_model = WrappedRidge(ridge_model=ridge_model)
mach = machine(wrapped_model, task)
evaluate!(mach, resampling=CV(), measure=rms, verbosity=0)
```

Another example of an exported learning network is given in the next
subsection.


### Static operations on nodes

Continuing to view nodes as "dynamic data", we can, in addition to
applying "dynamic" operations like `predict` and `transform` to nodes,
overload ordinary "static" operations as well. Common operations, like
addition, scalar multiplication, `exp` and `log` work out-of-the
box. To demonstrate this, consider the code below defining a composite
model that:

(1) one-hot encodes the input table `X`
(2) log transforms the continuous target `y`
(3) fits specified K-nearest neighbour and ridge regressor models to the data
(4) computes a weighted average of individual model predictions
(5) inverse transforms (exponentiates) the blended predictions

Note, in particular, the lines defining `zhat` and `yhat`, which
combine several static node operations.

```@example 1
mutable struct KNNRidgeBlend <:Deterministic{Node}

    knn_model
    ridge_model
    weights::Tuple{Float64, Float64}

end

function MLJ.fit(model::KNNRidgeBlend, X, y)
    
    Xs = source(X) 
    ys = source(y)

    hot = machine(OneHotEncoder(), Xs)

    # W, z, zhat and yhat are nodes in the network:
    
    W = transform(hot, Xs) # one-hot encode the input
    z = log(ys) # transform the target
    
    ridge_model = model.ridge_model
    knn_model = model.knn_model

    ridge = machine(ridge_model, W, z) 
    knn = machine(knn_model, W, z)

    # average the predictions of the KNN and ridge models
    zhat = model.weights[1]*predict(ridge, W) + weights[2]*predict(knn, W) 

    # inverse the target transformation
    yhat = exp(zhat) 

    fit!(yhat, verbosity=0)
    
    return yhat
end

```

```@example 1
task = load_boston()
knn_model = KNNRegressor(K=2)
ridge_model = RidgeRegressor(lambda=0.1)
weights = (0.9, 0.1)
blended_model = KNNRidgeBlend(knn_model, ridge_model, weights)
mach = machine(blended_model, task)
evaluate!(mach, resampling=Holdout(fraction_train=0.7), measure=rmsl) 
```

To overerload a function for application to nodes, we the `node`
method.  Here are some examples taken from MLJ source (at work in the
example above):

```julia
Base.log(v::Vector{<:Number}) = log.(v)
Base.log(X::AbstractNode) = node(log, X)

import Base.+
+(y1::AbstractNode, y2::AbstractNode) = node(+, y1, y2)
+(y1, y2::AbstractNode) = node(+, y1, y2)
+(y1::AbstractNode, y2) = node(+, y1, y2)
```

Here `AbstractNode` is the common supertype of `Node` and `Source`.

As a final example, here's how to extend row shuffling to nodes:

```julia
using Random
Random.shuffle(X::AbstractNode) = node(Y -> MLJ.selectrows(Y, Random.shuffle(1:nrows(Y))), X)
```

```@example 1
using Random # hide 
Random.shuffle(X::AbstractNode) = node(Y -> MLJ.selectrows(Y, Random.shuffle(1:nrows(Y))), X) # hide
X = (x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     x2 = [:one, :two, :three, :four, :five, :six, :seven, :eight, :nine, :ten])
Xs = source(X)
W = shuffle(Xs)
```
```@example 1
W()
```
    
    
### The learning network API

Three types are part of learning networks: `Source`, `Node` and
`NodalMachine`. A `NodalMachine` is returned by the `machine`
constructor when given nodal arguments instead of concrete data.

The definitions of `Node` and `NodalMachine` are coupled because every
`NodalMachine` has `Node` objects in its `args` field (the *training
arguments* specified in the constructor) and every `Node` must specify
a `NodalMachine`, unless it is static (see below).


### Source nodes

Only source nodes reference concrete data. A `Source` object has a
single field, `data`. 

```@docs
source(X)
sources
```

### Nodal machines

The key components of a `NodalMachine` object are:

- A *model*,  specifying a learning algorithm and hyperparameters.

- Training *arguments*, which specify the nodes acting as proxies for
  training data on calls to `fit!`.

- A *fit-result*, for storing the outcomes of calls to `fit!`.

- A dependency *tape* (a vector or DAG) containing elements of type
  `NodalMachine`, obtained by merging the tapes of all training
  arguments.

    
### Nodes

The key components of a `Node` are:

- An *operation*, which will either be *static* (a fixed function) or
  *dynamic* (such as `predict` or `transform`, dispatched on a nodal
  machine `NodalMachine`).

- A nodal *machine* on which to dispatch the operation (void if the
  operation is static).

- Upstream connections to other nodes (including source nodes)
  specified by *arguments* (one for each argument of the operation).
   
- A dependency *tape*, obtained by merging the the tapes of all
  arguments (nodal machines) and adding the present node's nodal
  machine.

```@docs
node
```

