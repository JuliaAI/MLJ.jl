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
the original test data. The machines are labelled yellow. We first
need to import the RidgeRegressor model (you will need MLJModels in
your load path):

```julia
@load RidgeRegressor
```

To implement the network, we begin by loading data needed for training
and evaluation into *source nodes*. For testing purposes, we'll use a
small synthetic data set:

```julia
using Statistics, DataFrames
x1 = rand(300)
x2 = rand(300)
x3 = rand(300)
y = exp.(x1 - x2 -2x3 + 0.1*rand(300))
X = DataFrame(x1=x1, x2=x2, x3=x3) # a column table
ys = source(y)
Xs = source(X)
```

```julia
Source @ 3…40
```

We label nodes we will construct according to their outputs in the
diagram. Notice that the nodes `z` and `yhat` use the same machine,
namely `box`, for different operations.

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
test, train = partition(eachindex(y), 0.8)
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

> **Notable feature.** The machine, `ridge::NodalMachine{RidgeRegressor}`, is retrained, because its underlying model has been mutated. However, since the outcome of this training has no effect on the training inputs of the machines `stand` and `box`, these transformers are left untouched. (During construction, each node and machine in a learning network determines and records all machines on which it depends.) This behaviour, which extends to exported learning networks, means we can tune our wrapped regressor (using a holdout set) without re-computing transformations each time the hyperparameter is changed. 


### Exporting a learning network as a stand-alone model

To export a learning network:

- Define a new `mutable struct` model type.

- Wrap the learning network code in a model `fit` method.

All learning networks that make determinisic (respectively, probabilistic)
predictions export as models of subtype `DeterministicNetwork`
(respectively, `ProbabilisticNetwork`):

```julia
mutable struct WrappedRidge <: DeterministicNetwork
    ridge_model
end

# keyword constructor
WrappedRidge(; ridge_model=RidgeRegressor) = WrappedRidge(ridge_model); 
```

Now satisfied that the learning network we defined above works, we
simply cut and paste its defining code into a `fit` method:


```julia
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

> **What's going on here?** MLJ's machine interface is built atop a more primitive *[model](simple_user_defined_models.md)* interface, implemented for each algorithm. Each supervised model type (eg, `RidgeRegressor`) requires model `fit` and `predict` methods, which are called by the corresponding machine `fit!` and `predict` methods. We don't need to define a  model `predict` method here because MLJ provides a fallback which simply calls the node returned by `fit` on the data supplied: `MLJ.predict(model::SupervisedNetwork, Xnew) = yhat(Xnew)`.

The export process is complete and we can wrap our exported model
around any data or task we like, and evaluate like any other model:

```julia
task = load_boston()
wrapped_model = WrappedRidge(ridge_model=ridge_model)
mach = machine(wrapped_model, task)
evaluate!(mach, resampling=CV(), measure=rms, verbosity=0)
```

```julia
6-element Array{Float64,1}:
 3.0225867093289347
 4.755707358891049 
 5.011312664189936 
 4.226827668908119 
 8.93385968738185  
 3.4788524973220545
```
    
Another example of an exported learning network is given in the next
subsection.


### Static operations on nodes

Continuing to view nodes as "dynamic data", we can, in addition to
applying "dynamic" operations like `predict` and `transform` to nodes,
overload ordinary "static" operations as well. Common operations, like
addition, scalar multiplication, `exp` and `log` work out-of-the
box. To demonstrate this, consider the code below defining a composite
model that: (i) One-hot encodes the input table `X`; (ii) Log
transforms the continuous target `y`; (iii) Fits specified K-nearest
neighbour and ridge regressor models to the data; (iv) Computes a
weighted average of individual model predictions; and (v) Inverse
transforms (exponentiates) the blended predictions.

Note, in particular, the lines defining `zhat` and `yhat`, which
combine several static node operations.

```julia
mutable struct KNNRidgeBlend <:DeterministicNetwork

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

```julia
task = load_boston()
knn_model = KNNRegressor(K=2)
ridge_model = RidgeRegressor(lambda=0.1)
weights = (0.9, 0.1)
blended_model = KNNRidgeBlend(knn_model, ridge_model, weights)
mach = machine(blended_model, task)
evaluate!(mach, resampling=Holdout(fraction_train=0.7), measure=rmsl) 
```

```julia
┌ Info: Evaluating using a holdout set. 
│ fraction_train=0.7 
│ shuffle=false 
│ measure=MLJ.rmsl 
│ operation=StatsBase.predict 
└ Resampling from all rows. 
0.5277143032101871
```

A `node` method allows us to overerload a given function to
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

Here `AbstractNode` is the common supertype of `Node` and `Source`.

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
   
- A dependency *tape*, obtained by merging the the tapes of all
  arguments (nodal machines) and adding the present node's nodal
  machine.

```@docs
node
```

```@docs
fit!(N::Node; rows=nothing, verbosity=1, force=false)
```

```@docs
fit!(mach::MLJ.AbstractMachine; rows=nothing, verbosity=1, force=false)
```



