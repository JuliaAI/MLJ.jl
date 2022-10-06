# Learning Networks

Below is a practical guide to the MLJ implementation of learning
networks, which have been described more abstractly in the article:

[Anthony D. Blaom and Sebastian J. Voller (2020): Flexible model
composition in machine learning and its implementation in MLJ.
Preprint, arXiv:2012.15505](https://arxiv.org/abs/2012.15505)

*Learning networks*, an advanced but powerful MLJ feature, are "blueprints" for combining
models in flexible ways, beyond ordinary linear pipelines and simple model ensembles. They
are simple transformations of your existing workflows which can be "exported" to define
new, re-usable composite model types (models which typically have other models as
hyperparameters).

Pipeline models (see [`Pipeline`](@ref)), and model stacks (see [`Stack`](@ref)) are both
implemented internally as exported learning networks.

!!! note

   While learning networks can be used for complex machine learning workflows, their main
   purpose is for defining new standalone composite model types, which behave just like
   any other model type: they can be evaluated, tuned, inserted into pipelines, etc.  In
   serious applications, users are encouraged to export their learning networks, as
   explained under [Exporting a learning network as a new composite model type](@ref)
   below, **after testing the network**, using a small training dataset.


## Learning networks by example

Learning networks are best explained by way of example.

### Lazy computation

The core idea of a learning network is delayed or *lazy* computation. Instead of

```@setup 42
using MLJ
MLJ.color_off()
```

```julia
X = 4
Y = 3
Z = 2*X
W = Y + Z
```

we can do

```@example 42
using MLJ

X = source(4)
Y = source(9)
Z = 2*X
W = Y + Z
W()
```

In the first computation `X`, `Y`, `Z` and `W` are all bound to ordinary data. In the
second, they are bound to objects called *nodes*. The nodes constituting entry points for
data, namely `X` and `Y`, are called *source nodes*. As the terminology suggests, we can
imagine these objects as part of a "network" (a directed acyclic graph) which can aid
conceptualization (but is less useful in more complicated examples):

![](assets/simple.png)

### The origin of a node

The source nodes on which a given node depends are called the
*origins* of the node:

```@example 42
os = origins(W)
X in os
```


### Re-using a network

The advantage of lazy evaluation is that we can change data at a source node to repeat the
calculation with new data. One way to do this (strongly discouraged in practice) is to use
`rebind!`:

```@example 42
rebind!(X, 1) # demonstration only!
W()
```

However, if a node has a unique origin, *one simply calls the node on the new data one
would like to rebind to that origin*:

```@example 42
origins(Z)
```

```@example 42
Z(6)
```

This has the advantage that you don't need to locate the origin and
rebind data directly, and the unique-origin restriction turns out to
be sufficient for the applications to learning we have in mind.

### Overloading functions for use on nodes

Several built-in function like `*` and `+` above are overloaded in
MLJBase to work on nodes, as illustrated above. Others that work
out-of-the-box include: `MLJBase.matrix`, `MLJBase.table`, `vcat`,
`hcat`, `mean`, `median`, `mode`, `first`, `last`, as well as
broadcasted versions of `log`, `exp`, `mean`, `mode` and `median`. A
function like `sqrt` is not overloaded, so that `Q = sqrt(Z)` will
throw an error. Instead, we do

```@example 42
Q = node(z->sqrt(z), Z)
Z()
```

```@example 42
Q()
```

### A network that learns

To incorporate learning in a network of nodes we need to:

- Let machines grab data from *nodes* instead of ordinary data containers

- Generate "operation" nodes when calling an operation like `predict` or `transform` on a
machine. Such nodes point to both a machine (storing learned parameters) and nodes from
which to fetch data for applying the operation (which, unlike the nodes seen so far,
depend on learned parameters to generate output).

For an example of a learning network that actually learns, we first synthesize some
training and production data:

```@example 42
using MLJ
X, y = make_blobs(cluster_std=10.0, rng=123)  # `X` is a table, `y` a vector
Xnew, _ = make_blobs(3) # `Xnew` is a table with the same number of columns
nothing # hide
```

We choose a model do some dimension reduction, and another to perform classification:

```@example 42
pca = (@load PCA pkg=MultivariateStats verbosity=0)()
tree = (@load DecisionTreeClassifier pkg=DecisionTree verbosity=0)()
```

To make our learning lazy, we wrap training data as source nodes:

```@example 42
Xs = source(X)
ys = source(y)
```

And, finally, proceed as we would in an ordinary MLJ workflow, with the exception that
there is no need to `fit!` our machines, as training will be carried out lazily later:

```@example 42
mach1 = machine(pca, Xs)
x = transform(mach1, Xs) # defines a new node because `Xs` is a node

mach2 = machine(tree, x, ys)
yhat = predict(mach2, x) # defines a new node because `x` is a node
```

Note that `mach1` and `mach2` are not themselves nodes. They *point* to the nodes they
need to *call* to get training data and they are in turn *pointed to* by other nodes. In
fact, an interesting implementation detail is that an "ordinary" machine is not actually
bound to directly to data at all but bound to data wrapped in source nodes, which
explains what you may have noticed about the way machines are displayed:

```@example 42
machine(pca, Xnew) # `Xnew` is ordinary data
```

Before calling a node, we need to `fit!` the node, to trigger training of all the machines
on which it depends:

```@example 42
fit!(yhat)   # can include same keyword options for `fit!(::Machine, ...)`
yhat()[1:2]  # or `yhat(rows=2)`
```

This represents the prediction on the *training* data, because that's what resides at our
source nodes. However, `yhat` has the unique origin `X` (because "training edges" in the
complete associated directed graph are excluded for this purpose). We can therefore call
`yhat` on our production data to get the corresponding predictions:

```@example 42
yhat(Xnew)
```

Training is smart, in the sense that mutating a hyper-parameter of some component model
does not force retraining of upstream machines:

```@example 42
tree.max_depth = 1
fit!(yhat)
yhat(Xnew)
```

### Multithreaded training

A more complicated learning network may contain machines that can be trained in
parallel. In that case, a call like the following could accelerate training:

```@example 42
tree.max_depth=2
fit!(yhat, acceleration=CPUThreads())
nothing # hide
```

## Exporting a learning network as a new composite model type

Once a learning network has been tested, typically on some small dummy data set, it is
ready to be exported as a new, standalone, re-usable model type (unattached to any
data). We demonstrate the process by way of examples.

### Example A - Mini-pipeline

First we export the simple learning network defined above.

#### Step 1 - Define a new model struct

We need a type with two fields, one for the preprocessor (`pca` in the network above)
and one for the classifier (`tree` in the network above).

The `DecisionTreeClassifier` type of `tree` has supertype `Probabilistic`, because it
makes probabilistic predictions, and we assume any classifier we want to swap out will be
the same.

```@example 42
supertype(typeof(tree))
```

In particular, our composite model will also need `Probabilistic` as supertype. In fact,
we give it the intermediate supertype `ProbabilisticNetworkComposite <: Probabilistic`, so
that we additionally flag it an exported learning network model type:

```@example 42
mutable struct CompositeA <: ProbabilisticNetworkComposite
        preprocessor
        classifier
end
```

The common alternatives are `DeterministicNetworkComposite` and
`UnsupervisedNetworkComposite`. But all options can be viewed as follows:

```@example 42
using MLJBase
NetworkComposite
```

We next make our learning network model-generic by *substituting each model instance
with the corresponding **symbol** representing a property (field) of the new model
struct*:

```@example 42
mach1 = machine(:preprocessor, Xs)   # <---- `pca` swapped out for `:preprocessor`
x = transform(mach1, Xs)
mach2 = machine(:classifier, x, ys)  # <---- `tree` swapped out for `:classifier`
yhat = predict(mach2, x)
```

This network can be used as before except we must provide an instance of `CompositeA` in
our `fit!` calls:

```@example 42
composite_a = CompositeA(pca, ConstantClassifier())
fit!(yhat, composite=composite_a)
yhat(Xnew)
```

Notice how we chose a new classifier here.

#### Step 2 - Wrap the learning network in `prefit`

Literally copy and paste the learning network above into the definition of a method called
`prefit`, as shown below:

```@example 42
import MLJBase
function MLJBase.prefit(composite::CompositeA, verbosity, X, y)

        # the learning network from above:
        Xs = source(X)
        ys = source(y)
        mach1 = machine(:preprocessor, Xs)
        x = transform(mach1, Xs)
        mach2 = machine(:classifier, x, ys)
        yhat = predict(mach2, x)

        verbosity > 0 && @info "I'm a noisy fellow!"

        # return "learning network interface":
        return (; predict=yhat)
end
```

That's it.

Generally, `prefit` always returns a *learning network interface*; see
[`MLJBase.prefit`](@ref) for what this means. In this example, the interface dictates that
calling `predict(mach, Xnew)` on a machine `mach` bound to some instance of `CompositeA`
should internally call `yhat(Xnew)`.

Here's our new composite model type `CompositeA` in action, combining standardization with
KNN classification:

```@example 42
X, y = @load_iris

knn = (@load KNNClassifier pkg=NearestNeighborModels)()
composite = CompositeA(Standardizer(), knn)
```

```@example 42
mach = machine(composite, X, y) |> fit!
predict(mach, X)[1:2]
```

```@example 42
report(mach).preprocessor
```

```@example 42
fitted_params(mach).classifier
```

### Example B - Multiple operations: transform and inverse transform

Here's a second mini-pipeline example composing two transformers which both implement
inverse transform. We show how to implement an `inverse_transform` for the composite model
too.

#### Step 1 - Define a new model struct

```@example 42
using MLJBase

mutable struct CompositeB <: DeterministicNetworkComposite
    transformer1
    transformer2
end
```

#### Step 2 - Wrap the learning network in `prefit`

```@example 42
function MLJBase.prefit(composite::CompositeB, verbosity, X)
    Xs = source(X)

    mach1 = machine(:transformer1, Xs)
    X1 = transform(mach1, Xs)
    mach2 = machine(:transformer2, X1)
    X2 = transform(mach2, X1)

    W1 = inverse_transform(mach2, Xs)
    W2 = inverse_transform(mach1, W1)

    # return "learning network interface":
    return (; transform=X2, inverse_transform=W2)
end
```

Here's a demonstration:

```julia
X = rand(100)

composite_b = CompositeB(UnivariateBoxCoxTransformer(), Standardizer())
mach = machine(composite_b, X) |> fit!
W =  transform(mach, X)
@assert inverse_transform(mach, W) ≈ X
```

### Example C - Exposing internal network state in reports

The code below defines a new composite model type `CompositeC` that predicts by taking the
weighted average of two regressors, and additionally exposes, in the model's report, a
measure of disagreement between the two models at time of training. In addition to the two
regressors, the new model has two other fields:

- `mix`, controlling the weighting

- `acceleration`, for the model of acceleration for training the model (e.g.,
  `CPUThreads()`).

#### Step 1 - Define a new model struct

```@example 42
using MLJBase

mutable struct CompositeC <: DeterministicNetworkComposite
    regressor1
    regressor2
    mix::Float64
    acceleration
end
```

#### Step 2 - Wrap the learning network in `prefit`

```@example 42
function MLJBase.prefit(composite::CompositeC, verbosity, X, y)

    Xs = source(X)
    ys = source(y)

    mach1 = machine(:regressor1, Xs, ys)   #  <--- symbol instead of model
    mach2 = machine(:regressor2, Xs, ys)

    yhat1 = predict(mach1, Xs)
    yhat2 = predict(mach2, Xs)

    # node to return agreement between the regressor predictions:
    disagreement = node((y1, y2) -> l2(y1, y2) |> mean, yhat1, yhat2)

    # get the weighted average the predictions of the regressors:
    λ = composite.mix
    yhat = (1 - λ)*yhat1 + λ*yhat2

    return (
        predict = yhat,
        report= (; training_disagreement=disagreement),
                acceleration = composite.acceleration,
    )

end
```

Here's a demonstration:

```@example 42
X, y = make_regression() # a table and a vector

knn = (@load KNNRegressor pkg=NearestNeighborModels)()
tree =  (@load DecisionTreeRegressor pkg=DecisionTree)()
composite_c = CompositeC(knn, tree, 0.2, CPUThreads())
mach = machine(composite_c, X, y) |> fit!
Xnew, _ = make_regression(3)
predict(mach, Xnew)
```

```@example 42
report(mach)
```

### Example D - Multiple nodes pointing to the same machine

When incorporating learned target tranformations (such as a standardization) in supervised
learning, it is desirable to apply the *inverse* transformation to predictions, to return
them to the original scale. This means re-using learned parameters from an earlier part of
your workflow. This poses no problem here, as the next example demonstrates.

The model type `CompositeD` defined below applies applies a preprocessing transformation
to input data `X` (e.g., standardization), learns a transformation for the target `y`
(e.g., an optimal Box-Cox transformation), predicts new target values using a regressor
(e.g., Ridge regression), and then inverse-transforms those predictions to restore them to
the original scale. (This represents a model we could alternatively build using the
[`TransformedTargetModel`](@ref) wrapper and a [`Pipeline`](@ref).)

#### Step 1 - Define a new model struct

```@example 42
using MLJBase

mutable struct CompositeD <: DeterministicNetworkComposite
    preprocessor
    target_transformer
    regressor
    acceleration
end
```

#### Step 2 - Wrap the learning network in `prefit`

Notice that both of the nodes `z` and `yhat` in the wrapped learning network point to the
same machine (learned parameters) `mach2`.

```@example 42
function MLJBase.prefit(composite::CompositeD, verbosity, X, y)

    Xs = source(X)
    ys = source(y)

    mach1 = machine(:preprocessor, Xs)
    W = transform(mach1, Xs)

    mach2 = machine(:target_transformer, ys)
    z = transform(mach2, ys)

    mach3 =machine(:regressor, W, z)
    zhat = predict(mach3, W)

    yhat = inverse_transform(mach2, zhat)

    return (
        predict = yhat,
        acceration = composite.acceleration,
    )

end
```

The flow of information in the wrapped learning network is visualized below.

![](img/target_transform2.png)

Here's an application of our new composite to the Boston dataset:

```@example 42
X, y = @load_boston

stand = Standardizer()
box = UnivariateBoxCoxTransformer()
ridge = (@load RidgeRegressor pkg=MultivariateStats)(lambda=92)
composite_d = CompositeD(stand, box, ridge, CPU1())
evaluate(composite_d, X, y, resampling=CV(nfolds=5), measure=l2, verbosity=0)
```


### Example E - Coupling component model hyper-parameters

The composite model in this example combines a clustering model used to reduce the
dimension of the feature space (`KMeans` or `KMedoids` from Clustering.jl) with ridge
regression, but has the following "coupling" of the hyperparameters: The amount of ridge
regularization depends on the number of specified clusters `k`, with less regularization
for a greater number of clusters. It includes a user-specified coupling coefficient `c`,
and exposes the `solver` hyper-parameter of the ridge regressor. (The ridge regressor
itself is not a hyperparameter of the composite.)

#### Step 1 - Define a new model struct

```@example 42
using MLJBase

mutable struct CompositeE <: DeterministicNetworkComposite
        clusterer     # `:kmeans` or `:kmedoids`
        k::Int        # number of clusters
        solver        # a ridge regression parameter we want to expose
        c::Float64    # a "coupling" coefficient
end
```

#### Step 2 - Wrap the learning network in `prefit`

```@example 42
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels
KMeans   = @load KMeans pkg=Clustering
KMedoids = @load KMedoids pkg=Clustering

function MLJBase.prefit(composite::CompositeE, verbosity, X, y)

        Xs = source(X)
        ys = source(y)

        k = composite.k
        solver = composite.solver
        c = composite.c

        clusterer = composite.clusterer == :kmeans ? KMeans(; k) : KMedoids(; k)
        mach1 = machine(clusterer, Xs)
        Xsmall = transform(mach1, Xs)

        # the coupling - ridge regularization depends on the number of
        # clusters `k` and the coupling coefficient `c`:
        lambda = exp(-c/k)

        ridge = RidgeRegressor(; lambda, solver)
        mach2 = machine(ridge, Xsmall, ys)
        yhat = predict(mach2, Xsmall)

        return (predict=yhat,)
end
```

Here's an application to the Boston dataset in which we optimize the coupling coefficient:

```@example 42
X, y = @load_boston # a table and a vector

composite_e = CompositeE(:kmeans, 3, nothing, 0.5)
r = range(composite_e, :c, lower = -2, upper=2, scale=x->10^x)
tuned_composite_e = TunedModel(
    composite_e,
    range=r,
    tuning=RandomSearch(rng=123),
    measure=l2,
    resampling=CV(nfolds=6),
    n=100,
)
mach = machine(tuned_composite_e, X, y) |> fit!
report(mach).best_model
```

