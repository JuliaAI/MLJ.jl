# Glossary

Note: This glossary includes some detail intended mainly for MLJ developers.

## Basics

### task (object of type `Task`)

Data plus a learning objective (e.g., "probabilistic prediction of
Sales"). In MLJ a task does not include a description of how the
completed task is to be evaluated.

### hyperparameters

Parameters on which some learning algorithm depends, specified before
the algorithm is applied, and where learning is interpreted in the
broadest sense. For example, PCA feature reduction is a
"preprocessing" transformation "learning" a projection from training
data, governed by a dimension hyperparameter. Hyperparameters in our
sense may specify configuration (eg, number of parallel processes)
even when this does not effect the end-product of learning. (But we
exclude verbosity level.)

### model (object of abstract type `Model`)

Object collecting together hyperameters of a single algorithm. Most
models are classified either as *supervised* or *unsupervised* models
(generally, "transformers").


### fit-result (type generally defined outside of MLJ)

Also known as "learned" or "fitted" parameters, these are "weights",
"coefficients", or similar paramaters learned by an algorithm, after
adopting the prescribed hyperparameters. For example, decision trees
of a random forest, the coefficients and intercept of a linear model,
or the rotation and projection matrices of PCA reduction scheme.


### operation

Data-manipulating operations (methods) parameterized by some
fit-result. For supervised learners, the `predict`, `predict_mean`,
`predict_median`, or `predict_mode` methods; for transformers, the
`transform` or `inverse_transform` method. An operation may also
refert to an ordinary data-manipulating method that does *not* depend
on a fit-result (e.g., a broadcasted logarithm) which is then called
*static* operation for clarity. An operation that is not static is
*dynamic*.

### machine (object of type `Machine`)

An object consisting of:

(1) A model

(2) A fit-result (undefined until training)

(3) *Training arguments* (one for each data argument of the model's
associated `fit` method). A training argument is data used for
training. Generally, there are two training arguments for supervised
models, and just one for unsuperivsed models.

In addition machines store "report" metadata, for recording
algorithm-specific statistics of training (eg, internal estimate of
generalization error, feature importances); and they cache information
allowing the fit-result to be updated without repeating unnecessary
information.

Machines are trained by calls to a `fit` method which may be
passed an optional argument specifying the rows of data to be used in
training.

## Learning Networks and Composite Models

*Note:* Multiple nodal machines may share the same model, and
multiple learning nodes may share the same nodal machine.

### source node (object of type `Source`)

A container for training data and point of entry for new data in a
learning network (see below).

### nodal machine (object of type `NodalMachine`)

Like a machine with the following exceptions:

(1) Training arguments are source nodes or regular nodes (see below)
in the learning network, instead of data.

(2) The object internally records dependencies on other other nodal
machines, as implied by the training arguments, and so on.


###  node (object of type `Node`)

Essentially a nodal machine wrapped in an associated operation
(e.g., `predict` or `inverse_transform`). It detail, it consists of:

1. An operation, static or dynamic.
1. A nodal machine, void if the operation is static.
1. Upstream connections to other learning or source nodes, specified by a list of *arguments* (one for each argument of the operation).
1. Metadata recording the dependencies of the object's machine, and the dependecies on other nodal machines implied by its arguments, and the training arguments of its nodel machine.


### learning network

An acyclic directed graph implicit in the connections of a collection
of source(s) and nodes. Each connected component is ordinarily
restricted to have a unique source.


### wrapper

Any model with one or more other models as hyperparameters.


### composite model

Any wrapper, or any learning network, "exported" as a model (see
[Composing Models](composing_models.md)).
