# Glossary

## Basics

### task (object of type `Task`)

Data plus a clearly specified learning objective. 

> In addition, a description of how the completed task is to be
> evaluated?


### hyperparameters

Parameters on which some learning algorithm depends, specified before
the algorithm is applied, and where learning is interpreted in the
broadest sense. For example, PCA feature reduction is a
"preprocessing" transformation "learning" a projection from training
data, governed by a dimension hyperparameter. Hyperparameters in our
sense may specify configuration (eg, number of parallel processes)
even when this does not effect the end-product of learning. (But we exlcude verbosity level.)

### model (object of abstract type `Model`)

Object collecting together hyperameters of a single algorithm. 


### learner (object of abstract type `Learner`)

Informally, any learning algorithm. More technically, a model
associated with such an algorithm.


### transformer (object of abstract type `Transformer`)

Informally, anything that transforms data or an algorithm that
"learns" such transforms from training data (eg, feature reduction,
normalization). Or, more technically, the model associated with
such an algorithm.


### fit-result (type generally defined outside of MLJ)

The "weights" or "paramaters" learned by an algorithm using the
hyperparameters prescribed in an associated model (eg, what a learner
needs to predict or what a transformer needs to transform). 


### method

What Julia calls a function. (In Julia, a "function" is a collection
of methods sharing the same name but different type signatures.)

Associated with every model is a `fit` method for computing assoicated
fit-results (training), and an `update!` method for retraining with
new hyperaparameters (but unchanged data).


### operation

Data-manipulating operations (methods) parameterized by some
fit-result. For learners, the `predict` or `predict_proba` methods, for
transformers, the `transform` or `inverse_transform` method. In some
contexts such an operation might be replaced by an ordinary operation
(method) that does *not* depend on an fit-result, which are then then
called *static* operations for clarity. An operation that is not static
is *dynamic*.

## Learning Networks and Composite Models

*Note:* Multiple trainable models may share the same model, and
multiple learning nodes may share the same trainable model.

### source node

A mutable container for training data, for use as the mimimal node in a
learning network (see below).


### trainable model

An object consisting of:

(1) A model 

(2) A fit-result (undefined until training)

(3) *Training arguments* (one for each data argument of the model's
associated `fit` method). A training argument is either a source node (see
above) or a *learning node*, as defined below.

(4) A cache object (undefined until training), for storing information
that allows the model to be retrained without repeating unnecessary
computations.

(5) "Report" metadata, recording algorithm-specific statistics of
training (eg, internal estimate of generalization error) or the
results of calls to access model-specific functionality.

(6) "Dependency" metadata, for recording dependencies on other other
trainable models that is implied by the training arguments (when they
are not source nodes).


### learning node

Essentially a trainable model wrapped in an assoicated operation
(e.g., `predict` or `inverse_transform`. It detail, it consists of:

(1) An operation, static or dynamic.

(2) A trainable model, void if the operation is static.

(3) Upstream connections to other learning or source nodes, specified by a list
   of *arguments* (one for each argument of the operation).
   
(4) Metadata recording the dependencies of the object's trainable
model, and the dependecies on other trainable models implied by its
arguments.


### learning network (implicity defined by dynamic data)

A directed graph implicit in the specification of a learning node. 

### composite model

A learning network codified as a model with attendent methods (`fit`,
`update!` and, e.g, `predict`).

