# Glossary

### task (object of type `Task`)

Data plus a clearly specified learning objective. In addition, a
description of how the completed task is to be evaluated.


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


### operator

Data-manipulating operations (methods) parameterized by some
fit-result. For learners, the `predict` or `predict_proba` methods, for
transformers, the `transform` or `inverse_transform` method. In some
contexts such an operator might be replaced by an ordinary operator
(method) that does *not* depend on an fit-result, which are then then
called *static* operators for clarity. An operator that is not static
is *dynamic*.


### trainable model

An object consisting of:

(1) A model 

(2) A fit-result (undefined until training)

(3) Sources for training data, called *training arguments*. A training
argument is either concrete data (eg, data frame) or *dynamic data*,
as defined below.

(4) A cache object (undefined until training) for storing information
required to restart an iterative model, for retraining with new model
hyperparameters without repeating redundant computations, or for
accesssing model-specific functionality (such as pruning a decision
tree).

(5) "Report" metadata, recording algorithm-specific statistics of
training (eg, internal estimate of generalization error) or the
results of calls to access model-specific functionality.

(6) "Dependency" metadata, for recording dependencies on other
trainable models implied by the training arguments (when they are
dynamic data; see below).


### dynamic data

A "trainable" data-like object consisting of:

(1) An operator, static or dynamic.

(2) A trainable model, void if the operator is static.

(3) Connections to other dynamic or static data, specified by a list
   of **arguments** (one for each argument of the operator); each
   argument is data, dynamic or static.

(4) Metadata recording the dependencies of the object's trainable
models, and the dependecies on other trainable models implied by its
arguments.

The "data-like" behaviour of dynamic data is implemented by
overloading Julia's indexing methods and is sketched [here](dynamic_data.md).


### learning network (implicity defined by dynamic data)

A directed graph implicit in the specification of dynamic data. All
nodes are dynamic data except for the source nodes, which are
static. Something like a scikit-learn pipeline.

