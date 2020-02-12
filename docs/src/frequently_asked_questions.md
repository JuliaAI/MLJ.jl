# Frequently Asked Questions

## Julia already has a great machine learning toolbox, ScitkitLearn.jl. Why MLJ?

An alternative machine learning toolbox for Julia users is
[ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl). Initially
intended as a Julia wrapper for the popular python library
[scikit-learn](https://scikit-learn.org/stable/), ML algorithms
written in Julia can also implement the ScikitLearn.jl
API. Meta-algorithms (systematic tuning, pipelining, etc) remain
python wrapped code, however.

While ScitkiLearn.jl provides the Julia user with access to a mature
and large library of machine learning models, the scikit-learn API on
which it is modeled, dating back to 2007, is not likely to
evolve significantly in the future. MLJ enjoys (or will enjoy) several
features that should make it an attractive alternative in the longer
term:

- **One language.** ScikitLearn.jl wraps python code, which in turn
  wraps C code for performance-critical routines. A Julia machine
  learning algorithm that implements the MLJ model interface is 100%
  Julia. Writing code in Julia is almost as fast as python and
  well-written Julia code runs almost as fast as C. Additionally, a
  single language design provides superior interoperability. For
  example, one can implement: (i) gradient-descent tuning of
  hyperparameters, using automatic differentiation libraries such as
  Flux.jl; and (ii) GPU performance boosts without major code
  refactoring, using CuArrays.jl.

- **Registry for model metadata.** In ScikitLearn.jl the list of
  available models, as well as model metadata (whether a model handles
  categorical inputs, whether is can make probabilistic predictions,
  etc) must be gleaned from documentation. In MLJ, this information is
  more structured and is accessible to MLJ via a searchable model
  registry (without the models needing to be loaded).

- **Flexible API for model composition.** Pipelines in scikit-learn are
  more of an afterthought than an integral part of the original
  design. By contrast, MLJ's user-interaction API was predicated on the
  requirements of a flexible "learning network" API, one that allows
  models to be connected in essentially arbitrary ways (including
  target transforming and inverse-transforming). Networks can be built
  and tested in stages before being exported as first-class
  stand-alone models. Networks feature "smart" training (only
  necessary components are retrained after parameter changes) and will
  eventually be trainable using a DAG scheduler. With the help of
  Julia's meta-programming features, constructing common
  architectures, such as linear pipelines and stacks, will be one-line
  operations.

- **Clean probabilistic API.** The scikit-learn API does not specify a
  universal standard for the form of probabilistic predictions. By
  fixing a probabilistic API along the lines of the
  [skpro](https://github.com/alan-turing-institute/skpro) project, MLJ
  aims to improve support for Bayesian statistics and probabilistic
  graphical models.

- **Universal adoption of categorical data types.** Python's
  scientific array library NumPy has no dedicated data type for
  representing categorical data (i.e., no type that tracks the pool of
  *all* possible classes). Generally scikit-learn models deal with
  this by requiring data to be relabeled as integers. However, the
  naive user trains a model on relabeled categorical data only to
  discover that evaluation on a test set crashes his code because a
  categorical feature takes on a value not observed in training. MLJ
  mitigates such issues by insisting on the use of categorical data
  types, and by insisting that MLJ model implementations preserve the
  class pools. If, for example, a training target contains classes in
  the pool that do not actually appear in the training set, a
  probabilistic prediction will nevertheless predict a distribution
  whose support includes the missing class, but which is appropriately
  weighted with probability zero.

Finally, we note that a large number of ScikitLearn.jl models are now
wrapped for use in MLJ.
