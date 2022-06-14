# Homogeneous Ensembles

Although an ensemble of models sharing a common set of hyperparameters
can defined using the learning network API, MLJ's `EnsembleModel`
model wrapper is preferred, for convenience and best
performance. Examples of using `EnsembleModel` are given in [this Data
Science
Tutorial](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/ensembles/).

When bagging decision trees, further randomness is normally introduced
by subsampling *features*, when training each node of each tree ([Ho
(1995)](https://web.archive.org/web/20160417030218/http://ect.bell-labs.com/who/tkh/publications/papers/odt.pdf),
[Brieman and Cutler
(2001)](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)). An
bagged ensemble of such trees is known as a [Random
Forest](https://en.wikipedia.org/wiki/Random_forest). You can see an
example of using `EnsembleModel` to build a random forest in [this
Data Science
Tutorial](https://juliaai.github.io/DataScienceTutorials.jl/getting-started/ensembles-2/). However,
you may also want to use a canned random forest model. Run
`models("RandomForest")` to list such models.

```@docs
MLJEnsembles.EnsembleModel
```


