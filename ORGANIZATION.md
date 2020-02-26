# Code Organization

![](material/MLJ_stack.svg)

*Dependency chart for MLJ repositories. Repositories with dashed
connections do not currently exist but are planned/proposed.*

Repositories of some possible interest outside of MLJ, or beyond
its conventional use, are marked with a ⟂ symbol:

* [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl) is the
  general user's point-of-entry for choosing, loading, composing,
  evaluating and tuning machine learning models. It pulls in most code
  from other repositories described below. (A current exception is
  [homogeneous ensembles code](src/ensembles.jl), to be migrated to
  MLJBase or its own repository MLJEnsembles.) MLJ also hosts the [MLJ
  manual](src/docs) which documents functionality across the
  repositories, with the exception of ScientificTypes, and
  MLJScientific types which host their own documentation. (The MLJ
  manual and MLJTutorials do provide overviews of scientific types.)

* [MLJModelInterface.jl](https://github.com/alan-turing-institute/MLJModelInterface.jl)
  is a lightweight package imported by packages implementing
  MLJ's interface for their machine learning models. It's *sole*
  dependency is ScientificTypes, which is a tiny package with *no*
  dependencies. 

* (⟂)
  [MLJBase.jl](https://github.com/alan-turing-institute/MLJBase.jl) is
  a large repository with two main purposes: (i) to give "dummy"
  methods defined in MLJModelInterface their intended functionality
  (which depends on third party packages, such as
  [Tables.jl](https://github.com/JuliaData/Tables.jl),
  [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
  and
  [CategoricalArrays.jl](https://github.com/JuliaData/CategoricalArrays.jl));
  and (ii) provide functionality essential to the MLJ user that has
  not been relegated to its own "satellite" repository for some
  reason. See the [MLJBase.jl
  readme](https://github.com/alan-turing-institute/MLJBase.jl) for a
  detailed description of MLJBase's contents.

* [MLJModels.jl](https://github.com/alan-turing-institute/MLJModels.jl)
  hosts the MLJ **registry**, which contains metadata on all the models
  the MLJ user can search and load from MLJ. Moreover, it provides
  the functionality for **loading model code** from MLJ on
  demand. Finally, it furnishes **model interfaces** for a number of third
  party model providers not implementing interfaces natively, such as
  [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl),
  [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl) or
  [XGBoost.jl](https://github.com/dmlc/XGBoost.jl). These packages are
  *not* imported by MLJModels and are not dependencies from the
  point-of-view of current package management.

* [MLJTuning.jl](https://github.com/alan-turing-institute/MLJTuning.jl)
  provides MLJ's interface for hyper-parameter tuning strategies, and
  selected implementations, such as grid search. 
  
* (⟂)
  [MLJLinearModels.jl](https://github.com/alan-turing-institute/MLJLinearModels.jl)
  is an experimental package for a wide range of julia-native penalized linear models
  such as Lasso, Elastic-Net, Robust regression, LAD regression,
  etc. 

* [MLJFlux.jl](https://github.com/alan-turing-institute/MLJFlux.jl) an
  experimental package for using **neural-network models**, built with
  [Flux.jl](https://github.com/FluxML/Flux.jl), in MLJ.
  
* (⟂)
  [ScientificTypes.jl](https://github.com/alan-turing-institute/ScientificTypes.jl)
  is a tiny, zero-dependency package providing "scientific" types,
  such as `Continuous`, `OrderedFactor`, `Image` and `Table`. It's
  purpose is to formalize conventions around the scientific
  interpretation of ordinary machine types, such as `Float32` and
  `DataFrame`.
  
* (⟂)
  [MLJScientificTypes.jl](https://github.com/alan-turing-institute/MLJScientificTypes.jl)
  articulates MLJ's own convention for the scientific interpretation of
  data.

* [MLJTutorials](https://github.com/alan-turing-institute/MLJTutorials)
  collects tutorials on how to use MLJ. 
