# Code Organization

![](material/MLJ_stack.svg)

*Dependency chart for MLJ repositories. Repositories with dashed
connections do not currently exist but are planned/proposed.*

Repositories of some possible interest outside of MLJ, or beyond
its conventional use, are marked with a ⟂ symbol:

* [MLJ.jl](https://github.com/JuliaAI/MLJ.jl) is the general user's point-of-entry for
  choosing, loading, composing, evaluating and tuning machine learning models. It pulls in
  most code from other repositories described below.  MLJ also hosts the [MLJ
  manual](src/docs) which documents functionality across the repositories, although some
  pages point to documentation hosted locally by a particular package.
  
  
* [MLJModelInterface.jl](https://github.com/JuliaAI/MLJModelInterface.jl) is a lightweight
  package imported by packages implementing MLJ's interface for their machine learning
  models. It's only dependencies are ScientificTypesBase.jl (which depends only on the
  standard library module `Random`) and
  [StatisticalTraits.jl](https://github.com/JuliaAI/StatisticalTraits.jl) (which depends
  only on ScientificTypesBase.jl).

* (⟂) [MLJBase.jl](https://github.com/JuliaAI/MLJBase.jl) is a large repository with two
  main purposes: (i) to give "dummy" methods defined in MLJModelInterface their intended
  functionality (which depends on third party packages, such as
  [Tables.jl](https://github.com/JuliaData/Tables.jl),
  [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) and
  [CategoricalArrays.jl](https://github.com/JuliaData/CategoricalArrays.jl)); and (ii)
  provide functionality essential to the MLJ user that has not been relegated to its own
  "satellite" repository for some reason. See the [MLJBase.jl
  readme](https://github.com/JuliaAI/MLJBase.jl) for a detailed description of MLJBase's
  contents.

* [StatisticalMeasures.jl](https://github.com/JuliaAI/StatisticalMeasures.jl) provides
  performance measures (metrics) such as losses and scores.

* [MLJModels.jl](https://github.com/JuliaAI/MLJModels.jl) hosts the *MLJ model registry*,
  which contains metadata on all the models the MLJ user can search and load from
  MLJ. Moreover, it provides the functionality for **loading model code** from MLJ on
  demand. Finally, it furnishes some commonly used transformers for data pre-processing,
  such as `ContinuousEncoder` and `Standardizer`.

* [MLJTuning.jl](https://github.com/JuliaAI/MLJTuning.jl) provides MLJ's `TunedModel`
  wrapper for hyper-parameter optimization, including the extendable API for tuning
  strategies, and selected in-house implementations, such as `Grid` and `RandomSearch`.
  
* [MLJEnsembles.jl](https://github.com/JuliaAI/MLJEnsembles.jl) provides MLJ's
  `EnsembleModel` wrapper, for creating homogeneous model ensembles.
  
* [MLJIteration.jl](https://github.com/JuliaAI/MLJIteration.jl) provides the
  `IteratedModel` wrapper for controlling iterative models (snapshots, early stopping
  criteria, etc)
  
* [MLJFlow.jl](https://github.com/JuliaAI/MLJFlow.jl) provides integration with the
  platform-agnostic machine learning tracking tool [MLflow](https://mlflow.org).
  
* (⟂) [OpenML.jl](https://github.com/JuliaAI/OpenML.jl) provides integration with the
  [OpenML](https://www.openml.org) data science exchange platform
  
* (⟂) [MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl) provides a wide
  range of julia-native penalized linear models such as Lasso, Elastic-Net, Robust
  regression, LAD regression, etc.

* [MLJFlux.jl](https://github.com/FluxML/MLJFlux.jl) an experimental package for
  gradient-descent models, such as traditional neural-networks, built with
  [Flux.jl](https://github.com/FluxML/Flux.jl), in MLJ.
  
* (⟂) [ScientificTypesBase.jl](https://github.com/JuliaAI/ScientificTypesBase.jl) is an
  ultra lightweight package providing "scientific" types, such as `Continuous`,
  `OrderedFactor`, `Image` and `Table`. It's purpose is to formalize conventions around
  the scientific interpretation of ordinary machine types, such as `Float32` and
  `DataFrame`.
  
* (⟂) [ScientificTypes.jl](https://github.com/JuliaAI/ScientificTypes.jl) articulates the
  particular convention for the scientific interpretation of data that MLJ adopts
  
* (⟂) [StatisticalTraits.jl](https://github.com/JuliaAI/StatisticalTraits.jl) An ultra
  lightweight package defining fall-back implementations for a collection of traits
  possessed by statistical objects, principally models and measures (metrics).

* (⟂) [DataScienceTutorials](https://github.com/JuliaAI/DataScienceTutorials.jl) collects
  tutorials on how to use MLJ, which are deployed
  [here](https://JuliaAI.github.io/DataScienceTutorials.jl/)

* [MLJTestInterface](https://github.com/JuliaAI/MLJTestInterface.jl) provides tests for
  implementations of the MLJ model interface

* [MLJTestIntegration](https://github.com/JuliaAI/MLJTestIntegration.jl) provides tests
  for the entire MLJ ecosystem. (Called when you run `ENV["MLJ_TEST_INTEGRATION"]="true";
  Pkg.test("MLJ")`.
