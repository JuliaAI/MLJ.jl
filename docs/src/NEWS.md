# MLJ News 

Development news for MLJ and its satellite packages, 
[MLJBase](https://github.com/alan-turing-institute/MLJBase.jl),
[MLJRegistry](https://github.com/alan-turing-institute/MLJRegistry.jl)
and [MLJModels](https://github.com/alan-turing-institute/MLJModels.jl)



## MLJ v0.2.2

- Resolved these isssues: 

    - Specifying new rows in calls to `fit!` on a Node not triggering
      retraining.
      [(#147)](https://github.com/alan-turing-institute/MLJ.jl/issues/147)
	
    - fit! of Node sometimes calls `update` on model when it should
      call `fit` on model
      [(#146)](https://github.com/alan-turing-institute/MLJ.jl/issues/146)
	
    - Error running the tour.ipynb notebook
      [(#140)](https://github.com/alan-turing-institute/MLJ.jl/issues/140)
	
    - For reproducibility, include a Manifest.toml file with all
      examples. [(#137)](https://github.com/alan-turing-institute/MLJ.jl/issues/137)
	
- Activated overalls code coverage
  [(#131)](https://github.com/alan-turing-institute/MLJ.jl/issues/131)
	
- Removed local version of MultivariateStats (now in MLJModels, see below).

- Minor changes to OneHotEncoder, in line with scitype philosophy.
	

## MLJBase v0.2.2

- Fix some minor bugs. 

- Added compatibility requirement CSV v0.5 or higher to allow removal
  of `allowmissing` keyword in `CSV.read`, which is to be depreciated.


## Announcement: MLJ tutorial and development sprint

 - Details
   [here](https://github.com/alan-turing-institute/MLJ.jl/wiki/2019-MLJ---sktime-tutorial-and-development-sprint)
   Applications close **May 29th** 5pm (GMTT + 1 = London)


## MLJModels v0.2.3

- The following support vector machine models from LIBSVM.jl have been
  added: EpsilonSVR, LinearSVC, NuSVR, NuSVC, SVC, OneClassSVM.

## MLJModels v0.2.2

- MulitivariateStats models RidgeRegressor and PCA migrated here from
  MLJ. Addresses: MLJ
  [#125](https://github.com/alan-turing-institute/MLJ.jl/issues/125).


## MLJModels v0.2.1

- ScikitLearn wraps ElasticNet and ElasticNetCV now available (and
  registered at MLJRegistry). Resolves: MLJ
  [#112](https://github.com/alan-turing-institute/MLJ.jl/issues/112)


## MLJ v0.2.1 

- Fix a bug and related problem in "Getting Started" docs:
  [#126](https://github.com/alan-turing-institute/MLJ.jl/issues/126 .


### MLJBase 0.2.0, MLJModels 0.2.0, MLJ 0.2.0

- Model API refactored to resolve
  [#93](https://github.com/alan-turing-institute/MLJ.jl/issues/93) and
  [#119](https://github.com/alan-turing-institute/MLJ.jl/issues/119)
  and hence simplify the model interface. This breaks all
  implementations of supervised models, and some scitype
  methods. However, for the regular user the effects are restricted
  to: (i) no more `target_type` hyperparameter for some models; (ii)
  `Deterministic{Node}` is now `DeterministicNetwork` and
  `Probabillistic{Node}` is now `ProbabilisticNetwork` when exporting
  learning networks as models.
  
- New feature: Task constructors now allow the user to explicitly
  specify scitypes of features/target. There is a `coerce` method for
  vectors and tables for the user who wants to do this
  manually. Resolves:
  [#119](https://github.com/alan-turing-institute/MLJ.jl/issues/119)


### Official registered versions of MLJBase 0.1.1, MLJModels 0.1.1, MLJ 0.1.1 released

- Minor revisions to the repos, doc updates, and a small breaking
  change around scitype method names and associated traits. Resolves:
  [#119](https://github.com/alan-turing-institute/MLJ.jl/issues/119)

### unversioned commits 12 April 2019 (around 00:10, GMT)

- Added out-of-bag estimates for performance in homogeneous
  ensembles. Resolves:
  [#77](https://github.com/alan-turing-institute/MLJ.jl/issues/77)


### unversioned commits 11 April 2019 (before noon, GMT)

- Removed dependency on unregistered package TOML.jl (using, Pkg.TOML
  instead). Resolves
  [#113](https://github.com/alan-turing-institute/MLJ.jl/issues/113)

### unversioned commits 8 April 2019 (some time after 20:00 GMT)

- Addition of XGBoost models XGBoostRegressor, XGBoostClassifier and XGBoostCount. Resolves [#65](https://github.com/alan-turing-institute/MLJ.jl/issues/65).

- Documentation reorganized as [GitHub pages](https://alan-turing-institute.github.io/MLJ.jl/dev/). Includes some additions but still a work in progress.

### unversioned commits 1 March 2019 (some time after 03:50 GMT)

- Addition of "scientific type" hierarchy, including `Continuous`,
  `Discrete`, `Multiclass`, and `Other` subtypes of `Found` (to
  complement `Missing`). See [Getting Started](index.md) for more one
  this.  Resolves:
  [#86](https://github.com/alan-turing-institute/MLJ.jl/issues/86)

- Revamp of model traits to take advantage of scientific types, with
  `output_kind` replaced with `target_scitype_union`, `input_kind` replaced
  with `input_scitype`. Also, `output_quantity` dropped,
  `input_quantity` replaced with `Bool`-valued
  `input_is_multivariate`, and `is_pure_julia` made `Bool`-valued.
  Trait definitions in all model implementations and effected
  meta-algorithms have been updated. Related:
  [#81](https://github.com/alan-turing-institute/MLJ.jl/issues/81)
  
- Substantial update of the core guide [Adding New
  Models](adding_models_for_general_use.md) to reflect above changes and in
  response to new model implementer queries. Some design "decisions"
  regarding multivariate targets now explict there.

- the order the `y` and `yhat` arguments of measures (aka loss
  functions) have been reversed. Progress on:
  [#91](https://github.com/alan-turing-institute/MLJ.jl/issues/91)
  
- Update of Standardizer and OneHotEncoder to mesh with new scitypes.

- New improved task constructors infer task metadata from data
  scitypes. This brings us close to a simple implementation of basic
  task-model matching. Query the doc-strings for `SupervisedTask` and
  `UnsupervisedTask` for details.  Machines can now dispatch on tasks
  instead of `X` and `y`. A task, `task`, is now callable: `task()`
  returns `(X, y)` for supervised models, and `X` for unsupervised
  models.  Progress on:  [\#86](https://github.com/alan-turing-institute/MLJ.jl/issues/68)

- the data in the `load_ames()` test task has been replaced by the
  full data set, and `load_reduced_ames()` now loads a reduced set.



