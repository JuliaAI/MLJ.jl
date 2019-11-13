# MLJ News 

News for MLJ and its satellite packages: [MLJBase](https://github.com/alan-turing-institute/MLJBase.jl),
[MLJModels](https://github.com/alan-turing-institute/MLJModels.jl), and
[ScientificTypes](https://github.com/alan-turing-institute/ScientificTypes.jl).


## Latest release notes

[MLJ](https://github.com/alan-turing-institute/MLJ.jl/releases) (general users)

[MLJBase](https://github.com/alan-turing-institute/MLJBase.jl/releases) | 
[MLJModels](https://github.com/alan-turing-institute/MLJModels.jl/releases) | 
[ScientificTypes](https://github.com/alan-turing-institute/ScientificTypes.jl/releases) (mainly for developers)




## News

*Note:* New patch releases are no longer being announced below. Refer to the
links above for complete release notes.


### 30 Oct 2019

- MLJModels 0.5.3 released.

- MLJBase 0.7.2 released.

### 22 Oct 2019

MLJ 0.5.1 released.

### 21 Oct 2019

- MLJBase 0.7.1 released.

- ScientificTypes 0.2.2 released.

- MLJModels  0.5.2 released.

### 17 Oct 2019

MLJBase 0.7 released.

### 11 Oct 2019

MLJModels 0.5.1 released.

### 30 Sep 2019

MLJ 0.5 released.

### 29 Sep 2019

MLJModels 0.5 released.

### 26 Sep 2019

MLJBase 0.6 released.


## Older release notes


### MLJ 0.4.0

-  (Enhancment) Update to MLJBase 0.5.0 and MLJModels 0.4.0. In
   particular, this updates considerably the list of wrapped
   scikit-learn models available to the MLJ user:

  * [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl)
    * **SVM**: `SVMClassifier`, `SVMRegressor`, `SVMNuClassifier`,
      `SVMNuRegressor`, `SVMLClassifier`, `SVMLRegressor`,
    * **Linear Models** (regressors): `ARDRegressor`,
      `BayesianRidgeRegressor`, `ElasticNetRegressor`,
      `ElasticNetCVRegressor`, `HuberRegressor`, `LarsRegressor`,
      `LarsCVRegressor`, `LassoRegressor`, `LassoCVRegressor`,
      `LassoLarsRegressor`, `LassoLarsCVRegressor`,
      `LassoLarsICRegressor`, `LinearRegressor`,
      `OrthogonalMatchingPursuitRegressor`,
      `OrthogonalMatchingPursuitCVRegressor`,
      `PassiveAggressiveRegressor`, `RidgeRegressor`,
      `RidgeCVRegressor`, `SGDRegressor`, `TheilSenRegressor`

- (Enhancement) The macro `@pipeline` allows one to construct linear
  (non-branching) pipeline composite models with one line of code. One
  may include static transformations (ordinary functions) in the
  pipeline, as well as target transformations for the supervised case
  (when one component model is supervised).

- (Breaking) Source nodes (type `Source`) now have a `kind` field,
  which is either `:input`,`:target` or `:other`, with `:input` the
  default value in the `source` constructor.  If building a learning
  network, and the network is to be exported as a standalone model,
  then it is now necessary to tag the source nodes accordingly, as in
  `Xs = source(X)` and `ys = source(y, kind=:target)`.

- (Breaking) By virtue of the preceding change, the syntax for
  exporting a learning network is simplified. Do`?@from_network` for
  details. Also, one now uses `fitresults(N)` instead of `fit
  results(N, X, y)` and `fitresults(N, X)` when exporting a learning
  network `N` "by hand"; see the updated
  [manual](https://github.com/alan-turing-institute/MLJ.jl/blob/pipelines/docs/src/composing_models.md)
  for details.

- (Breaking) One must explicitly state if a supervised learning
  network being exported with `@from_network` is probabilistic by
  adding `is_probablistic=true` to the macro expression. Before, this
  information was unreliably inferred from the network.

- (Enhancement) Add macro-free method for loading model code into an arbitrary
  module. Do `?load` for details.
  
- (Enhancement) `@load` now returns a mode instance with default
  hyperparameters (instead of nothing), as in `tree_model = @load
  DecisionTreeRegressor`
  
- (Breaking) `info("PCA")` now returns a named-tuple, instead of a
  dictionary, of the properties of a the model named "PCA"

- (Breaking) The list returned by `models(conditional)` is now a list
  of complete metadata entries (named-tuples, as returned by
  `info`). An entry `proxy` appears in the list exactly when
  `conditional(proxy) == true`.  Model query is simplified; for
  example `models() do model model.is_supervised &&
  model.is_pure_julia end` finds all pure julia supervised models.
  
- (Bug fix) Introduce new private methods to avoid relying on MLJBase
  type piracy [MLJBase #30](https://github.com/alan-turing-institute/MLJBase.jl/issues/30).
  
- (Enhancement) If `composite` is a a learning network exported as a
  model, and `m = machine(composite, args...)` then `report(m)`
  returns the reports for each machine in the learning network, and
  similarly for `fitted_params(m)`.

- (Enhancement) `MLJ.table`, `vcat` and `hcat` now overloaded for
  `AbstractNode`, so that they can immediately be used in defining
  learning networks. For example, if `X = source(rand(20,3))` and
  `y=source(rand(20))` then `MLJ.table(X)` and `vcat(y, y)` both make
  sense and define new nodes.
  
- (Enhancement) `pretty(X)` prints a pretty version of any table `X`,
  complete with types and scitype annotations. Do `?pretty` for
  options. A wrap of `pretty_table` from `PrettyTables.jl`.
  
- (Enhancement) `std` is re-exported from `Statistics`

- (Enhancement) The [manual and MLJ cheatsheet](https://alan-turing-institute.github.io/MLJ.jl/stable/)
  have been updated.
  
- Performance measures have been migrated to MLJBase, while the model
  registry and model load/search facilities have migrated to
  MLJModels. As relevant methods are re-exported to MLJ, this is
  unlikely to effect many users.


### MLJModels 0.4.0

- (Enhancement) Add a number of
  [scikit-learn](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
  model wraps. See the above MLJ 0.4.0 release notes for a detailed
  list.
  
- The following have all been migrated to MLJModels from MLJ:

    - MLJ's built-in models (e.g., basic transformers such as `OneHotEncoder`)
	
	- The model registry metadata (src/registry/METADATA.toml)
	
	- The metadata `@update` facility for administrator registration
      of new models
	  
	- The `@load` macro and `load` function for loading code for a registered model
	
	- The `models` and `localmodels` model-search functions
	
	- The `info` command for returning the metadata entry of a model
	
- (Breaking) MLJBase v0.5.0, which introduces [some changes and additions](https://github.com/alan-turing-institute/MLJBase.jl/pull/39)
  to model traits, is a requirement, meaning the format of metadata as
  changed.
  
- (Breaking) The `model` method for retrieving model metadata has been
  renamed back to `info`, but continues to return a named-tuple. (The
  `MLJBase.info` method, returning the dictionary form of the
  metadata, is now called `MLJBase.info_dic`).


### MLJBase 0.5.0

- Bump ScientificTypes requirement to v0.2.0

- (Enhancement) The performance measures API (built-in measures +
  adaptor for external measures) from MLJ has been migrated to MLJBase.
  MLJ.
  
- (Breaking) `info`, which returns a dictionary (needed for TOML
  serialization) is renamed to `info_dic`. In this way "info" is
  reserved for a method in MLJModels/MLJ that returns a
  more-convenient named-tuple

- (Breaking) The `is_probabilistic` model trait is replaced with
  `prediction_type`, which can have the values `:deterministic`,
  `:probabilistic` or `:interval`, to allow for models predicting real
  intervals, and for consistency with measures API.
  
- (Bug fix, mildly breaking) The `package_license` model trait is now included in
  `info_dict` in the case of unsupervisd models.
  
- (Enhancement, mildly breaking) Add new model traits `hyperparameters`, 
  `hyperparameter_types`, `docstring`, and `implemented_operations` (`fit`, `predict`, `inverse_transform`, etc)
  ([#36](https://github.com/alan-turing-institute/MLJBase.jl/issues/36),
  [#37](https://github.com/alan-turing-institute/MLJBase.jl/issues/37),
  [#38](https://github.com/alan-turing-institute/MLJBase.jl/issues/38))
  
- (Enhancement) The `MLJBase.table` and `MLJBase.matrix` operations
  are now direct wraps of the corresponding `Tables.jl` operations for
  improved performance. In particular
  `MLJBase.matrix(MLJBase.table(A))` is essentially a non-operation,
  and one can pass `MLJBase.matrix` the keyword argument
  `transpose=...` .
  
- (Breaking) The built-in dataset methods `load_iris`, `load_boston`,
  `load_ames`, `load_reduced_ames`, `load_crabs` return a raw
  `DataFrame`, instead of an `MLJTask` object, and continue to require
  `import CSV` to become available. However, macro versions
  `@load_iris`, etc, are always available, automatically triggering
  `import CSV`; these macros return a tuple `(X, y)` of input
  `DataFrame` and target vector `y`, with scitypes appropriately
  coerced. (MLJ
  [#224](https://github.com/alan-turing-institute/MLJ.jl/issues/224))
  
- (Enhancement) `selectrows` now works for matrices. Needed to allow
  matrices as "node type" in MLJ learning networks; see [MLJ #209](https://github.com/alan-turing-institute/MLJ.jl/issues/209).

- (Bug) Fix problem with `==` for `MLJType` objects
  ([#35](https://github.com/alan-turing-institute/MLJBase.jl/issues/35))

- (Breaking) Update requirement on ScientficTypes.jl to v0.2.0 to
  mitigate bug with coercion of column scitypes for tables that are
  also AbstractVectors, and to make `coerce` more convenient.

- (Enhancement) Add new method `unpack` for splitting tables, as in `y, X = unpack(df,==(:target),!=(:dummy))`. See  doc-string for details. 

- (Bug fix) Remove type piracy in get/setproperty! ([#30](https://github.com/alan-turing-institute/MLJBase.jl/issues/30))


### ScientificTypes 0.2.0

- (Breaking) The argument order is switched in `coerce` methods. So
  now use `coerce(v, T)` for a vector `v` and scientific type `T` and
  `coerce(X, d)` for a table `X` and dictionary `d`.

- (Feature) You can now call `coerce` on tables without needing to
  wrap specs in a dictionary, as in `scitype(X, :age => Continuous,
  :ncalls => Count)`.


### ScientficTypes 0.1.3

[Release notes](https://github.com/alan-turing-institute/ScientificTypes.jl/releases/tag/v0.1.3)


### MLJ 0.4.0 

- Introduction of traits for measures (loss functions, etc); see top
  of /src/measures.jl for definitions. This
    - allows user to use loss functions from LossFunctions.jl,
    - enables improved measure checks and error message reporting with measures
    - allows `evaluate!` to report per-observation measures when
      available (for later use by Bayesian optimisers, for example)
    - allows support for sample-weighted measures playing nicely
      with rest of API

- Improvements to resampling: 
    - `evaluate!` method now reports per-observation measures when
      available
    - sample weights can be passed to `evaluate!` for use by measures
      that support weights
    - user can pass a list of train/evaluation pairs of row indices
      directly to `evaluate!`, in place of a `ResamplingStrategy`
      object
    - implementing a new `ResamplingStrategy` is now straightforward (see docs)
    - one can call `evaluate` (no exclamation mark) directly on
      model + data without first constructing a machine, if desired

- Doc strings and the
  [manual](https://alan-turing-institute.github.io/MLJ.jl/dev/) have
  been revised and updated. The manual includes a new section "Tuning
  models", and extra material under "Learning networks" explaining how
  to export learning networks as stand-alone models using the
  `@from_network` macro.

- Improved checks and error-reporting for binding models to data in
  machines.

- (Breaking) CSV is now an optional dependency, which means you now
  need to import CSV before you can load tasks with `load_boston()`,
  `load_iris()`, `load_crabs()`, `load_ames()`, `load_reduced_ames()`

- Added `schema` method for tables (re-exported from
  ScientificTypes.jl). Returns a named tuple with keys `:names`,
  `:types`, `:scitypes` and `:nrows`.

- (Breaking) Eliminate `scitypes` method. The scientific types of a
  table are returned as part of ScientificTypes `schema` method (see
  above)


### MLJModels 0.3.0

[Release notes](https://github.com/alan-turing-institute/MLJModels.jl/releases)


### MLJBase v0.4.0

[Release notes](https://github.com/alan-turing-institute/MLJBase.jl/releases/tag/v0.4.0)


### ScientificTypes 0.1.2

- New
  [package](https://github.com/alan-turing-institute/ScientificTypes.jl)
  to which the scientific types API has been moved (from MLJBase).


### MLJBase v0.3.0

- Make CSV an optional dependency (breaking). To use `load_iris()`,
  `load_ames()`, etc, need first to import CSV.


### MLJBase v0.2.4

- Add ColorImage and GreyImage scitypes

- Overload `in` method for subtypes of `Model` (apparently causing
  Julia crashes in an untagged commit, because of a method signature
  ambiguity, now resolved).


### MLJ v0.2.5

- Add MLJ [cheatsheet](https://github.com/alan-turing-institute/MLJ.jl/blob/master/docs/src/mlj_cheatsheet.md)

- Allow `models` to query specific traits, in addition to tasks. Query `?models` for details

- add `@from_networks` macro for exporting learning networks as models (experimental). 


### MLJModels v0.2.4

- Add compatibility requirement MLJBase="0.2.3" 


### MLJBase v0.2.3

- Small changes on definitions of `==` and `isequal` for `MLJType`
  objects. In particular, fields that are random number generators may
  change state without effecting an object's `==` equivalence class. 
  
- Add `@set_defaults` macro for generating keywork constructors for
    `Model` subtypes. 
	
- Add abstract type `UnsupervisedNetwork <: Unsupervised`.


### MLJ v0.2.3

- Fixed bug in models(::MLJTask) method which excluded some relevant
  models. [(#153)](https://github.com/alan-turing-institute/MLJ.jl/issues/153)

- Fixed some broken links to the tour.ipynb.



### MLJ v0.2.2

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
	

### MLJBase v0.2.2

- Fix some minor bugs. 

- Added compatibility requirement CSV v0.5 or higher to allow removal
  of `allowmissing` keyword in `CSV.read`, which is to be depreciated.


### Announcement: MLJ tutorial and development sprint

 - Details
   [here](https://github.com/alan-turing-institute/MLJ.jl/wiki/2019-MLJ---sktime-tutorial-and-development-sprint)
   Applications close **May 29th** 5pm (GMTT + 1 = London)


### MLJModels v0.2.3

- The following support vector machine models from LIBSVM.jl have been
  added: EpsilonSVR, LinearSVC, NuSVR, NuSVC, SVC, OneClassSVM.

### MLJModels v0.2.2

- MulitivariateStats models RidgeRegressor and PCA migrated here from
  MLJ. Addresses: MLJ
  [#125](https://github.com/alan-turing-institute/MLJ.jl/issues/125).


### MLJModels v0.2.1

- ScikitLearn wraps ElasticNet and ElasticNetCV now available (and
  registered at MLJRegistry). Resolves: MLJ
  [#112](https://github.com/alan-turing-institute/MLJ.jl/issues/112)


### MLJ v0.2.1 

- Fix a bug and related problem in "Getting Started" docs:
  [#126](https://github.com/alan-turing-institute/MLJ.jl/issues/126 .


#### MLJBase 0.2.0, MLJModels 0.2.0, MLJ 0.2.0

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


#### Official registered versions of MLJBase 0.1.1, MLJModels 0.1.1, MLJ 0.1.1 released

- Minor revisions to the repos, doc updates, and a small breaking
  change around scitype method names and associated traits. Resolves:
  [#119](https://github.com/alan-turing-institute/MLJ.jl/issues/119)

#### unversioned commits 12 April 2019 (around 00:10, GMT)

- Added out-of-bag estimates for performance in homogeneous
  ensembles. Resolves:
  [#77](https://github.com/alan-turing-institute/MLJ.jl/issues/77)


#### unversioned commits 11 April 2019 (before noon, GMT)

- Removed dependency on unregistered package TOML.jl (using, Pkg.TOML
  instead). Resolves
  [#113](https://github.com/alan-turing-institute/MLJjl/issues/113)

#### unversioned commits 8 April 2019 (some time after 20:00 GMT)

- Addition of XGBoost models XGBoostRegressor, XGBoostClassifier and XGBoostCount. Resolves [#65](https://github.com/alan-turing-institute/MLJ.jl/issues/65).

- Documentation reorganized as [GitHub pages](https://alan-turing-institute.github.io/MLJ.jl/dev/). Includes some additions but still a work in progress.

#### unversioned commits 1 March 2019 (some time after 03:50 GMT)

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



