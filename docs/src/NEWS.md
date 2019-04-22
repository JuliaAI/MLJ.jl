# MLJ News 

Development news for MLJ and its satellite packages, 
[MLJBase](https://github.com/alan-turing-institute/MLJBase.jl),
[MLJRegistry](https://github.com/alan-turing-institute/MLJRegistry.jl)
and [MLJModels](https://github.com/alan-turing-institute/MLJModels.jl)


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



