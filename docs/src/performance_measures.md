# Performance Measures

## Quick links

- [List of aliases of all
  measures](https://juliaai.github.io/StatisticalMeasures.jl/dev/auto_generated_list_of_measures/#aliases)

- [Migration guide for changes to measures in MLJBase 1.0](@ref)

## Introduction

In MLJ loss functions, scoring rules, confusion matrices, sensitivities, etc, are
collectively referred to as *measures*. These measures are provided by the package
[StatisticalMeasures.jl](https://juliaai.github.io/StatisticalMeasures.jl/dev/) but are
immediately available to the MLJ user. Here's a simple example of direct application of
the `log_loss` measures to compute a training loss:

```@example measures
using MLJ
X, y = @load_iris
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
tree = DecisionTreeClassifier(max_depth=2)
mach = machine(tree, X, y) |> fit!
yhat = predict(mach, X)
log_loss(yhat, y)
```

For more examples of direct measure usage, see the StatisticalMeasures.jl
[tutorial](https://juliaai.github.io/StatisticalMeasures.jl/dev/examples_of_usage/).

A list of all measures, ready to use after running `using MLJ` or `using
StatisticalMeasures`, is
[here](https://juliaai.github.io/StatisticalMeasures.jl/dev/auto_generated_list_of_measures/). Alternatively,
call [`measures()`](@ref StatisticalMeasures.measures) (experimental) to generate a
dictionary keyed on available measure constructors, with measure metadata as values.


## Custom measures

Any measure-like object with appropriate [calling
behavior](https://juliaai.github.io/StatisticalMeasuresBase.jl/dev/implementing_new_measures/#definitions)
can be used with MLJ. To quickly build custom measures, we recommend using the package
[StatisticalMeasuresBase.jl](https://juliaai.github.io/StatisticalMeasuresBase.jl/dev/),
which provides [this](https://juliaai.github.io/StatisticalMeasuresBase.jl/dev/tutorial/)
tutorial. Note, in particular, that an "atomic" measure can be transformed into a
multi-target measure using this package.

## Uses of measures

In MLJ, measures are specified:

- when evaluating model performance using [`evaluate!`](@ref)/[`evaluate`](@ref); see
  [Evaluating Model Performance](@ref)
- when wrapping models using [`TunedModel`](@ref) - see [Tuning Models](@ref)
- when wrapping iterative models using [`IteratedModel`](@ref) - see [Controlling Iterative Models](@ref)
- when generating learning curves using [`learning_curve`](@ref) - see [Learning Curves](@ref)

and elsewhere.

## Using LossFunctions.jl

In previous versions of MLJ, measures from LossFunctions.jl were also available. Now
measures from that package must be explicitly imported and wrapped, as described
[here](https://juliaai.github.io/StatisticalMeasures.jl/dev/examples_of_usage/#Using-losses-from-LossFunctions.jl).

## Receiver operator characteristics

A related performance evaluation tool provided by StatisticalMeasures.jl, and hence by MLJ, is the `roc_curve` method:

```@docs
StatisticalMeasures.roc_curve
```

## Migration guide for changes to measures in MLJBase 1.0

Prior to MLJBase.jl 1.0 (respectivey, MLJ.jl version 0.19.6) measures were defined in
MLJBase.jl (a dependency of MLJ.jl) but now they are provided by MLJ.jl dependency
[StatisticalMeasures](https://juliaai.github.io/StatisticalMeasures.jl/dev/). Effects
on users are detailed below:


### Breaking behavior likely relevant to many users

- If `using MLJBase` without MLJ, then, in Julia 1.9 or higher, `StatisticalMeasures` must
  be explicitly imported to use measures that were previously part of MLJBase. If `using
  MLJ`, then all previous measures are still available, with the exception of those
  corresponding to LossFunctions.jl (see below).

- All measures return a *single* aggregated measurement. In other words, measures
  previously reporting a measurement *per-observation* (previously subtyping
  `Unaggregated`) no longer do so. To get per-observation measurements, use the new method
  `StatisticalMeasures.measurements(measure, ŷ, y[, weights, class_weights])`.

- The default measure for regression models (used in `evaluate/evaluate!` when `measures`
  is unspecified) is changed from `rms` to `l2=LPLoss(2)` (mean sum of squares).

- `MeanAbsoluteError` has been removed and instead `mae` is an alias for `LPLoss(p=1)`.

- Measures that previously skipped `NaN` values will now (at least by default) propagate
   those values. Missing value behavior is unchanged, except some measures that
   previously did not support `missing` now do.

- Aliases for measure *types* have been removed. For example `RMSE` (alias for
  `RootMeanSquaredError`) is gone. Aliases for instances, such as `rms` and
  `cross_entropy` persist. The exception is `precision`, for which `ppv` can
  be used in its place. (This is to avoid conflict with `Base.precision`, which was
  previously pirated.)

- `info(measure)` has been decommissioned; query docstrings or access the new [measure
  traits](https://juliaai.github.io/StatisticalMeasuresBase.jl/dev/methods/#Traits)
  individually instead. These traits are now provided by StatisticalMeasures.jl and not
  are not exported. For example, to access the orientation of the measure `rms`, do
  `import StatisticalMeasures as SM; SM.orientation(rms)`.

- Behavior of the `measures()` method, to list all measures and associated traits, has
  changed. It now returns a dictionary instead of a vector of named tuples;
  `measures(predicate)` is decommissioned, but `measures(needle)` is preserved. (This
  method, owned by StatisticalMeasures.jl, has some other search options, but is
  experimental.)

- Measures that were wraps of losses from LossFunctions.jl are no longer exposed by
  MLJBase or MLJ. To use such a loss, you must explicitly `import LossFunctions` and wrap
  the loss appropriately.  See [Using losses from
  LossFunctions.jl](https://juliaai.github.io/StatisticalMeasures.jl/dev/examples_of_usage/#Using-losses-from-LossFunctions.jl)
  for examples.

- Some user-defined measures working in previous versions of MLJBase.jl may not work
  without modification, as they must conform to the new [StatisticalMeasuresBase.jl
  API](https://juliaai.github.io/StatisticalMeasuresBase.jl/dev/implementing_new_measures/#definitions). See
  [this tutorial](https://juliaai.github.io/StatisticalMeasuresBase.jl/dev/tutorial/) on
  how define new measures.

- Measures with a "feature argument" `X`, as in `some_measure(ŷ, y, X)`, are no longer
  supported. See [What is a
  measure?](https://juliaai.github.io/StatisticalMeasuresBase.jl/dev/implementing_new_measures/#definitions)
  for allowed signatures in measures.

### Packages implementing the MLJ model interface

The migration of measures is not expected to require any changes to the source code in
packges providing implementations of the MLJ model interface (MLJModelInterface.jl) such
as MLJDecisionTreeInterface.jl and MLJFlux.jl, and this is confirmed by extensive
integration tests. However, some current tests will fail, if they use MLJBase
measures. The following should generally suffice to adapt such tests:

- Add StatisticalMeasures as test dependency, and add `using StatisticalMeasures` to your
  `runtests.jl` (and/or included submodules).

- If measures are qualified, as in `MLJBase.rms`, then the qualification must be removed
  or changed to `StatisticalMeasures.rms`, etc.

- Be aware that the default measure used in methods such as `evaluate!`, when `measure` is
  not specified, is changed from `rms` to `l2` for regression models.

- Be aware of that all measures now report a measurement for every observation, and never
  an aggregate. See second point above.

### Breaking behavior possibly relevant to some developers

- The abstract measure types `Aggregated`, `Unaggregated`, `Measure` have been
  decommissioned. (A measure is now defined purely by its [calling
  behavior](https://juliaai.github.io/StatisticalMeasuresBase.jl/dev/implementing_new_measures/#definitions).)

- What were previously exported as measure types are now only constructors.

- `target_scitype(measure)` is decommissioned. Related is
  `StatisticalMeasures.observation_scitype(measure)` which declares an upper bound on the
  allowed scitype *of a single observation*.

- `prediction_type(measure)` is decommissioned. Instead use
  `StatisticalMeasures.kind_of_proxy(measure)`.

- The trait `reports_each_observation` is decommissioned. Related is
  `StatisticalMeasures.can_report_unaggregated`; if `false` the new `measurements` method
  simply returns `n` copies of the aggregated measurement, where `n` is the number of
  observations provided, instead of individual observation-dependent measurements.

- `aggregation(measure)` has been decommissioned. Instead use
  `StatisticalMeasures.external_mode_of_aggregation(measure)`.

- `instances(measure)` has been decommissioned; query docstrings for measure aliases, or
  follow this example: `aliases = measures()[RootMeanSquaredError].aliases`.

- `is_feature_dependent(measure)` has been decommissioned. Measures consuming feature data
  are not longer supported; see above.

- `distribution_type(measure)` has been decommissioned.

- `docstring(measure)` has been decommissioned.

- Behavior of `aggregate` [has changed](https://juliaai.github.io/StatisticalMeasuresBase.jl/dev/methods/#StatisticalMeasuresBase.aggregate).

- The following traits, previously exported by MLJBase and MLJ, cannot be applied to
  measures: `supports_weights`, `supports_class_weights`, `orientation`,
  `human_name`. Instead use the traits with these names provided by
  StatisticalMeausures.jl (they will need to be qualified, as in `import
  StatisticalMeasures; StatisticalMeasures.orientation(measure)`).
