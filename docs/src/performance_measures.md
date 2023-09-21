# Performance Measures

**Quick link:** [List of aliases of all
measures](https://juliaai.github.io/StatisticalMeasures.jl/dev/auto_generated_list_of_measures/#aliases)

In MLJ loss functions, scoring rules, confusion matrices, sensitivities, etc, are
collectively referred to as *measures*. These measures are provided by the package
[StatisticalMeasures.jl](https://juliaai.github.io/StatisticalMeasures.jl/dev/). As this
package is a dependency of MLJ, and all its methods are re-exported, the measures are
immediately available to the MLJ user. Commonly measures are passed to MLJ meta-algorithms
(see [Uses of measures](@ref) below) but to learn how to call measures directly, see
[this](https://juliaai.github.io/StatisticalMeasures.jl/dev/examples_of_usage/)
StatisticalMeasures.jl tutorial.

A list of all measures ready to use after running `using MLJ` or `using
StatisticalMeasures`, is
[here](https://juliaai.github.io/StatisticalMeasures.jl/dev/auto_generated_list_of_measures/). Alternatively,
call [`measures()`](@ref) (experimental) to generate a dictionary keyed on available
measure constructors, with measure metadata as values.


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

- when evaluating model performance using
[`evaluate!`](@ref)/[`evaluate`](@ref) - see [Evaluating Model Performance](@ref)

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
roc_curve
```
