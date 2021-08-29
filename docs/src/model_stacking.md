# Model Stacking

In a model stack, as introduced by [Wolpert
(1992)](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231),
an adjucating model learns the best way to combine the predictions of
multiple base models. In MLJ, such models are constructed using the
`Stack` constructor. To learn more about stacking and to see how to
construct a stack "by hand" using the [Learning Networks](@ref)
described later, see [this Data Science in Julia
tutorial](https://alan-turing-institute.github.io/DataScienceTutorials.jl/getting-started/stacking/))

```@docs
MLJBase.Stack
```
