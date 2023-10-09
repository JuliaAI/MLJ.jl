# Correcting Class Imbalance

## Oversampling and undersampling methods

Models providing oversampling or undersampling methods, to correct for class imbalance,
are listed under [Class Imbalance](@ref). In particular, several popular algorithms are
provided by the [Imbalance.jl]() package, which includes detailed documentation and
tutorials.

## Incorporating class imbalance in supervised learning pipelines

One or more oversampling/undersampling algorithms can be fused with an MLJ classifier
using the [`BalancedModel`](@ref) wrapper. This creates a new classifier which can be
treated like any other; resampling to correct for class imbalance, relevant only for
*training* of the atomic classifier, is then carried out internally. If, for example, one
applies cross-validation to the wrapped classifier (using [`evaluate!`](@ref), say), then
this means over/undersampling is then repeated for each fold automatically.

Refer to the [MLJBalancing.jl]() documentation for further details.

```@docs
MLJBalancing.BalancedModel
```
