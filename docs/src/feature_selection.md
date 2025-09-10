# Feature Selection

For manually selecting features by hand, use the [FeatureSelector](@ref) transformer.

## Recursive feature elimination

Supervised models that report feature importances can be wrapped using
[`RecursiveFeatureElmination`](@ref) method, to carry out [recursive feature
elimination](https://link.springer.com/article/10.1023/A:1012487302797). A model (or model
type) `m` reports feature importances if `reports_feature_importances(m)` is `true`.

See the [FeatureSelection.jl
documentation](https://juliaai.github.io/FeatureSelection.jl/dev/) for examples, including
recursive feature elimination with cross-validation to learn the optimal number of
features to retain.

## Reference

```@docs
FeatureSelector
RecursiveFeatureElimination
```

