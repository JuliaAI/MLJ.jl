# Freezing Learned Parameters

If a model has a hyper-parameter called `frozen`, and that model is bound to data in a
machine `mach`, then calling `fit!(mach; kwargs...)` has no effect on the learned
parameters, unles `mach` has not yet been trained. This is true even if other
hyper-parameters have changed, or one specifies new views of the data, as in `fit(mach,
rows=...)`.  This can be useful for freezing one component model in a [`Pipeline`](@ref)
or model [`Stack`](@ref) when retraining that component is expensive.

While most models don't have the `frozen` hyper-parameter, you can achieve the same effect
by wrapping your model using `Freezable` as explained below.

By wrapping a self-tuning model, [`TunedModel`](@ref)`(model; ...)` in `Freezable`, you
can evaluate its performance using [`evaluate`](@ref) without the usual expensive nested
cross-validation. The tuned parameters are then based just on the first `evaluate`
training fold. So results must be interpreted carefully. 

!!! warning Experimental feature

    The `Freeable` wrapper is not a mature feature. It is difficult to reason about in parallized workflows  (e.g., when an `acceleration` option is not the default `CPU1()`) and  may have unexpected behaviour in those cases.
	
	
```@docs
Freezable
```
