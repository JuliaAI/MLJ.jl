# Target Transformations

Some supervised models work best if the target variable has been
standardized, i.e., rescaled to have zero mean and unit variance.
Such a target transformation is learned from the values of the training
target variable. In particular, one generally learns a different
transformation when training on a proper subset of the training
data. Good data hygiene prescribes that a new transformation should
be computed each time the supervised model is trained on new
data - for example in cross-validation.

Additionally, one generally wants to *inverse* transform the
predictions of the supervised model for the final target predictions
to be on the original scale.

All these concerns are addressed by wrapping the supervised model
using `TransformedTargetModel`:

```@setup 123
using MLJ
MLJ.color_off()
```

```@example 123
Ridge = @load RidgeRegressor pkg=MLJLinearModels verbosity=0
ridge = Ridge(fit_intercept=false)
ridge2 = TransformedTargetModel(ridge, transformer=Standardizer())
```
Note that all the original hyperparameters, as well as those of
the `Standardizer`, are accessible as nested hyper-parameters of the
wrapped model, which can be trained or evaluated like any other:

```@example 123
X, y = make_regression(rng=1234, intercept=false)
y = y*10^5
mach = machine(ridge2, X, y)
fit!(mach, rows=1:60, verbosity=0)
predict(mach, rows=61:62)
```

Training and predicting using `ridge2` as above means:

1. Standardizing the target `y` using the first 60 rows to get a new target `z`

2. Training the original `ridge` model using the first 60 rows of `X` and `z`

3. Calling `predict` on the machine trained in Step 2 on rows `61:62` of `X`

4. Applying the inverse scaling learned in Step 1 to those predictions (to get the final output shown above)

Since both `ridge` and `ridge2` return predictions on the original scale, we can
meaningfully compare the corresponding mean absolute errors, which are indeed different in
this case.

```@example 123
evaluate(ridge, X, y, measure=l1)
```

```@example 123
evaluate(ridge2, X, y, measure=l1)
```

Ordinary functions can also be used in target transformations but an
inverse must be explicitly specified:

```@example 123
ridge3 = TransformedTargetModel(ridge, transformer=y->log.(y), inverse=z->exp.(z))
X, y = @load_boston
evaluate(ridge3, X, y, measure=l1)
```

```@docs
TransformedTargetModel
```
