# Thresholding Probabilistic Predictors

Although one can call `predict_mode` on a probabilistic binary
classifier to get deterministic predictions, a more flexible strategy
is to wrap the model using `BinaryThresholdPredictor`, as this allows
the user to specify the threshold probability for predicting a
positive class. This wrapping converts a probabilistic classifier into a
deterministic one.

The positive class is always the second class returned when calling
`levels` on the training target `y`.

```@docs
MLJModels.BinaryThresholdPredictor
```
