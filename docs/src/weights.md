# Weights

In machine learning it is possible to assign each observation an
independent significance, or *weight*, either in **training** or in
**performance evaluation**, or both. 

There are two kinds of weights in use in MLJ:

- *per observation weights* (also just called *weights*) refer to
  weight vectors of the same length as the number of observations

- *class weights* refer to dictionaries keyed on the target classes
  (levels) for use in classification problems
  

## Specifying weights in training

To specify weights in training you bind the weights to the model along
with the data when constructing a machine.  For supervised models the
weights are specified last:

```julia
KNNRegressor = @load KNNRegressor
model = KNNRegressor()
X, y = make_regression(10, 3)
w = rand(length(y))

mach = machine(model, X, y, w) |> fit!
```

Note that `model` supports per observation weights if
`supports_weights(model)` is `true`. To list all such models, do

```julia
models() do m
    m.supports_weights
end
```

The model `model` supports class weights if
`supports_class_weights(model)` is `true`.


## Specifying weights in performance evaluation

When calling an MLJ measure (metric) that supports weights, provide the
weights as the last argument, as in

```julia
_, y = @load_iris
ŷ = shuffle(y)
w = Dict("versicolor" => 1, "setosa" => 2, "virginica"=> 3)
macro_f1score(ŷ, y, w)
```

You can use `supports_weights` and `supports_class_weights` on
measures to check weight support. For example, to list all measures
supporting per observation weights, do

```julia 
measures() do m 
   m.supports_weights 
end 
```

See also [Evaluating Model Performance](@ref).

To pass weights to all the measures listed in an `evaluate!/evaluate`
call, use the keyword specifiers `weights=...` or
`class_weights=...`. For details, see [`evaluate!`](@ref).
