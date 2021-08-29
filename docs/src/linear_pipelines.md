# Linear Pipelines

In MLJ a *pipeline* is a composite model in which models are chained
together in a linear (non-branching) chain. Pipelines can include
learned or static target transformations, if one of the models is
supervised.

To illustrate basic construction of a pipeline, consider the following
toy data:

```@setup 7
using MLJ
MLJ.color_off()
```

```@example 7
using MLJ
X = (age    = [23, 45, 34, 25, 67],
	 gender = categorical(['m', 'm', 'f', 'm', 'f']));
height = [67.0, 81.5, 55.6, 90.0, 61.1]
nothing # hide
```

The code below defines a new model type, and an *instance* of that
ype called `pipe`, for performing the following operations:

- standardize the target variable `:height` to have mean zero and
  standard deviation one
- coerce the `:age` field to have `Continuous` scitype
- one-hot encode the categorical feature `:gender`
- train a K-nearest neighbor model on the transformed inputs and
  transformed target
- restore the predictions of the KNN model to the original `:height`
  scale (i.e., invert the standardization)

```@setup 7
const KNNRegressor = @load KNNRegressor pkg=NearestNeighborModels
```

```julia>
KNNRegressor = @load KNNRegressor
pipe = @pipeline(X -> coerce(X, :age=>Continuous),
				 OneHotEncoder,
				 KNNRegressor(K=3),
				 target = Standardizer())

Pipeline326(
	one_hot_encoder = OneHotEncoder(
			features = Symbol[],
			drop_last = false,
			ordered_factor = true,
			ignore = false),
	knn_regressor = KNNRegressor(
			K = 3,
			algorithm = :kdtree,
			metric = Distances.Euclidean(0.0),
			leafsize = 10,
			reorder = true,
			weights = :uniform),
	target = Standardizer(
			features = Symbol[],
			ignore = false,
			ordered_factor = false,
			count = false)) @287
```

Notice that field names for the composite are automatically generated
based on the component model type names. The automatically generated
name of the new model composite model type, `Pipeline406`, can be
replaced with a user-defined one by specifying, say,
`name=MyPipe`. **If you are planning on serializing (saving) a
pipeline-machine, you will need to specify a name.**.

The new model can be used just like any other non-composite model:

```julia
pipe.knn_regressor.K = 2
pipe.one_hot_encoder.drop_last = true
evaluate(pipe, X, height, resampling=Holdout(), measure=l2, verbosity=2)

[ Info: Training Machine{Pipeline406} @959.
[ Info: Training Machine{UnivariateStandardizer} @422.
[ Info: Training Machine{OneHotEncoder} @745.
[ Info: Spawning 1 sub-features to one-hot encode feature :gender.
[ Info: Training Machine{KNNRegressor} @005.
┌───────────┬───────────────┬────────────┐
│ _.measure │ _.measurement │ _.per_fold │
├───────────┼───────────────┼────────────┤
│ l2        │ 55.5          │ [55.5]     │
└───────────┴───────────────┴────────────┘
_.per_observation = [[[55.502499999999934]]]

```

For important details on including target transformations, see below.

```@docs
@pipeline
```
