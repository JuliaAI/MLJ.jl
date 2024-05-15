# Preparing Data

## Splitting data

MLJ has two tools for splitting data. To split data *vertically* (that is,
to split by observations) use [`partition`](@ref). This is commonly applied to
a vector of observation *indices*, but can also be applied to datasets
themselves, provided they are vectors, matrices or tables.

To split tabular data *horizontally* (i.e., break up a table based on
feature names) use [`unpack`](@ref).

```@docs
MLJBase.partition
MLJBase.unpack
```

## Bridging the gap between data type and model requirements

As outlined in [Getting Started](@ref), it is important that the
[scientific type](https://github.com/JuliaAI/ScientificTypesBase.jl) of
data matches the requirements of the model of interest. For example,
while the majority of supervised learning models require input
features to be `Continuous`, newcomers to MLJ are sometimes
surprised at the disappointing results of [model queries](@ref
model_search) such as this one:

```@setup poot
using MLJ
```
```@example poot
X = (height   = [185, 153, 163, 114, 180],
     time     = [2.3, 4.5, 4.2, 1.8, 7.1],
     mark     = ["D", "A", "C", "B", "A"],
     admitted = ["yes", "no", missing, "yes"]);
y = [12.4, 12.5, 12.0, 31.9, 43.0]
models(matching(X, y))
```

Or are unsure about the source of the following warning:

```julia-repl
julia> Tree = @load DecisionTreeRegressor pkg=DecisionTree verbosity=0;
julia> tree = Tree();

julia> machine(tree, X, y)
┌ Warning: The scitype of `X`, in `machine(model, X, ...)` is incompatible with `model=DecisionTreeRegressor @378`:
│ scitype(X) = Table{Union{AbstractVector{Continuous}, AbstractVector{Count}, AbstractVector{Textual}, AbstractVector{Union{Missing, Textual}}}}
│ input_scitype(model) = Table{var"#s46"} where var"#s46"<:Union{AbstractVector{var"#s9"} where var"#s9"<:Continuous, AbstractVector{var"#s9"} where var"#s9"<:Count, AbstractVector{var"#s9"} where var"#s9"<:OrderedFactor}.
└ @ MLJBase ~/Dropbox/Julia7/MLJ/MLJBase/src/machines.jl:103
Machine{DecisionTreeRegressor,…} @198 trained 0 times; caches data
  args:
    1:  Source @628 ⏎ `Table{Union{AbstractVector{Continuous}, AbstractVector{Count}, AbstractVector{Textual}, AbstractVector{Union{Missing, Textual}}}}`
    2:  Source @544 ⏎ `AbstractVector{Continuous}`
```

The meaning of the warning is:

- The input `X` is a table with column scitypes `Continuous`, `Count`, and `Textual` and `Union{Missing, Textual}`, which can also see by inspecting the schema:

  ```@example poot
  schema(X)
  ```

- The model requires a table whose column element scitypes subtype `Continuous`, an incompatibility.

### Common data preprocessing workflows

There are two tools for addressing data-model type mismatches like the
above, with links to further documentation given below:

**Scientific type coercion:** We coerce machine types to obtain the
intended scientific interpretation. If `height` in the above example
is intended to be `Continuous`, `mark` is supposed to be
`OrderedFactor`, and `admitted` a (binary) `Multiclass`, then we can do

```@example poot
X_coerced = coerce(X, :height=>Continuous, :mark=>OrderedFactor, :admitted=>Multiclass);
schema(X_coerced)
```

**Data transformations:** We carry out conventional data
transformations, such as missing value imputation and feature encoding:

```@example poot
imputer = FillImputer()
mach = machine(imputer, X_coerced) |> fit!
X_imputed = transform(mach, X_coerced);
schema(X_imputed)
```

```@example poot
encoder = ContinuousEncoder()
mach = machine(encoder, X_imputed) |> fit!
X_encoded = transform(mach, X_imputed)
```

```@example poot
schema(X_encoded)
```

Such transformations can also be combined in a pipeline; see [Linear
Pipelines](@ref).


## Scientific type coercion

Scientific type coercion is documented in detail at
[ScientificTypesBase.jl](https://github.com/JuliaAI/ScientificTypesBase.jl). See
also the tutorial at the [this MLJ
Workshop](https://github.com/ablaom/MachineLearningInJulia2020)
(specifically,
[here](https://github.com/ablaom/MachineLearningInJulia2020/blob/master/tutorials.md#fixing-scientific-types-in-tabular-data))
and [this Data Science in Julia
tutorial](https://JuliaAI.github.io/DataScienceTutorials.jl/data/scitype/).

Also relevant is the section, [Working with Categorical Data](@ref).


## Data transformation

MLJ's Built-in transformers are documented at [Transformers and Other Unsupervised Models](@ref).
The most relevant in the present context are: [`ContinuousEncoder`](@ref),
[`OneHotEncoder`](@ref), [`FeatureSelector`](@ref) and [`FillImputer`](@ref).
A Gaussian mixture models imputer is provided by BetaML, which can be loaded with

```julia
MissingImputator = @load MissingImputator pkg=BetaML
```

[This MLJ
Workshop](https://github.com/ablaom/MachineLearningInJulia2020), and
the "End-to-end examples" in [Data Science in Julia
tutorials](https://JuliaAI.github.io/DataScienceTutorials.jl/)
give further illustrations of data preprocessing in MLJ.
