### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ f0cc864c-8b26-441f-9bca-7c69b794f8ce
md"# MLJ for Data Scientists in Two Hours"

# ╔═╡ 6842b3a9-7c80-4b8a-b795-033f6f2a0674
md"""
An end-to-end application of the [MLJ
toolbox](https://alan-turing-institute.github.io/MLJ.jl/dev/) to the
Telco Customer Churn dataset, aimed at practicing data scientists
new to MLJ (Machine Learning in Julia).
"""

# ╔═╡ aa49e638-95dc-4249-935f-ddf6a6bfbbdd
md"""
MLJ is a *multi-paradigm* machine learning toolbox (i.e., not just
deep-learning).
"""

# ╔═╡ fc400b43-ced4-4953-af2a-25235e544a31
md"""
**New to machine learning?**
Try the "Introduction to Statistical Learning" notebooks at [Data
Science
Tutorials](https://juliaai.github.io/DataScienceTutorials.jl/),
starting with [this
tutorial](https://juliaai.github.io/DataScienceTutorials.jl/isl/lab-2/).
Or have a look at the [Julia Data
Science](https://github.com/JuliaDataScience/JuliaDataScience) book.
"""

# ╔═╡ 2e640a2f-253e-4522-8af5-148d95ea2900
md"""
**Want a shorter tour of key MLJ functionality?**
Try this [Lightning
Tour](https://github.com/alan-turing-institute/MLJ.jl/blob/dev/examples/lightning_tour/lightning_tour.ipynb).
"""

# ╔═╡ 5453f5d9-4dae-49ed-a6bf-47194d7e8e12
md"""
**Something more leisurely?** See
[MLJTutorial](https://github.com/ablaom/MLJTutorial.jl).
"""

# ╔═╡ 7807ea8d-887d-4fe0-828a-4cc485149963
md"""
**Completely new to Julia?** Browse [these
resources](https://julialang.org/learning/) or visit
[HelloJulia](https://github.com/ablaom/HelloJulia.jl).
"""

# ╔═╡ 4769b983-89fb-4a58-9e54-3f7b3ca87ecb
md"""
For more end-to-end examples, see [Data Science
Tutorials](https://juliaai.github.io/DataScienceTutorials.jl).
"""

# ╔═╡ 4eb8dff4-c23a-4b41-ba1f-6363f43ec697
md"""
**Topics covered**: Grabbing and preparing a dataset, basic
fit/predict workflow, constructing a pipeline to include data
pre-processing, estimating performance metrics, ROC curves, confusion
matrices, feature importance, basic feature selection, controlling iterative
models, hyper-parameter optimization (tuning).
"""

# ╔═╡ a583d175-d623-4888-95ea-2955abd45275
md"""
**Prerequisites for this tutorial.** Previous experience building,
evaluating, and optimizing machine learning models using
scikit-learn, caret, MLR, weka, or similar tool. No previous
experience with MLJ. Only fairly basic familiarity with Julia is
required. Uses
[DataFrames.jl](https://dataframes.juliadata.org/stable/) but in a
minimal way ([this
cheatsheet](https://ahsmart.com/pub/data-wrangling-with-data-frames-jl-cheat-sheet/index.html)
may help).
"""

# ╔═╡ 4830fc64-3d70-4869-b1b3-dd47e367ba54
md"**Time.** Between two and three hours, first time through."

# ╔═╡ 28197138-d6b7-433c-8d7e-8e449afd1c48
md"## Summary of methods and types introduced"

# ╔═╡ 39c599b2-3c1f-4292-a949-7f3bd292fe31
md"""
|code   | purpose|
|:-------|:-------------------------------------------------------|
|`OpenML.load(id)` | grab a dataset from [OpenML.org](https://www.openml.org)|
|`scitype(X)`      | inspect the scientific type (scitype) of object `X`|
|`schema(X)`       | inspect the column scitypes (scientific types) of a table `X`|
|`coerce(X, ...)`   | fix column encodings to get appropriate scitypes|
|`partition(data, frac1, frac2, ...; rng=...)` | vertically split `data`, which can be a table, vector or matrix|
|`unpack(table, f1, f2, ...)` | horizontally split `table` based on conditions `f1`, `f2`, ..., applied to column names|
|`@load ModelType pkg=...`           | load code defining a model type|
|`input_scitype(model)` | inspect the scitype that a model requires for features (inputs)|
|`target_scitype(model)`| inspect the scitype that a model requires for the target (labels)|
|`ContinuousEncoder`   | built-in model type for re-encoding all features as `Continuous`|# |`model1 |> model2` |> ...` | combine multiple models into a pipeline
| `measures("under curve")` | list all measures (metrics) with string "under curve" in documentation
| `accuracy(yhat, y)` | compute accuracy of predictions `yhat` against ground truth observations `y`
| `auc(yhat, y)`, `brier_loss(yhat, y)` | evaluate two probabilistic measures (`yhat` a vector of probability distributions)
| `machine(model, X, y)` | bind `model` to training data `X` (features) and `y` (target)
| `fit!(mach, rows=...)` | train machine using specified rows (observation indices)
| `predict(mach, rows=...)`, | make in-sample model predictions given specified rows
| `predict(mach, Xnew)` | make predictions given new features `Xnew`
| `fitted_params(mach)` | inspect learned parameters
| `report(mach)`        | inspect other outcomes of training
| `confmat(yhat, y)`    | confusion matrix for predictions `yhat` and ground truth `y`
| `roc(yhat, y)` | compute points on the receiver-operator Characteristic
| `StratifiedCV(nfolds=6)` | 6-fold stratified cross-validation resampling strategy
| `Holdout(fraction_train=0.7)` | holdout resampling strategy
| `evaluate(model, X, y; resampling=..., options...)` | estimate performance metrics `model` using the data `X`, `y`
| `FeatureSelector()` | transformer for selecting features
| `Step(3)` | iteration control for stepping 3 iterations
| `NumberSinceBest(6)`, `TimeLimit(60/5), InvalidValue()` | iteration control stopping criteria
| `IteratedModel(model=..., controls=..., options...)` | wrap an iterative `model` in control strategies
| `range(model,  :some_hyperparam, lower=..., upper=...)` | define a numeric range
| `RandomSearch()` | random search tuning strategy
| `TunedModel(model=..., tuning=..., options...)` | wrap the supervised `model` in specified `tuning` strategy
"""

# ╔═╡ 7c0464a0-4114-46bf-8514-99938a2932db
md"## Instantiate a Julia environment"

# ╔═╡ 40838996-676c-4b66-a0de-1721c1bc2df2
md"""
The following code replicates precisely the set of Julia packages
used to develop this tutorial. If this is your first time running
the notebook, package instantiation and pre-compilation may take a
minute or so to complete.
"""

# ╔═╡ 60fe49c1-3434-4a77-bca8-7eec7950fd82
begin
  using Pkg
  Pkg.activate(@__DIR__) # get env from TOML files in same directory as this notebook
  Pkg.instantiate()
end

# ╔═╡ e8c13e9d-7910-4a0e-9873-1b1430e635cc
md"## Warm up: Building a model for the iris dataset"

# ╔═╡ 6bf6ef98-302f-478c-b43e-d0ebe87da176
md"""
Before turning to the Telco Customer Churn dataset, we very quickly
build a predictive model for Fisher's well-known iris data set, as way of
introducing the main actors in any MLJ workflow. Details that you
don't fully grasp should become clearer in the Telco study.
"""

# ╔═╡ 33ca287e-8cba-47d1-9009-4cfb2012998f
md"""
This section is a condensed adaption of the [Getting Started
example](https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/#Fit-and-predict)
in the MLJ documentation.
"""

# ╔═╡ d5b8bf1d-6c9a-46c1-abd3-cb77d7a79683
md"""
First, using the built-in iris dataset, we load and inspect the features
`X_iris` (a table) and target variable `y_iris` (a vector):
"""

# ╔═╡ 9b7b9ade-9318-4b95-879e-dc128f3db7b3
using MLJ

# ╔═╡ 4f3f061d-259d-4479-a367-9aa3c6cf34d0
begin
  const X_iris, y_iris = @load_iris;
  schema(X_iris)
end

# ╔═╡ a546f372-4ae0-48c2-bf32-94657e65284e
y_iris[1:4]

# ╔═╡ f8f79dd2-b6de-48e8-9afd-65a5b5facac9
levels(y_iris)

# ╔═╡ 617d63f6-8a62-40db-b6c8-54f5ed90a964
md"We load a decision tree model, from the package DecisionTree.jl:"

# ╔═╡ ceda7ed0-3b98-44b3-9292-fe822525fc77
begin
  DecisionTree = @load DecisionTreeClassifier pkg=DecisionTree # model type
  model = DecisionTree(min_samples_split=5)                    # model instance
end

# ╔═╡ e69cd764-e4b8-4ff8-ae5e-4b83dcbc9675
md"""
In MLJ, a *model* is just a container for hyper-parameters of
some learning algorithm. It does not store learned parameters.
"""

# ╔═╡ 4a175e0f-4b87-4b53-8a29-32f11452654a
md"""
Next, we bind the model together with the available data in what's
called a *machine*:
"""

# ╔═╡ 3c23e9f8-49bd-4cde-a5f1-151fcbe229a2
mach = machine(model, X_iris, y_iris)

# ╔═╡ 6ad971c7-9fe5-45a9-81bc-2d5603785a09
md"""
A machine is essentially just a model (ie, hyper-parameters) plus data, but
it additionally stores *learned parameters* (the tree) once it is
trained on some view of the data:
"""

# ╔═╡ 63151d31-8428-4ebd-9d87-1d1fbb0e3997
begin
  train_rows = vcat(1:60, 91:150); # some row indices (observations are rows not columns)
  fit!(mach, rows=train_rows)
  fitted_params(mach)
end

# ╔═╡ 0f978839-cc95-4c3a-b950-fddef2a1fb10
md"""
A machine stores some other information enabling [warm
restart](https://alan-turing-institute.github.io/MLJ.jl/dev/machines/#Warm-restarts)
for some models, but we won't go into that here. You are allowed to
access and mutate the `model` parameter:
"""

# ╔═╡ 5edc98cd-df7d-428a-951b-64cb2a36c8e3
begin
  mach.model.min_samples_split  = 10
  fit!(mach, rows=train_rows) # re-train with new hyper-parameter
end

# ╔═╡ 1848109f-c94c-4cdd-b0e6-441461cc8770
md"Now we can make predictions on some other view of the data, as in"

# ╔═╡ eee948b7-2b42-439e-8cb0-de601961bc02
predict(mach, rows=71:73)

# ╔═╡ 7c98e10b-1fa3-4ab6-a87d-950c50fb2955
md"or on completely new data, as in"

# ╔═╡ 413d9c3c-a68e-4bf0-8448-4bcb089096cd
begin
  Xnew = (sepal_length = [5.1, 6.3],
          sepal_width = [3.0, 2.5],
          petal_length = [1.4, 4.9],
          petal_width = [0.3, 1.5])
  yhat = predict(mach, Xnew)
end

# ╔═╡ eb79663c-c671-4c4c-a012-a84240254fb6
md"""
These are probabilistic predictions which can be manipulated using a
widely adopted interface defined in the Distributions.jl
package. For example, we can get raw probabilities like this:
"""

# ╔═╡ ae9e9377-552c-4186-bbdd-9799f7bb2e60
pdf.(yhat, "virginica")

# ╔═╡ 5c70ee06-edb9-4789-97a7-88e3af4f10ee
md"We now turn to the Telco dataset."

# ╔═╡ bde7bc43-a4bb-4a42-b372-8c9866e51852
md"## Getting the Telco data"

# ╔═╡ d5a9e9b8-c67e-4bdd-8f3d-b82f1e7b6f7a
import DataFrames

# ╔═╡ 8de2c7b4-1950-4f05-ab04-579e5608ae53
begin
  data = OpenML.load(42178) # data set from OpenML.org
  df0 = DataFrames.DataFrame(data)
  first(df0, 4)
end

# ╔═╡ 768f369a-5f3d-4dcc-86cf-35420d9e6995
md"""
The object of this tutorial is to build and evaluate supervised
learning models to predict the `:Churn` variable, a binary variable
measuring customer retention, based on other variables that are
relevant.
"""

# ╔═╡ bc4c0eb9-3b5f-49fc-a29a-8d0445351914
md"""
In the table, observations correspond to rows, and features to
columns, which is the convention for representing all
two-dimensional data in MLJ.
"""

# ╔═╡ f04becbb-42a6-409f-be65-28d0fcca50a8
md"## Type coercion"

# ╔═╡ 1d08ed75-5377-4971-9a2e-c468345d87d1
md"> Introduces: `scitype`, `schema`, `coerce`"

# ╔═╡ eea585d4-d55b-4ccc-b5fa-cb79ebf595ef
md"""
A ["scientific
type"](https://juliaai.github.io/ScientificTypes.jl/dev/) or
*scitype* indicates how MLJ will *interpret* data. For example,
`typeof(3.14) == Float64`, while `scitype(3.14) == Continuous` and
also `scitype(3.14f0) == Continuous`. In MLJ, model data
requirements are articulated using scitypes.
"""

# ╔═╡ ca482134-299b-459a-91c5-524da38aa391
md"Here are common \"scalar\" scitypes:"

# ╔═╡ 5c7c97e4-fd8b-4ae5-ad90-b4405b216771
md"![](assets/scitypes.png)"

# ╔═╡ fd103be7-6fc9-43bc-895b-997a92b731e0
md"""
There are also container scitypes. For example, the scitype of any
`N`-dimensional array is `AbstractArray{S, N}`, where `S` is the scitype of the
elements:
"""

# ╔═╡ 9d2f0b19-2942-47ac-a525-e9b04a4bd246
scitype(["cat", "mouse", "dog"])

# ╔═╡ 1d18aca8-11f4-4104-80f0-ba7781e173cf
md"The `schema` operator summarizes the column scitypes of a table:"

# ╔═╡ accd3a09-5371-45ab-9cbc-9a2c39793333
schema(df0) |> DataFrames.DataFrame  # converted to DataFrame for better display

# ╔═╡ bde0406f-3964-42b8-b887-2b16710e5502
md"""
All of the fields being interpreted as `Textual` are really
something else, either `Multiclass` or, in the case of
`:TotalCharges`, `Continuous`. In fact, `:TotalCharges` is
mostly floats wrapped as strings. However, it needs special
treatment because some elements consist of a single space, " ",
which we'll treat as "0.0".
"""

# ╔═╡ 25e60566-fc64-4f35-9452-5298a8a4a401
begin
  fix_blanks(v) = map(v) do x
      if x == " "
          return "0.0"
      else
          return x
      end
  end
  
  df0.TotalCharges = fix_blanks(df0.TotalCharges);
end

# ╔═╡ 441317a8-3b72-4bf1-b017-96f1602f2cad
md"Coercing the `:TotalCharges` type to ensure a `Continuous` scitype:"

# ╔═╡ d2f2b67e-d06e-4c0b-8be2-2a4017c45345
coerce!(df0, :TotalCharges => Continuous);

# ╔═╡ 7c93b820-cbce-48be-a7ad-80d84f5b0070
md"Coercing all remaining `Textual` data to `Multiclass`:"

# ╔═╡ 5ac4a13e-9983-41dd-8378-5aa686f0b407
coerce!(df0, Textual => Multiclass);

# ╔═╡ 9f28e6c3-5f18-4fb2-9f43-49763e8691a1
md"""
Finally, we'll coerce our target variable `:Churn` to be
`OrderedFactor`, rather than `Multiclass`, to enable a reliable
interpretation of metrics like "true positive rate".  By convention,
the first class is the negative one:
"""

# ╔═╡ c9506e5b-c111-442c-bb0e-3a45761c733a
begin
  coerce!(df0, :Churn => OrderedFactor)
  levels(df0.Churn) # to check order
end

# ╔═╡ 8c865df5-ac95-438f-9979-7044b2f2df33
md"Re-inspecting the scitypes:"

# ╔═╡ 22690828-0992-4ed0-b2a1-fbbde543f620
schema(df0) |> DataFrames.DataFrame

# ╔═╡ 3bd753fc-163a-4b50-910b-f8a9a217eff2
md"## Preparing a holdout set for final testing"

# ╔═╡ 7b12673f-cc63-4cd9-aa36-d2e8546da46a
md"> Introduces: `partition`"

# ╔═╡ 3c98ab4e-47cd-4247-88a1-a2bf91434413
md"""
To reduce training times for the purposes of this tutorial, we're
going to dump 90% of observations (after shuffling) and split off
30% of the remainder for use as a lock-and-throw-away-the-key
holdout set:
"""

# ╔═╡ 1f6c7ad0-fe8d-44e5-a1cb-54e34396a855
df, df_test, df_dumped = partition(df0, 0.07, 0.003, # in ratios 7:3:90
                                   stratified=df.Churn,
                                   rng=123);

# ╔═╡ 4fd341cc-fb8f-4127-8037-0de5806e1a54
md"""
The reader interested in including all data can instead do `df,
df_test = partition(df0, 0.7, rng=123)`.
"""

# ╔═╡ 81aa1aa9-27d9-423f-9961-a09632c33fb0
md"## Splitting data into target and features"

# ╔═╡ fff47e76-d7ac-4377-b7cc-46f4ef988c68
md"> Introduces: `unpack`"

# ╔═╡ f1dbbc22-9844-4e1c-90fb-7f2721f6fcc7
md"""
In the following call, the column with name `:Churn` is copied over
to a vector `y`, and every remaining column, except `:customerID`
(which contains no useful information) goes into a table `X`. Here
`:Churn` is the target variable for which we seek predictions, given
new versions of the features `X`.
"""

# ╔═╡ 20713fb3-3a3b-42e7-af65-ccd25ecb9818
begin
  const y, X = unpack(df, ==(:Churn), !=(:customerID));
  schema(X).names
end

# ╔═╡ 369cb359-8512-4f80-8890-979691212d9b
intersect([:Churn, :customerID], schema(X).names)

# ╔═╡ af484868-b29c-4808-a6fa-eaf14df5d44b
md"We'll do the same for the holdout data:"

# ╔═╡ 8f3eda66-6183-4fe4-8025-5084804a9f6c
const ytest, Xtest = unpack(df_test, ==(:Churn), !=(:customerID));

# ╔═╡ 598142b5-0828-49ac-9e90-09023d201062
md"## Loading a model and checking type requirements"

# ╔═╡ c5309d64-7409-4148-b7bb-2f33ef765cc0
md"> Introduces: `@load`, `input_scitype`, `target_scitype`"

# ╔═╡ f97969e2-c15c-42cf-9625-01172c4a0081
md"""
For tools helping us to identify suitable models, see the [Model
Search](https://alan-turing-institute.github.io/MLJ.jl/dev/model_search/#model_search)
section of the manual. We will build a gradient tree-boosting model,
a popular first choice for structured data like we have here. Model
code is contained in a third-party package called
[EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl) which is
loaded as follows:
"""

# ╔═╡ edf1d726-542a-4fc1-af4f-11c7de9e21dd
Booster = @load EvoTreeClassifier pkg=EvoTrees

# ╔═╡ ee41a121-df0d-40ad-8dba-241d9b744683
md"""
Recall that a *model* is just a container for some algorithm's
hyper-parameters. Let's create a `Booster` with default values for
the hyper-parameters:
"""

# ╔═╡ be6cef47-d585-45fe-a6e5-2f0b4dca5c59
booster = Booster()

# ╔═╡ 452edd51-b878-46d0-8550-20498aa03ed0
md"""
This model is appropriate for the kind of target variable we have because of
the following passing test:
"""

# ╔═╡ fdd20843-c981-41ad-9e7b-5b943cf6b560
scitype(y) <: target_scitype(booster)

# ╔═╡ 2b25f9cb-d12d-4a6f-bce6-0a1379cc1259
md"However, our features `X` cannot be directly used with `booster`:"

# ╔═╡ b139c43b-382b-415a-9608-b5032c116833
scitype(X) <: input_scitype(booster)

# ╔═╡ 91cda5db-ba6b-453e-b473-4aa968e6937a
md"""
As it turns out, this is because `booster`, like the majority of MLJ
supervised models, expects the features to be `Continuous`. (With
some experience, this can be gleaned from `input_scitype(booster)`.)
So we need feature encoding, discussed next.
"""

# ╔═╡ fc97e937-66ca-4434-8d9e-644a1b3cc6b6
md"## Building a model pipeline to incorporate feature encoding"

# ╔═╡ b1d95557-51fc-4f1e-ac0a-23ded81445da
md"> Introduces: `ContinuousEncoder`, pipeline operator `|>`"

# ╔═╡ 38961daa-dd5f-4f97-8535-1a088a6a3228
md"""
The built-in `ContinuousEncoder` model transforms an arbitrary table
to a table whose features are all `Continuous` (dropping any fields
it does not know how to encode). In particular, all `Multiclass`
features are one-hot encoded.
"""

# ╔═╡ ccd3ffd0-ed41-497d-a39f-7893c73eef39
md"""
A *pipeline* is a stand-alone model that internally combines one or
more models in a linear (non-branching) pipeline. Here's a pipeline
that adds the `ContinuousEncoder` as a pre-processor to the
gradient tree-boosting model above:
"""

# ╔═╡ 35cf2011-4fb6-4e7a-bcca-51a27994a151
pipe = ContinuousEncoder() |> booster

# ╔═╡ 4b6e3239-fd47-495a-9b33-09b4b6661170
md"""
Note that the component models appear as hyper-parameters of
`pipe`. Pipelines are an implementation of a more general [model
composition](https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/#Composing-Models)
interface provided by MLJ that advanced users may want to learn about.
"""

# ╔═╡ ce9a3479-601a-472a-b45d-88d068bb0fa2
md"""
From the above display, we see that component model hyper-parameters
are now *nested*, but they are still accessible (important in hyper-parameter
optimization):
"""

# ╔═╡ 45c2d8f9-cbf9-4cda-92c8-6db3a590d963
pipe.evo_tree_classifier.max_depth

# ╔═╡ 232b7775-1f79-43f7-abf5-96f457eb2bdf
md"## Evaluating the pipeline model's performance"

# ╔═╡ ac84c13f-391f-405a-8a60-2e0214c059f5
md"""
> Introduces: `measures` (function), **measures:** `brier_loss`, `auc`, `accuracy`;
> `machine`, `fit!`, `predict`, `fitted_params`, `report`, `roc`, **resampling strategy** `StratifiedCV`, `evaluate`, `FeatureSelector`
"""

# ╔═╡ 4e70bf68-a78b-40dc-a38a-beb5c7157b57
md"""
Without touching our test set `Xtest`, `ytest`, we will estimate the
performance of our pipeline model, with default hyper-parameters, in
two different ways.
"""

# ╔═╡ f2f4194f-7003-43cc-81f5-507803ea9ed6
md"""
First, we'll do this "by hand" using the `fit!` and `predict`
workflow illustrated for the iris data set above, using a
holdout resampling strategy. At the same time we'll see how to
generate a **confusion matrix**, **ROC curve**, and inspect
**feature importances**.
"""

# ╔═╡ 10ca41be-b6cb-4b96-9b21-6d4eb642d87e
md"""
Then we'll apply the more typical and convenient `evaluate`
workflow, but using `StratifiedCV` (stratified cross-validation)
which is more informative.
"""

# ╔═╡ 2c4016c6-dc7e-44ec-b98c-3df1f31879bf
md"""
In any case, we need to choose some measures (metrics) to quantify
the performance of our model. For a complete list of measures, one
does `measures()`. Or we also can do:
"""

# ╔═╡ 236c098c-0189-4a96-92b7-4ceba56e97ad
measures("Brier")

# ╔═╡ 9ffe1654-7e32-4c4b-b122-214de244406c
md"""
We will be primarily using `brier_loss`, but also `auc` (area under
the ROC curve) and `accuracy`.
"""

# ╔═╡ 48413a7b-d01b-4005-8a4b-c83694978e38
md"### Evaluating by hand (with a holdout set)"

# ╔═╡ 84d7bcbc-df3c-4602-a8b6-f7ff516dedc4
md"""
Our pipeline model can be trained just like the decision tree model
we built for the iris data set. Binding all non-test data to the
pipeline model:
"""

# ╔═╡ f8552404-f95f-4484-81e1-74ef03c2e79e
mach_pipe = machine(pipe, X, y)

# ╔═╡ e0b1bffb-dbd5-4b0a-a04e-0b76409c14a7
md"""
We already encountered the `partition` method above. Here we apply
it to row indices, instead of data containers, as `fit!` and
`predict` only need a *view* of the data to work.
"""

# ╔═╡ d8ad2c21-fd27-44fb-b979-3458f2f26667
begin
  train, validation = partition(1:length(y), 0.7)
  fit!(mach_pipe, rows=train)
end

# ╔═╡ ceff13ba-a526-4ecb-97e3-6677afc6ca9f
md"We note in passing that we can access two kinds of information from a trained machine:"

# ╔═╡ b589fba6-2e0d-4bb2-b10e-562ce21caa04
md"""
- The **learned parameters** (eg, coefficients of a linear model): We use `fitted_params(mach)`
- Other **by-products of training** (eg, feature importances): We use `report(mach)`
"""

# ╔═╡ 442be3e3-b2e9-499d-8f78-ab549ef1544e
begin
  fp = fitted_params(mach_pipe);
  keys(fp)
end

# ╔═╡ f03fdd56-6c30-4d73-a8a4-a53f5149481e
md"For example, we can check that the encoder did not actually drop any features:"

# ╔═╡ 3afbe65c-18bf-4576-8704-dbab8e09b4f1
Set(fp.continuous_encoder.features_to_keep) == Set(schema(X).names)

# ╔═╡ 527b193b-737d-4701-a02f-27b6c05e4d02
md"And, from the report, extract feature importances:"

# ╔═╡ b332842d-6397-45c6-bea4-3d467d48781c
begin
  rpt = report(mach_pipe)
  keys(rpt.evo_tree_classifier)
end

# ╔═╡ 3035e6d0-3801-424d-97c5-2979af8a507d
begin
  fi = rpt.evo_tree_classifier.feature_importances
  feature_importance_table =
      (feature=Symbol.(first.(fi)), importance=last.(fi)) |> DataFrames.DataFrame
end

# ╔═╡ 8b67dcac-6a86-4502-b639-63c76c72c513
md"""
For models not reporting feature importances, we recommend the
[Shapley.jl](https://expandingman.gitlab.io/Shapley.jl/) package.
"""

# ╔═╡ 9cd42f39-942e-4011-8f5a-e1ee9eb5c15c
md"Returning to predictions and evaluations of our measures:"

# ╔═╡ 0fecce89-102f-416f-adc5-e671db8bca5d
begin
  ŷ = predict(mach_pipe, rows=validation);
  @info("Measurements",
        brier_loss(ŷ, y[validation]) |> mean,
        auc(ŷ, y[validation]),
        accuracy(mode.(ŷ), y[validation])
        )
end

# ╔═╡ 0b11a81e-42a7-4519-86f1-0ea38de21abb
md"""
Note that we need `mode` in the last case because `accuracy` expects
point predictions, not probabilistic ones. (One can alternatively
use `predict_mode` to generate the predictions.)
"""

# ╔═╡ e6f6fc92-f9e7-4472-a55c-0cb44ab816d7
md"""
While we're here, lets also generate a **confusion matrix** and
[receiver-operator
characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
(ROC):
"""

# ╔═╡ f3933688-1ea4-4239-be86-92ecfd0d2343
confmat(mode.(ŷ), y[validation])

# ╔═╡ ce90835f-6709-4c34-9cf1-facf39e3f302
md"""
Note: Importing the plotting package and calling the plotting
functions for the first time can take a minute or so.
"""

# ╔═╡ 15996832-b736-45f4-b61c-748aec38e674
using Plots

# ╔═╡ 3c4984d2-9e1a-450b-9727-ed822e4fd85d
begin
  roc_curve = roc(ŷ, y[validation])
  plt = scatter(roc_curve, legend=false)
  plot!(plt, xlab="false positive rate", ylab="true positive rate")
  plot!([0, 1], [0, 1], linewidth=2, linestyle=:dash, color=:black)
end

# ╔═╡ 27221b90-8506-4857-b2f2-e08965e5be66
md"### Automated performance evaluation (more typical workflow)"

# ╔═╡ 445146e5-e45e-450d-897c-f79712f9ec7c
md"""
We can also get performance estimates with a single call to the
`evaluate` function, which also allows for more complicated
resampling - in this case stratified cross-validation. To make this
more comprehensive, we set `repeats=3` below to make our
cross-validation "Monte Carlo" (3 random size-6 partitions of the
observation space, for a total of 18 folds) and set
`acceleration=CPUThreads()` to parallelize the computation.
"""

# ╔═╡ 562887bb-b7fb-430f-a543-f9324a87efad
md"""
We choose a `StratifiedCV` resampling strategy; the complete list of options is
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/evaluating_model_performance/#Built-in-resampling-strategies).
"""

# ╔═╡ f9be989e-2604-44c2-8657-9fba0caf3cb7
e_pipe = evaluate(pipe, X, y,
                  resampling=StratifiedCV(nfolds=6, rng=123),
                  measures=[brier_loss, auc, accuracy],
                  repeats=3,
                  acceleration=CPUThreads())

# ╔═╡ ff7cfc36-b9fc-4570-a217-e58f442fc85c
md"""
(There is also a version of `evaluate` for machines. Query the
`evaluate` and `evaluate!` doc-strings to learn more about these
functions and what the `PerformanceEvaluation` object `e_pipe` records.)
"""

# ╔═╡ 2468db48-0ffd-459e-b8ad-0340f15a03ba
md"""
While [less than ideal](https://arxiv.org/abs/2104.00673), let's
adopt the common practice of using the standard error of a
cross-validation score as an estimate of the uncertainty of a
performance measure's expected value. Here's a utility function to
calculate confidence intervals for our performance estimates based
on this practice, and it's application to the current evaluation:
"""

# ╔═╡ 0f76a79f-8675-4ec1-946d-b124a8db5f7c
using Measurements

# ╔═╡ f98c1e94-50a6-424b-b582-dfcbeb05bcc5
begin
  function confidence_intervals(e)
      measure = e.measure
      nfolds = length(measure)
      measurement = [e.measurement[j] ± std(e.per_fold[j])/sqrt(nfolds - 1)
                     for j in eachindex(measure)]
      table = (measure=measure, measurement=measurement)
      return DataFrames.DataFrame(table)
  end
  
  const confidence_intervals_basic_model = confidence_intervals(e_pipe)
end

# ╔═╡ c3f71e42-8bbe-47fe-9143-546b2286a5fe
md"## Filtering out unimportant features"

# ╔═╡ db354064-c2dd-4e6a-a7d8-c3b94fb18495
md"> Introduces: `FeatureSelector`"

# ╔═╡ 3bbb26ed-7d1e-46ac-8396-93fa872d2512
md"""
Before continuing, we'll modify our pipeline to drop those features
with low feature importance, to speed up later optimization:
"""

# ╔═╡ cdfe840d-4e87-467f-a4ac-2ddb495858ce
begin
  unimportant_features = filter(:importance => <(0.005), feature_importance_table).feature
  
  pipe2 = ContinuousEncoder() |>
      FeatureSelector(features=unimportant_features, ignore=true) |> booster
end

# ╔═╡ cfc5adc0-6524-478d-806e-1bc580dc349d
md"""
The reader can check this change makes negligible difference to the
model's performance.
"""

# ╔═╡ ab1bc84b-e4e3-45be-9703-288eae064e2a
md"## Wrapping our model in control strategies."

# ╔═╡ 0b80d69c-b60c-4cf4-b2c3-413ae5867f7d
md"> Introduces: **control strategies:** `Step`, `NumberSinceBest`, `TimeLimit`, `InvalidValue`, **model wrapper** `IteratedModel`, **resampling strategy:** `Holdout`"

# ╔═╡ 19e7e4c9-95c0-49d6-93d8-53c527b0a48c
md"""
We want to optimize the hyper-parameters of our model. Since our
model is iterative, these parameters include the (nested) iteration
parameter `pipe.evo_tree_classifier.nrounds`. Sometimes this
parameter is optimized first, fixed, and then maybe optimized again
after the other parameters. Here we take a more principled approach,
**wrapping our model in a control strategy** that makes it
"self-iterating". The strategy applies a stopping criterion to
*out-of-sample* estimates of the model performance, constructed
using an internally constructed holdout set. In this way, we avoid
some data hygiene issues, and, when we subsequently optimize other
parameters, we will always being using an optimal number of
iterations.
"""

# ╔═╡ 6ff08b40-906f-4154-af98-8dffdf3118fc
md"""
Note that this approach can be applied to any iterative MLJ model,
eg, the neural network models provided by
[MLJFlux.jl](https://github.com/FluxML/MLJFlux.jl).
"""

# ╔═╡ 8fc99d35-d8cc-455f-8620-127a8c4021ec
md"""
First, we select appropriate controls from [this
list](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/#Controls-provided):
"""

# ╔═╡ 2c5d0b23-3cb1-4c50-a1f6-fec4c3edfa7b
controls = [
    Step(1),              # increment to iteration parameter (`pipe.nrounds`)
    NumberSinceBest(n=6), # main stopping criterion
    TimeLimit(0.5/60),    # never train longer than half a minute
    InvalidValue()        # stop if NaN or ±Inf encountered
]

# ╔═╡ 9f80b5b5-95b9-4f01-82f7-021485ee0115
md"""
Now we wrap our pipeline model using the `IteratedModel` wrapper,
being sure to specify the `measure` on which internal estimates of
the out-of-sample performance will be based:
"""

# ╔═╡ 9aa97026-ebdb-4713-9ecb-94593d972599
iterated_pipe = IteratedModel(model=pipe2,
                              controls=controls,
                              measure=brier_loss, # or BrierLoss()
                              resampling=Holdout(fraction_train=0.7))

# ╔═╡ bb7a34eb-4bf1-41d9-b54b-e44bea97c579
md"""
We've set `resampling=Holdout(fraction_train=0.7)` to arrange that
data attached to our model should be internally split into a train
set (70%) and a holdout set (30%) for determining the out-of-sample
estimate of the Brier loss.
"""

# ╔═╡ 71bead68-2d38-468c-9121-4e392242994e
md"""
For demonstration purposes, let's bind `iterated_model` to all data
not in our don't-touch holdout set, and train on all of that data:
"""

# ╔═╡ f9e196d8-071c-4dd6-b221-79886442efe7
begin
  mach_iterated_pipe = machine(iterated_pipe, X, y)
  fit!(mach_iterated_pipe, force=true);
end

# ╔═╡ 76867b4d-56ab-4f24-8df5-b7371beb6b3f
md"Note that internally this training is split into two separate steps:"

# ╔═╡ a2e2a37c-5fdb-4c56-a475-d52dc8eba727
md"""
- A controlled iteration step, training on the holdout set, with the total number of iterations determined by the specified stopping criteria (based on the out-of-sample performance estimates)
- A final step that trains the atomic model on *all* available
  data using the number of iterations determined in the first step. Calling `predict` on the `mach` means using the learned parameters of the second step.
"""

# ╔═╡ ed9f284e-1e45-46e7-804b-89f5009710b0
md"## Hyper-parameter optimization (model tuning)"

# ╔═╡ b8b0e5ee-8468-4e02-a14a-b0adc2955e1c
md"> Introduces: `range`, **model wrapper** `TunedModel`, `RandomSearch`"

# ╔═╡ 50372dc8-bb10-4f9e-bd20-120a7a4020d0
md"""
We now turn to hyper-parameter optimization. A tool not discussed
here is the `learning_curve` function, which can be useful when
wanting to visualize the effect of changes to a *single*
hyper-parameter (which could be an iteration parameter). See, for
example, [this section of the
manual](https://alan-turing-institute.github.io/MLJ.jl/dev/learning_curves/)
or [this
tutorial](https://github.com/ablaom/MLJTutorial.jl/blob/dev/notebooks/04_tuning/notebook.ipynb).
"""

# ╔═╡ 3e385eb4-0a44-40f6-93a0-9d99274137e8
md"""
Fine tuning the hyper-parameters of a gradient booster can be
somewhat involved. Here we settle for simultaneously optimizing two
key parameters: `max_depth` and `η` (learning_rate).
"""

# ╔═╡ caa5153f-6633-41f7-af76-60dddeecbe6c
md"""
Like iteration control, **model optimization in MLJ is implemented as
a model wrapper**, called `TunedModel`. After wrapping a model in a
tuning strategy and binding the wrapped model to data in a machine
called `mach`, calling `fit!(mach)` instigates a search for optimal
model hyperparameters, within a specified range, and then uses all
supplied data to train the best model. To predict using that model,
one then calls `predict(mach, Xnew)`. In this way the wrapped model
may be viewed as a "self-tuning" version of the unwrapped
model. That is, wrapping the model simply transforms certain
hyper-parameters into learned parameters (just as `IteratedModel`
does for an iteration parameter).
"""

# ╔═╡ 0b74bfe8-bff6-469d-9074-c069a0e97d7e
md"""
To start with, we define ranges for the parameters of
interest. Since these parameters are nested, let's force a
display of our model to a larger depth:
"""

# ╔═╡ b17f6f89-2fd4-46de-ac4a-024c5894013e
show(iterated_pipe, 2)

# ╔═╡ 0a7ab003-9e10-44cf-82ca-0687059409ae
begin
  p1 = :(model.evo_tree_classifier.η)
  p2 = :(model.evo_tree_classifier.max_depth)
  
  r1 = range(iterated_pipe, p1, lower=-3, upper=-2, scale=x->10^x)
  r2 = range(iterated_pipe, p2, lower=2, upper=6)
end

# ╔═╡ f46af08e-ddb9-4aec-9e9e-2379bd3c438e
md"""
Nominal ranges are defined by specifying `values` instead of `lower`
and `upper`.
"""

# ╔═╡ af3023e6-920f-478d-bf9e-f9077f3deea4
md"""
Next, we choose an optimization strategy from [this
list](https://alan-turing-institute.github.io/MLJ.jl/dev/tuning_models/#Tuning-Models):
"""

# ╔═╡ 93c17a9b-b49c-4780-9b74-5a67b6e8b15f
tuning = RandomSearch(rng=123)

# ╔═╡ 21f14f18-cc5a-4ed9-b1f4-bf9663e97bb7
md"""
Then we wrap the model, specifying a `resampling` strategy and a
`measure`, as we did for `IteratedModel`.  In fact, we can include a
battery of `measures`; by default, optimization is with respect to
performance estimates based on the first measure, but estimates for
all measures can be accessed from the model's `report`.
"""

# ╔═╡ 147cc2c5-f3c2-4aec-8dca-7ecb1b94fa1b
md"""
The keyword `n` specifies the total number of models (sets of
hyper-parameters) to evaluate.
"""

# ╔═╡ b3bc6848-6b44-41e3-aeca-6e3d5d94d8fa
tuned_iterated_pipe = TunedModel(model=iterated_pipe,
                                 range=[r1, r2],
                                 tuning=tuning,
                                 measures=[brier_loss, auc, accuracy],
                                 resampling=StratifiedCV(nfolds=6, rng=123),
                                 acceleration=CPUThreads(),
                                 n=25)

# ╔═╡ fbf15fc2-347a-432a-8aa0-20af95403dd9
md"To save time, we skip the `repeats` here."

# ╔═╡ 3922d490-8018-4b8c-a120-518242409f79
md"Binding our final model to data and training:"

# ╔═╡ d59dc0af-8eb7-4ac4-bcf7-0be279ee1434
begin
  mach_tuned_iterated_pipe = machine(tuned_iterated_pipe, X, y)
  fit!(mach_tuned_iterated_pipe)
end

# ╔═╡ ed1d8575-1c50-4bc7-9df6-c78b3bed8b80
md"""
As explained above, the training we have just performed was split
internally into two separate steps:
"""

# ╔═╡ 1dc17d71-3a8f-43e2-b9cc-1427739824b3
md"""
- A step to determine the parameter values that optimize the aggregated cross-validation scores
- A final step that trains the optimal model on *all* available data. Future predictions `predict(mach, ...)` are based on this final training step.
"""

# ╔═╡ 954a354f-0a1d-4a7f-9051-24fa20a24653
md"""
From `report(mach_tuned_iterated_pipe)` we can extract details about
the optimization procedure. For example:
"""

# ╔═╡ 0f9e72c1-fb6e-4ea7-ac11-6a06d822d067
begin
  rpt2 = report(mach_tuned_iterated_pipe);
  best_booster = rpt2.best_model.model.evo_tree_classifier
end

# ╔═╡ 242b8047-0508-4ce2-8d27-497b9a4e8f4b
@info "Optimal hyper-parameters:" best_booster.max_depth best_booster.η;

# ╔═╡ af6d8f76-f96e-4136-a8e6-955351cd26f5
md"Using the `confidence_intervals` function we defined earlier:"

# ╔═╡ c0ef7138-42da-41da-bf7b-99c47ef72fd2
begin
  e_best = rpt2.best_history_entry
  confidence_intervals(e_best) |> DataFrames.DataFrame # for pretty printing
end

# ╔═╡ 84fb98ba-cb8a-4689-9b3c-c87e36798d40
md"And we can visualize the optimization results:"

# ╔═╡ e0dba28c-5f07-4305-bc52-7a03f8a4f046
plot(mach_tuned_iterated_pipe, size=(600,450))

# ╔═╡ d92c7d6c-6eda-4083-9814-5dd23028b7dd
md"## Saving our model"

# ╔═╡ 7626383d-5212-4ee5-aea9-6b095d52d246
md"> Introduces: `MLJ.save`"

# ╔═╡ f38bca90-a19e-4faa-8a6a-3fad94d47b89
md"""
Here's how to serialize our final, trained self-iterating,
self-tuning pipeline machine:
"""

# ╔═╡ 36e8600e-f5ee-4e5f-ab7f-23dfd6fe43e8
MLJ.save("tuned_iterated_pipe.jlso", mach_tuned_iterated_pipe)

# ╔═╡ 4de39bbb-e421-4e1b-873e-e6fb8e7dca1a
md"We'll deserialize this in \"Testing the final model\" below."

# ╔═╡ c1aa36a1-a81a-4ed5-9dd4-420f3ba8803c
md"## Final performance estimate;;;"

# ╔═╡ cd043584-6881-45df-b994-ae59732958cb
md"""
Finally, to get an even more accurate estimate of performance, we
can evaluate our model using stratified cross-validation and all the
data attached to our machine. Because this evaluation implies
[nested
resampling](https://mlr.mlr-org.com/articles/tutorial/nested_resampling.html),
this computation takes quite a bit longer than the previous one
(which is being repeated six times, using 5/6th of the data each
time):
"""

# ╔═╡ 9c90d7a7-1966-4d21-9aa5-a206354b401f
e_tuned_iterated_pipe = evaluate(tuned_iterated_pipe, X, y,
                                 resampling=StratifiedCV(nfolds=6, rng=123),
                                 measures=[brier_loss, auc, accuracy])

# ╔═╡ f5058fe5-53de-43ad-b666-1307eccc221d
confidence_intervals(e_tuned_iterated_pipe)

# ╔═╡ 669b492b-69a7-4d88-8f9a-f7439f35ea8f
md"""
For comparison, here are the confidence intervals for the basic
pipeline model (no feature selection and default hyperparameters):
"""

# ╔═╡ 5c1754b1-21c2-4173-a8bb-11a6d1761f50
confidence_intervals_basic_model

# ╔═╡ 93bcd4ed-940c-410b-8c71-7d4e18e2f699
md"""
So we see a small improvement in the `brier_score` and `auc`, but
these are not statistically significant improvements; default
`booster` hyper-parameters do a pretty good job.
"""

# ╔═╡ 19bb6a91-8c9d-4ed0-a590-db6dcb21b2d3
md"## Testing the final model"

# ╔═╡ fce028bb-dd04-4d7a-bec6-3fa17d8c7b35
md"""
We now determine the performance of our model on our
lock-and-throw-away-the-key holdout set. To demonstrate
deserialization, we'll pretend we're in a new Julia session (but
have and called `import`/`using` on the same packages). Then the
following should suffice to recover our model trained under
"Hyper-parameter optimization" above:
"""

# ╔═╡ 695926dd-3faf-48a0-97eb-2de8afd657be
mach_restored = machine("tuned_iterated_pipe.jlso")

# ╔═╡ 24ca2761-e264-4cf9-bba0-54e6f740a5b5
md"We compute predictions on the holdout set:"

# ╔═╡ 9883bd87-d70f-4fea-94c0-45caa9808777
begin
  ŷ_tuned = predict(mach_restored, Xtest);
  ŷ_tuned[1]
end

# ╔═╡ 9b7a12e7-2b3c-445a-adf5-76545beae885
md"And can compute the final performance measures:"

# ╔═╡ 7fa067b5-7e77-4dda-8715-73a20e2ae31b
@info("Tuned model measurements on test:",
      brier_loss(ŷ_tuned, ytest) |> mean,
      auc(ŷ_tuned, ytest),
      accuracy(mode.(ŷ_tuned), ytest)
      )

# ╔═╡ f4dc71cf-64c8-4857-aaca-a3a0d5954313
md"For comparison, here's the performance for the basic pipeline model"

# ╔═╡ 00257755-7145-45e6-83ea-8b0687d511d9
begin
  mach_basic = machine(pipe, X, y)
  fit!(mach_basic, verbosity=0)
  
  ŷ_basic = predict(mach_basic, Xtest);
  
  @info("Basic model measurements on test set:",
        brier_loss(ŷ_basic, ytest) |> mean,
        auc(ŷ_basic, ytest),
        accuracy(mode.(ŷ_basic), ytest)
        )
end

# ╔═╡ 135dac9b-0bd9-4e1d-9d1f-efc03a3fdb47
md"""
---

*This notebook was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
"""

# ╔═╡ Cell order:
# ╟─f0cc864c-8b26-441f-9bca-7c69b794f8ce
# ╟─6842b3a9-7c80-4b8a-b795-033f6f2a0674
# ╟─aa49e638-95dc-4249-935f-ddf6a6bfbbdd
# ╟─fc400b43-ced4-4953-af2a-25235e544a31
# ╟─2e640a2f-253e-4522-8af5-148d95ea2900
# ╟─5453f5d9-4dae-49ed-a6bf-47194d7e8e12
# ╟─7807ea8d-887d-4fe0-828a-4cc485149963
# ╟─4769b983-89fb-4a58-9e54-3f7b3ca87ecb
# ╟─4eb8dff4-c23a-4b41-ba1f-6363f43ec697
# ╟─a583d175-d623-4888-95ea-2955abd45275
# ╟─4830fc64-3d70-4869-b1b3-dd47e367ba54
# ╟─28197138-d6b7-433c-8d7e-8e449afd1c48
# ╟─39c599b2-3c1f-4292-a949-7f3bd292fe31
# ╟─7c0464a0-4114-46bf-8514-99938a2932db
# ╟─40838996-676c-4b66-a0de-1721c1bc2df2
# ╠═60fe49c1-3434-4a77-bca8-7eec7950fd82
# ╟─e8c13e9d-7910-4a0e-9873-1b1430e635cc
# ╟─6bf6ef98-302f-478c-b43e-d0ebe87da176
# ╟─33ca287e-8cba-47d1-9009-4cfb2012998f
# ╟─d5b8bf1d-6c9a-46c1-abd3-cb77d7a79683
# ╠═9b7b9ade-9318-4b95-879e-dc128f3db7b3
# ╠═4f3f061d-259d-4479-a367-9aa3c6cf34d0
# ╠═a546f372-4ae0-48c2-bf32-94657e65284e
# ╠═f8f79dd2-b6de-48e8-9afd-65a5b5facac9
# ╟─617d63f6-8a62-40db-b6c8-54f5ed90a964
# ╠═ceda7ed0-3b98-44b3-9292-fe822525fc77
# ╟─e69cd764-e4b8-4ff8-ae5e-4b83dcbc9675
# ╟─4a175e0f-4b87-4b53-8a29-32f11452654a
# ╠═3c23e9f8-49bd-4cde-a5f1-151fcbe229a2
# ╟─6ad971c7-9fe5-45a9-81bc-2d5603785a09
# ╠═63151d31-8428-4ebd-9d87-1d1fbb0e3997
# ╟─0f978839-cc95-4c3a-b950-fddef2a1fb10
# ╠═5edc98cd-df7d-428a-951b-64cb2a36c8e3
# ╟─1848109f-c94c-4cdd-b0e6-441461cc8770
# ╠═eee948b7-2b42-439e-8cb0-de601961bc02
# ╟─7c98e10b-1fa3-4ab6-a87d-950c50fb2955
# ╠═413d9c3c-a68e-4bf0-8448-4bcb089096cd
# ╟─eb79663c-c671-4c4c-a012-a84240254fb6
# ╠═ae9e9377-552c-4186-bbdd-9799f7bb2e60
# ╟─5c70ee06-edb9-4789-97a7-88e3af4f10ee
# ╟─bde7bc43-a4bb-4a42-b372-8c9866e51852
# ╠═d5a9e9b8-c67e-4bdd-8f3d-b82f1e7b6f7a
# ╠═8de2c7b4-1950-4f05-ab04-579e5608ae53
# ╟─768f369a-5f3d-4dcc-86cf-35420d9e6995
# ╟─bc4c0eb9-3b5f-49fc-a29a-8d0445351914
# ╟─f04becbb-42a6-409f-be65-28d0fcca50a8
# ╟─1d08ed75-5377-4971-9a2e-c468345d87d1
# ╟─eea585d4-d55b-4ccc-b5fa-cb79ebf595ef
# ╟─ca482134-299b-459a-91c5-524da38aa391
# ╟─5c7c97e4-fd8b-4ae5-ad90-b4405b216771
# ╟─fd103be7-6fc9-43bc-895b-997a92b731e0
# ╠═9d2f0b19-2942-47ac-a525-e9b04a4bd246
# ╟─1d18aca8-11f4-4104-80f0-ba7781e173cf
# ╠═accd3a09-5371-45ab-9cbc-9a2c39793333
# ╟─bde0406f-3964-42b8-b887-2b16710e5502
# ╠═25e60566-fc64-4f35-9452-5298a8a4a401
# ╟─441317a8-3b72-4bf1-b017-96f1602f2cad
# ╠═d2f2b67e-d06e-4c0b-8be2-2a4017c45345
# ╟─7c93b820-cbce-48be-a7ad-80d84f5b0070
# ╠═5ac4a13e-9983-41dd-8378-5aa686f0b407
# ╟─9f28e6c3-5f18-4fb2-9f43-49763e8691a1
# ╠═c9506e5b-c111-442c-bb0e-3a45761c733a
# ╟─8c865df5-ac95-438f-9979-7044b2f2df33
# ╠═22690828-0992-4ed0-b2a1-fbbde543f620
# ╟─3bd753fc-163a-4b50-910b-f8a9a217eff2
# ╟─7b12673f-cc63-4cd9-aa36-d2e8546da46a
# ╟─3c98ab4e-47cd-4247-88a1-a2bf91434413
# ╠═1f6c7ad0-fe8d-44e5-a1cb-54e34396a855
# ╟─4fd341cc-fb8f-4127-8037-0de5806e1a54
# ╟─81aa1aa9-27d9-423f-9961-a09632c33fb0
# ╟─fff47e76-d7ac-4377-b7cc-46f4ef988c68
# ╟─f1dbbc22-9844-4e1c-90fb-7f2721f6fcc7
# ╠═20713fb3-3a3b-42e7-af65-ccd25ecb9818
# ╠═369cb359-8512-4f80-8890-979691212d9b
# ╟─af484868-b29c-4808-a6fa-eaf14df5d44b
# ╠═8f3eda66-6183-4fe4-8025-5084804a9f6c
# ╟─598142b5-0828-49ac-9e90-09023d201062
# ╟─c5309d64-7409-4148-b7bb-2f33ef765cc0
# ╟─f97969e2-c15c-42cf-9625-01172c4a0081
# ╠═edf1d726-542a-4fc1-af4f-11c7de9e21dd
# ╟─ee41a121-df0d-40ad-8dba-241d9b744683
# ╠═be6cef47-d585-45fe-a6e5-2f0b4dca5c59
# ╟─452edd51-b878-46d0-8550-20498aa03ed0
# ╠═fdd20843-c981-41ad-9e7b-5b943cf6b560
# ╟─2b25f9cb-d12d-4a6f-bce6-0a1379cc1259
# ╠═b139c43b-382b-415a-9608-b5032c116833
# ╟─91cda5db-ba6b-453e-b473-4aa968e6937a
# ╟─fc97e937-66ca-4434-8d9e-644a1b3cc6b6
# ╟─b1d95557-51fc-4f1e-ac0a-23ded81445da
# ╟─38961daa-dd5f-4f97-8535-1a088a6a3228
# ╟─ccd3ffd0-ed41-497d-a39f-7893c73eef39
# ╠═35cf2011-4fb6-4e7a-bcca-51a27994a151
# ╟─4b6e3239-fd47-495a-9b33-09b4b6661170
# ╟─ce9a3479-601a-472a-b45d-88d068bb0fa2
# ╠═45c2d8f9-cbf9-4cda-92c8-6db3a590d963
# ╟─232b7775-1f79-43f7-abf5-96f457eb2bdf
# ╟─ac84c13f-391f-405a-8a60-2e0214c059f5
# ╟─4e70bf68-a78b-40dc-a38a-beb5c7157b57
# ╟─f2f4194f-7003-43cc-81f5-507803ea9ed6
# ╟─10ca41be-b6cb-4b96-9b21-6d4eb642d87e
# ╟─2c4016c6-dc7e-44ec-b98c-3df1f31879bf
# ╠═236c098c-0189-4a96-92b7-4ceba56e97ad
# ╟─9ffe1654-7e32-4c4b-b122-214de244406c
# ╟─48413a7b-d01b-4005-8a4b-c83694978e38
# ╟─84d7bcbc-df3c-4602-a8b6-f7ff516dedc4
# ╠═f8552404-f95f-4484-81e1-74ef03c2e79e
# ╟─e0b1bffb-dbd5-4b0a-a04e-0b76409c14a7
# ╠═d8ad2c21-fd27-44fb-b979-3458f2f26667
# ╟─ceff13ba-a526-4ecb-97e3-6677afc6ca9f
# ╟─b589fba6-2e0d-4bb2-b10e-562ce21caa04
# ╠═442be3e3-b2e9-499d-8f78-ab549ef1544e
# ╟─f03fdd56-6c30-4d73-a8a4-a53f5149481e
# ╠═3afbe65c-18bf-4576-8704-dbab8e09b4f1
# ╟─527b193b-737d-4701-a02f-27b6c05e4d02
# ╠═b332842d-6397-45c6-bea4-3d467d48781c
# ╠═3035e6d0-3801-424d-97c5-2979af8a507d
# ╟─8b67dcac-6a86-4502-b639-63c76c72c513
# ╟─9cd42f39-942e-4011-8f5a-e1ee9eb5c15c
# ╠═0fecce89-102f-416f-adc5-e671db8bca5d
# ╟─0b11a81e-42a7-4519-86f1-0ea38de21abb
# ╟─e6f6fc92-f9e7-4472-a55c-0cb44ab816d7
# ╠═f3933688-1ea4-4239-be86-92ecfd0d2343
# ╟─ce90835f-6709-4c34-9cf1-facf39e3f302
# ╠═15996832-b736-45f4-b61c-748aec38e674
# ╠═3c4984d2-9e1a-450b-9727-ed822e4fd85d
# ╟─27221b90-8506-4857-b2f2-e08965e5be66
# ╟─445146e5-e45e-450d-897c-f79712f9ec7c
# ╟─562887bb-b7fb-430f-a543-f9324a87efad
# ╠═f9be989e-2604-44c2-8657-9fba0caf3cb7
# ╟─ff7cfc36-b9fc-4570-a217-e58f442fc85c
# ╟─2468db48-0ffd-459e-b8ad-0340f15a03ba
# ╠═0f76a79f-8675-4ec1-946d-b124a8db5f7c
# ╠═f98c1e94-50a6-424b-b582-dfcbeb05bcc5
# ╟─c3f71e42-8bbe-47fe-9143-546b2286a5fe
# ╟─db354064-c2dd-4e6a-a7d8-c3b94fb18495
# ╟─3bbb26ed-7d1e-46ac-8396-93fa872d2512
# ╠═cdfe840d-4e87-467f-a4ac-2ddb495858ce
# ╟─cfc5adc0-6524-478d-806e-1bc580dc349d
# ╟─ab1bc84b-e4e3-45be-9703-288eae064e2a
# ╟─0b80d69c-b60c-4cf4-b2c3-413ae5867f7d
# ╟─19e7e4c9-95c0-49d6-93d8-53c527b0a48c
# ╟─6ff08b40-906f-4154-af98-8dffdf3118fc
# ╟─8fc99d35-d8cc-455f-8620-127a8c4021ec
# ╠═2c5d0b23-3cb1-4c50-a1f6-fec4c3edfa7b
# ╟─9f80b5b5-95b9-4f01-82f7-021485ee0115
# ╠═9aa97026-ebdb-4713-9ecb-94593d972599
# ╟─bb7a34eb-4bf1-41d9-b54b-e44bea97c579
# ╟─71bead68-2d38-468c-9121-4e392242994e
# ╠═f9e196d8-071c-4dd6-b221-79886442efe7
# ╟─76867b4d-56ab-4f24-8df5-b7371beb6b3f
# ╟─a2e2a37c-5fdb-4c56-a475-d52dc8eba727
# ╟─ed9f284e-1e45-46e7-804b-89f5009710b0
# ╟─b8b0e5ee-8468-4e02-a14a-b0adc2955e1c
# ╟─50372dc8-bb10-4f9e-bd20-120a7a4020d0
# ╟─3e385eb4-0a44-40f6-93a0-9d99274137e8
# ╟─caa5153f-6633-41f7-af76-60dddeecbe6c
# ╟─0b74bfe8-bff6-469d-9074-c069a0e97d7e
# ╠═b17f6f89-2fd4-46de-ac4a-024c5894013e
# ╠═0a7ab003-9e10-44cf-82ca-0687059409ae
# ╟─f46af08e-ddb9-4aec-9e9e-2379bd3c438e
# ╟─af3023e6-920f-478d-bf9e-f9077f3deea4
# ╠═93c17a9b-b49c-4780-9b74-5a67b6e8b15f
# ╟─21f14f18-cc5a-4ed9-b1f4-bf9663e97bb7
# ╟─147cc2c5-f3c2-4aec-8dca-7ecb1b94fa1b
# ╠═b3bc6848-6b44-41e3-aeca-6e3d5d94d8fa
# ╟─fbf15fc2-347a-432a-8aa0-20af95403dd9
# ╟─3922d490-8018-4b8c-a120-518242409f79
# ╠═d59dc0af-8eb7-4ac4-bcf7-0be279ee1434
# ╟─ed1d8575-1c50-4bc7-9df6-c78b3bed8b80
# ╟─1dc17d71-3a8f-43e2-b9cc-1427739824b3
# ╟─954a354f-0a1d-4a7f-9051-24fa20a24653
# ╠═0f9e72c1-fb6e-4ea7-ac11-6a06d822d067
# ╠═242b8047-0508-4ce2-8d27-497b9a4e8f4b
# ╟─af6d8f76-f96e-4136-a8e6-955351cd26f5
# ╠═c0ef7138-42da-41da-bf7b-99c47ef72fd2
# ╟─84fb98ba-cb8a-4689-9b3c-c87e36798d40
# ╠═e0dba28c-5f07-4305-bc52-7a03f8a4f046
# ╟─d92c7d6c-6eda-4083-9814-5dd23028b7dd
# ╟─7626383d-5212-4ee5-aea9-6b095d52d246
# ╟─f38bca90-a19e-4faa-8a6a-3fad94d47b89
# ╠═36e8600e-f5ee-4e5f-ab7f-23dfd6fe43e8
# ╟─4de39bbb-e421-4e1b-873e-e6fb8e7dca1a
# ╟─c1aa36a1-a81a-4ed5-9dd4-420f3ba8803c
# ╟─cd043584-6881-45df-b994-ae59732958cb
# ╠═9c90d7a7-1966-4d21-9aa5-a206354b401f
# ╠═f5058fe5-53de-43ad-b666-1307eccc221d
# ╟─669b492b-69a7-4d88-8f9a-f7439f35ea8f
# ╠═5c1754b1-21c2-4173-a8bb-11a6d1761f50
# ╟─93bcd4ed-940c-410b-8c71-7d4e18e2f699
# ╟─19bb6a91-8c9d-4ed0-a590-db6dcb21b2d3
# ╟─fce028bb-dd04-4d7a-bec6-3fa17d8c7b35
# ╠═695926dd-3faf-48a0-97eb-2de8afd657be
# ╟─24ca2761-e264-4cf9-bba0-54e6f740a5b5
# ╠═9883bd87-d70f-4fea-94c0-45caa9808777
# ╟─9b7a12e7-2b3c-445a-adf5-76545beae885
# ╠═7fa067b5-7e77-4dda-8715-73a20e2ae31b
# ╟─f4dc71cf-64c8-4857-aaca-a3a0d5954313
# ╠═00257755-7145-45e6-83ea-8b0687d511d9
# ╟─135dac9b-0bd9-4e1d-9d1f-efc03a3fdb47
