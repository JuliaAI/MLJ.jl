# OpenML Integration

[OpenML](https://www.openml.org) provides an integration platform for
carrying out and comparing machine learning solutions across a broad
collection of public datasets and software platforms. Integration of
MLJ with OpenML is a work in progress.

## Loading IRIS Dataset

As an example, we will try to load iris dataset using `OpenML.load(taskID)`.

```@example OpenML
using MLJ.MLJBase
```

## Task ID

`OpenML.load` requires task ID of the the dataset to be loaded. This ID can 
be found on OpenML website. The task ID for iris dataset is 61, as mentioned in this [OpenML Page](https://www.openml.org/d/61)

```@repl OpenML
rowtable = OpenML.load(61)
```

## Coverting to DataFrame

```@repl OpenML
using DataFrames
df = DataFrame(rowtable)
df2 = coerce(df, :class=>Multiclass)
```

```@docs
OpenML.load
```
