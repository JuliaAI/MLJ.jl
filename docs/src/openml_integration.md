# OpenML Integration

[OpenML](https://www.openml.org) provides an integration platform for
carrying out and comparing machine learning solutions across a broad
collection of public datasets and software platforms. 

Integration of MLJ with OpenML is a work in progress. Currently
functionality is limited to querying and downloading datasets:

- `OpenML.list_datasets(; tag=nothing, filter=nothing, output_format=...)`: for listing available datasets
- `OpenML.list_tags()`: for listing all dataset tags
- `OpenML.describe(id)`: to describe a particular dataset
- `OpenML.load(id; parser=:csv)`: to download a dataset

Docstrings are given below. 

## Sample usage

```@repl new
using MLJ, DataFrames
ds = OpenML.list_datasets(
          tag = "OpenML100",
          filter = "number_instances/100..1000/number_features/1..10",
          output_format = DataFrame)
OpenML.list_tags()
OpenML.describe_dataset(37)
df = OpenML.load(37, parser=:openml) |> DataFrame
schema(df)
df = OpenML.load(37, parser=:auto)
schema(df)
```

## API

```@docs
MLJOpenML.list_datasets
MLJOpenML.describe_dataset
MLJOpenML.load
```
