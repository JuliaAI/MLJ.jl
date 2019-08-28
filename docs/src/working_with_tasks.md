# Working with Tasks

*Warning.* The task API described here is likely change soon, with the notion of
task being not bound to any particular data set.

In MLJ a *task* is a synthesis of three elements: *data*, an
*interpretation* of that data, and a *learning objective*. Once one has a
task one is ready to choose learning models.

### Scientific types and the interpretation of data

Generally the columns of a table, such as a DataFrame, represents real
quantities. However, the nature of a quantity is not always clear from
the representation. For example, we might count phone calls using the
`UInt32` type but also use `UInt32` to represent a categorical
feature, such as the species of conifers. MLJ mitigates such ambiguity
with the use of scientific types. See [Getting Started](index.md)) for details

Explicitly specifying scientific types during the construction of a
MLJ task is the user's opportunity to articulate how the supplied data
should be interpreted.


### Learning objectives

In MLJ specifying a learning objective means specifying: (i) whether
learning is supervised or not; (ii) whether, in the supervised case,
predictions are to be probabilistic or deterministic; and (iii) what
part of the data is relevant and what role is each part to play.


### Sample usage

Load a built-in task:

```@example 1
using MLJ
using CSV
MLJ.color_off() # hide
task = load_iris()
```

Extract input and target:

```@example 1 
X, y = task()
X[1:3, :]
```

Now, starting with some tabular data...

```@example 1
using RDatasets
df = dataset("boot", "channing");
first(df, 4)
```

...we can check MLJ's interpretation of that data:

```@example 1
schema(df)
```

And construct a task by wrapping the data in a learning objective, and
coercing the data into a form MLJ will correctly interpret. (The middle three
fields of `df` refer to ages, in months, the last is a flag.):

```@example 1
task = supervised(data=df,
                  target=:Exit,
                  ignore=:Time,
                  is_probabilistic=true,
                  types=Dict(:Entry=>Continuous,
                             :Exit=>Continuous,
                             :Cens=>Multiclass))
schema(task.X)
```

Shuffle the rows of a task:

```@example 1
task = load_iris()
using Random
rng = MersenneTwister(1234)
shuffle!(rng, task) # rng is optional
task[1:4].y
```

Counting and selecting rows of a task:

```@example 1
nrows(task)
```

```@example 1
task[1:2].y
```

Listing the models available to complete a task:

```@example 1
models(task)
```

Binding a model to a task and evaluating performance:

```@example 1
@load DecisionTreeClassifier;
mach = machine(DecisionTreeClassifier(), task)
evaluate!(mach, operation=predict_mode, resampling=Holdout(), measure=misclassification_rate, verbosity=0)
```

### API Reference

```@docs
supervised
```

```@docs
unsupervised
```

```@docs
models
```

```@docs
localmodels
```
