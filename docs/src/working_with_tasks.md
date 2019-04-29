# Working with Tasks

In MLJ a *task* is a synthesis of three elements: *data*, an
*interpretation* of that data, and a *learning objective*. Once one has a
task one is ready to choose learning models.

### Scientific types and the interpretation of data

Generally the columns of a table, such as a DataFrame, represents real
quantities. However, the nature of a quantity is not always clear from
the representation. For example, we might count phone calls using the
`UInt32` type but also use `UInt32` to represent a categorical
feature, such as the species of conifers. MLJ mitigates such ambiguity
by: (i) distinguishing between the machine and *[scientific
type](index.md)* of scalar data; (ii) disallowing the
representation of multiple scientific types by the same machine type
during learning; and (iii) establising a convention for what
scientific types a given machine type may represent (see the
table at the end of [Getting Started](index.md)).

Explicitly specifying scientific types during the construction of a
MLJ task is the user's opportunity to articulate how the supplied data
should be interpreted.

> WIP: At present scitypes cannot be specified and the user must manually coerce data before task construction. 


### Learning objectives

In MLJ specifying a learing objective means specifying: (i) whether
learning is supervised or not; (ii) whether, in the supervised case,
predictions are to be probabilistic or deterministic; and (iii) what
part of the data is relevant and what role is each part to play.


### Sample usage

Load a built-in task:

```@example 1
using MLJ
MLJ.color_off() # hide
task = load_iris()
```

Extract input and target:

```@example 1 
X, y = task()
X[1:3, :]
```

Supposing we have some new data, say

```@example 1
coltable = (height = Float64[183, 145, 160, 78, 182, 76],
            gender = categorical([:m, :f, :f, :f, :m, :m]),
            weight = Float64[92, 67, 62, 25, 80, 31],
            age = Float64[53, 12, 60, 5, 31, 7],
            overall_health = categorical([1, 2, 1, 3, 3, 1], ordered=true))
```

we can check MLJ's default interpretation of that data:

```@example 1
scitypes(X)
```

And construct a associated task:

```@example 1
task = SupervisedTask(data=coltable, target=:overall_health, ignore=:gender, is_probabilistic=true)
```

> WIP: In the near future users will be able to override the default interpretation of the data.

To list models matching a task:

```@example 1
models(task)
```

Row selection for a task:

```@example 1
nrows(task)
```

```@example 1
task[1:2].y
```

Shuffle the rows of a task:

```@example 1
task = load_iris()
using Random
rng = MersenneTwister(1234)
shuffle!(rng, task) # rng is optional
task[1:4].y
```

Binding a model to a task and evalutating performance:

```@example 1
@load DecisionTreeClassifier
mach = machine(DecisionTreeClassifier(target_type=String), task)
evaluate!(mach, operation=predict_mode, resampling=Holdout(), measure=misclassification_rate, verbosity=0)
```


### API Reference   

```@docs
UnsupervisedTask
```

```@docs
SupervisedTask
```

```@docs
models()
```

```@docs
localmodels()
```
