# Working with Tasks

In MLJ a *task* is a synthesis of three elements: *data*, an
*interpretation* of data, and a *learning objective*. Once one has a
task - no more and no less - one is ready to choose a learning model.

### Scientific types and the interpretation of data

Generally the columns of a table, such as a DataFrame, represents real
quantities. However, the nature of a quantity is not always clear from
the representation. For example, we might count phone calls using the
`UInt32` type but also use `UInt32` to represent a categorical
feature, such as the species of conifers. MLJ mitigates such ambiguity
by: (i) distinguishing between the machine and *[scientific
type](scientific_data_types.md)* of scalar data; (ii) disallowing the
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

```@example 1
# load a built-in task:
using MLJ
task = load_iris()
```

```@example 1 
# deconstruct:
X, y = task()
X[1:3, :]
```

```@example 1
# reconstruct:
df = copy(X)
df.species = y
task = SupervisedTask(data=df, target=:species, is_probabilistic=true)
show(task, 1)
```

```@example 1
models(task)
```

```@example 1
@load DecisionTreeClassifier
mach = machine(DecisionTreeClassifier(target_type=String), task)
evaluate!(mach, operation=predict_mode, resampling=Holdout(), measure=misclassification_rate, verbosity=0)
```


### API    

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
