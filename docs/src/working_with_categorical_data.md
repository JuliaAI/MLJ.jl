# Working with Categorical Data

## Scientific types for discrete data

Recall that models articulate their data requirements using scientific
types (see [Getting Started](@ref) or the MLJScientificTypes.jl
[documentation](https://alan-turing-institute.github.io/MLJScientificTypes.jl/dev/)). There
are three scientific types discrete data can have: `Count`,
`OrderedFactor` and `Multiclass`.


### Count data

In MLJ you cannot use integers to represent (finite) categorical
data. Integers are reserved for discrete data you want interpreted as
`Count <: Infinite`:

```@example hut
using MLJ # hide
scitype([1, 4, 5, 6])
```

The `Count` scientific type includes things like the number of phone
calls, or city populations, and other "frequency" data of a generally
unbounded nature.

That said, you may have data that is theoretically `Count`, but which
you coerce to `OrderedFactor` to enable the use of more models,
trusting to your knowledge of how those models work to inform an
appropriate interpretation.


### OrderedFactor and Multiclass data

Other integer data, such as the number of an animal's legs, or number
of rooms of homes, are generally coerced to `OrderedFactor <:
Finite`. The other categorical scientific type is `Multiclass <:
Finite`, which is for *unordered* categorical data. Coercing data to
one of these two forms is discussed under [ Detecting and coercing
improperly represented categorical data](@ref) below.


### Binary data

There is no separate scientific type for binary data. Binary data is
either `OrderedFactor{2}` if ordered, and `Multiclass{2}` otherwise.
Data with type `OrderedFactor{2}` is considered to have an instrinsic
"positive" class, e.g., the outcome of a medical test, and the
"pass/fail" outcome of an exam. MLJ measures, such as `true_positive`
assume the *second* class in the ordering is the "positive"
class. Inspecting and changing order is discussed below.

If data has type `Bool` it is considered `Count` data (as `Bool <:
Integer`) and in generally users will want to coerce to a binary type.


## Detecting and coercing improperly represented categorical data

One inspects the scientific type of data using `scitype` as shown
above. To inspect all column scientific types in a table
simultaneously, use `schema`. (Tables also have a `scitype`, in which
this information appears in a condensed form more appropriate for type
dispatch.)

```@example hut
using DataFrames
X = DataFrame(
         name       = ["Siri", "Robo", "Alexa", "Cortana"],
         gender     = ["male", "male", "Female", "female"],
         likes_soup = [true, false, false, true],
         height     = [152, missing, 148, 163],
         rating     = [2, 5, 2, 1],
         outcome    = ["rejected", "accepted", "accepted", "rejected"])
schema(X)
```

Coercing a single column:

```@example hut
X.outcome = coerce(X.outcome, OrderedFactor)
```

Inspecting the order of the levels:

```julia
levels(X.outcome)
```

Since we wish to regard "accepted" as the positive class, it should
appear second, which we correct with the `levels!` function:

```@example hut
levels!(X.outcome, ["rejected", "accepted"]);
```
Coercing all remaining types simultaneously:

```@example hut
Xnew = coerce(X, :gender    => Multiclass,
                 :like_soup => OrderedFactor,
                 :height    => Continuous,
                 :rating    => OrderedFactor)
schema(Xnew)
```

(For `DataFrame`s there is also in-place coercion using `coerce!`.)


## Tracking all levels

The key property of vectors of scientific type `OrderedFactor` and
 `Multiclass` is that the pool of all levels is not lost when
separating out one or more elements:

```@example hut
v = Xnew.rating
```

```@example hut
levels(v)
```

```@example hut
levels(v[1:2])
```

```@example hut
levels(v[2])
```
By tracking all classes in this way, MLJ
avoids common pain points around categorical data, such as evaluating
models on an evaluation set only to crash your code because classes appear
there which were not seen during training.


## Under the hood: CategoricalValue and CategoricalArray

In MLJ the atomic objects with `OrderedFactor` or `Multiclass`
scientific are `CategoricalValue`s, from the [CategoricalArrays.jl]
(https://juliadata.github.io/CategoricalArrays.jl/stable/) package.
In some sense `CategoricalValue`s are an implementation detail users
can ignore for the most part, as shown above. However, some users may
want some basic understanding of these types - and those implementing
MLJ's model interface for new alogorithms will have to understand
them, which we do so informally now.  For the complete API, see the
CategoricalArrays.jl
[documentation](https://juliadata.github.io/CategoricalArrays.jl/stable/)


To construct an `OrderedFactor` or `Multiclass` vector from raw
labels, one uses `categorical`:

```
@example hut
using CategoricalArrays # hide
v = categorical([:A, :B, :A, :A, :C])
typeof(v)
```

```
@example hut
scitype(v)
```

```
@example hut
v = categorical([:A, :B, :A, :A, :C], ordered=true)
scitype(v)
```

When you index a `CategoricalVector` you don't get a raw label, but
instead an instance of `CategoricalValue`. As explained above, this
value knows the complete pool of levels from vector from which it
came. Use `get(val)` to extract the raw label from a value `val`. 

Despite the distinction that exists between a value (element) and a
label, the two are the same, from the point of `==` and `in`:

```@julia
v[1] == :A # true
:A in v    # true
```


## Probablilistic predictions of categorical data
