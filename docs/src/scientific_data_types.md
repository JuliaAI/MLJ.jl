# Scientific Data Types

"Scientific" data types are formalized in MLJ with the addition of a new
Julia type hierarchy, rooted in an abstract type called `Found`. A
*scientific type* is defined to be any subtype of `Union{Missing, Found}`:

![](scitypes.png)

Note that `Multiclass{2}` has the alias `Binary`.

These types are used in MLJ purely for dispatch and are never
instantiated.

A given scientific type may have multiple machine type representations
(eg, `Float64` and `Float32` both represent `Continuous`). However,
MLJ adopts the simplifying (but not universally adopted) convention
that *the same machine data type cannot be used to represent multiple
scientific types*. We may accordingly express MLJ's convention on
scientific type representations using a function, `scitype`, which
associates to every Julia object a corresponding scientific type:

**Basic data convention** Scalar data with intended scientific
type `K` can be represented in MLJ by a Julia object `x` if and only if
`scitype(x) = K`.

So, one may check that `scitype(4.56) = Continuous` and
`scitype(4) = Count`. In particular, you cannot use
integers (which include booleans) to represent (nominal) multiclass
data. You should use a `CategoricalString` or `CategoricalValue` type
(automatic if your data appears in a `CategoricalVector`).

In fact, if `scitype(x) = K` then `scitype(y) = K` for *all* `y` with
`typeof(y)=typeof(x)` *unless*, `x` and `y` are instances of
`CategoricalValue` or `CategoricalString`. In this latter case, the
assertion continues to hold if and only if `x` and `y` have the same
`ordered` flag and the same number of levels.

Here, is the definition of `scitype` on scalar types:

````julia
nlevels(x) = length(levels(x.pool))

scitype(::Any) = Other # fallback

scitype(::Missing) = Missing
scitype(::Real) = Continuous
scitype(::Integer) = Count

scitype(c::CategoricalValue) =
    c.pool.ordered ? FiniteOrderedFactor{nlevels(c)} : Multiclass{nlevels(c)}

scitype(c::CategoricalString) =
    c.pool.ordered ? FiniteOrderedFactor{nlevels(c)} : Multiclass{nlevels(c)}
````

```@docs
union_scitypes
column_scitypes_as_tuple
```





