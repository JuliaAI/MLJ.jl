# Conventions on representing data (DRAFT for discussion at [#86](https://github.com/alan-turing-institute/MLJ.jl/issues/86))

## What types should I use to represent scalar data?

It is useful to distinguish between *machine* data types on the one
hand (`Float64`, `Bool`, `String`, etc) , and *scientific* data types
on the other ("continuous", "binary", "ordered factor", etc). Learning
algorithms are naturally classified according to the scientific type
of the data they process, rather than by particular
representations. In applying MLJ algorithms, the MLJ user needs to
understand MLJ's conventions for representing scientific data types.

**Quick-and-dirty data type convention.** Continuous data in MLJ can
be represented by any `AbstractFloat` object, while finite-cardinality
discrete data can be represented by `CategoricalString` or
`CategoricalValue` objects (from the package CategoricalArrays.jl),
providing these have the appropriate `ordered` field value, according
to whether the data represents an ordered or unordered factor. "Count"
type data (ordered but unbounded) can be represented by any subtype of
`Int`. 

One cannot use a subtype `I<:Integer` to represent finite-cardinality
data; such data can be encoded as `CategoricalValue{I}`.

The scientific type that a Julia object `x` may represent is given by
`scitype(x)`. So, for example, `scitype(4.5) = Continuous`. (Here
`Continuous` is a Julia type defined in MLJ used for dispatch. For
details see below.)


## Formal scientific data types (scitype) AN IMPLEMENTATION DETAIL

Scientific data types are formalized in MLJ with the addition of new
Julia type hierarchy rooted in an abstract type called `Found`, and we define
a *scientific type* to be any subtype of `Union{Missing, Found}`:

````julia
Continuous <: Found 
Discrete <: Found
	Multiclass <: Discrete
	    Binary <: Multiclass
    OrderedFactor <: Discrete
	    OrderedFactorFinite <: OrderedFactor 
	    OrderedFactorInfinite <: OrderedFactor
Other <: Found
````

Note that `Multiclass` and `OrderedFactorFinite` are parameterized by
the number of classes, and `Binary = Multiclass{2}`.

As an aside, note that these types are used in MLJ purely for
dispatch; they are never instantiated. 

A given scientific type may have multiple machine type representations
(eg, `Float64` and `Float32` both represent `Continuous`). However,
MLJ adopts the simplifying (but not universally adopted) convention
that *the same machine data type cannot be used to represent multiple
scientific types*. We may accordingly express MLJ's convention on
scientific type representations using a function, `scitype`, which
associates to every Julia object a corresponding subtype of `Found`:

**Definitive type convention** Scalar data with intended scientific
type `K` can be represented in MLJ by a Julia object `x` if
`scitype(x) = K`.

So, for example, you may check that `scitype(4.56) = Continuous` and
`scitype(4) = OrderedFactorInfinite`. In particular, you cannot use
integers (which include booleans) to represent (nominal) multiclass
data!! You should use a `CategoralString` or `CategoricalValue` type
(automatic if your data appears in a `CategoricaVector`).

In fact, if `scitype(x) = K` then `scitype(y) = K` for *all* `y` with
`typeof(y)=typeof(x)` *unless*, `x` and `y` are instances of
`CategorcalValue` or `CategoricalString`. In this latter case, the
assertion continues to hold if and only if `x` and `y` have the same
`ordered` flag and the same number of levels.

For the interested reader, here is MLJ's definition of the `scitype` function:

````julia
scitype(::Any) = Other
scitype(::Missing) = Missing
scitype(::Real) = Continuous
scitype(::Integer) = OrderedFactorInfinite
scitype(c::CategoricalValue) =
    c.ordered ? OrderedFactorFinite{length(levels(c))} : Multiclass{length(levels(c))}
scitype(c::CategoricalString) =
    c.ordered ? OrderedFactorFinite{length(levels(c))} : Multiclass{length(levels(c))}

````




