# Conventions on representing data (DRAFT for discussion at [#86](https://github.com/alan-turing-institute/MLJ.jl/issues/86))

## What types should I use to represent my data?

It is useful to distinguish between *machine* data types on the one
hand (`Float64`, `Bool`, `String`, etc) , and *scientific* data types
on the other ("continuous", "binary", "ordered factor", etc). Learning
algorithms are naturally classified according to the scientific type
of the data they process, rather than by particular
representations. In applying MLJ algorithms, the MLJ user needs to
understand MLJ's conventions for representing scientific data types.

**Quick-and-dirty data type convention.** Continuous data in MLJ can be
represented by any `AbstractFloat` object, while finite-cardinality
discrete data can be represented by `CategoricalString` or
`CategoricalValue` objects (from the package CategoricalArrays.jl) providing
these have the appropriate `ordered` field value, according to whether the
data represents an ordered or unordered factor. "Count" type data
(ordered but unbounded) can be represented by any subtype of
`Int`.

The definitive convention is formulated in the next section.


## Formal scientific data types (kinds)

Scientific data types are formalized in MLJ with the addition
of new Julia type hierarchy rooted in the abstract type `Kind`:

````julia
Continuous <: Kind 
Discrete <: Kind
	Multiclass <: Discrete
	    Binary <: Multiclass
    OrderedFactor <: Discrete
	    OrderedFactorFinite <: OrderedFactor 
	    OrderedFactorInfinite <: OrderedFactor 	
Other <: Kind
````

Note that `Multiclass` is parameterized by the number of classes, and
`Binary = Multiclass{2}`. 

As an aside, note that these types are used in MLJ purely for
dispatch; they are never instantiated. 

A given scientific type may have multiple machine type representations
(eg, `Float64` and `Float32` both represent "continuous"). However, MLJ adopts
the simplifying (but not universally adopted) convention that *the same
machine data type cannote be used to represent multiple scientific
types*. We may accordingly express MLJ scientific type representation
using a function, `kind`, which associates to every Julia object a
corresponding subtype of `Kind`:

**Definitive type convention** Scalar data with intended
scientific type `K` can be represented in MLJ by a Julia object `x` if `kind(x) = K`.

So, for example, you may check that `kind(4.56) = Continuous` and
`kind(4) = OrderedFactorInfinite`. In particular, you cannot use
integers to represent (nominal) multiclass data!! You should use a
`CategoralString` or `CategoricalValue` type (automatic if your data
appears in a `CategoricaVector`).

In fact, if `kind(x) = K` then `kind(y) = K` for *all* `y` with
`typeof(y)=typeof(x)` *unless*, `x` and `y` are instances of
`CategorcalValue` or `CategoricalString`. In this latter case, the
assertion continues to hold if and only if `x` and `y` have the same
`ordered` flag and the same number of levels.

For the interested reader, here is MLJ's definition of the `kind` function:

````julia
kind(::Any) = Other
kind(::Real) = Continuous
kind(::Integer) = OrderedFactorInfinite
kind(c::CategoricalValue) =
    c.ordered ? OrderedFactorFinite : Multiclass{length(levels(c))}
kind(c::CategoricalString) =
    c.ordered ? OrderedFactorFinite : Multiclass{length(levels(c))}

````




