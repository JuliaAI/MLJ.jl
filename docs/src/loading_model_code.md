# Loading Model Code

Once the name of a model, and the package providing that model, have
been identified (see [Model Search](@ref model_search)) one can either
import the model type interactively with `@iload`, as shown under
[Installation](@ref), or use `@load` as shown below. The `@load` macro
works from within a module, a package or a function, provided the
relevant package providing the MLJ interface has been added to your
package environment. It will attempt to load the model type into the
global namespace of the module in which `@load` is invoked (`Main` if
invoked at the REPL).

In general, the code providing core functionality for the model
(living in a package you should consult for documentation) may be
different from the package providing the MLJ interface. Since the core
package is a dependency of the interface package, only the interface
package needs to be added to your environment.

For instance, suppose you have activated a Julia package environment
`my_env` that you wish to use for your MLJ project; for example, you
have run:


```julia
using Pkg
Pkg.activate("my_env", shared=true)
```

Furthermore, suppose you want to use `DecisionTreeClassifier`,
provided by the
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl)
package. Then, to determine which package provides the MLJ interface
you call `load_path`:

```julia
julia> load_path("DecisionTreeClassifier", pkg="DecisionTree")
"MLJDecisionTreeInterface.DecisionTreeClassifier"
```

In this case, we see that the package required is
MLJDecisionTreeInterface.jl. If this package is not in `my_env` (do
`Pkg.status()` to check) you add it by running

```julia
julia> Pkg.add("MLJDecisionTreeInterface");
```

So long as `my_env` is the active environment, this action need never
be repeated (unless you run `Pkg.rm("MLJDecisionTreeInterface")`). You
are now ready to instantiate a decision tree classifier:

```julia
julia> Tree = @load DecisionTree pkg=DecisionTree
julia> tree = Tree()
```

which is equivalent to

```julia
julia> import MLJDecisionTreeInterface.DecisionTreeClassifier
julia> Tree = MLJDecisionTreeInterface.DecisionTreeClassifier
julia> tree = Tree()
```

*Tip.* The specification `pkg=...` above can be dropped for the many
models that are provided by only a single package.


## API

```@docs
load_path
@load
@iload
```
