# Customizing Behavior

To customize behaviour of MLJ you will need to clone the relevant
component package (e.g., MLJBase.jl) - or a fork thereof - and modify
your local julia environment to use your local clone in place of the
official release. For example, you might proceed something like this:

```julia
using Pkg
Pkg.activate("my_MLJ_enf", shared=true)
Pkg.develop("path/to/my/local/MLJBase")
```

To test your local clone, do

```julia
Pkg.test("MLJBase")
```

For more on package management, see [here](https://julialang.github.io/Pkg.jl/v1/).

