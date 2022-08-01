# Known Issues

Routine issues are posted
[here](https://github.com/alan-turing-institute/MLJ.jl/issues). Below
are some longer term issues and limitations.

#### ScikitLearn/MKL issue

For users of Mac OS using Julia 1.3 or higher, using ScikitLearn
models can lead to unexpected MKL errors due to an issue not related
to MLJ. See
[this Julia Discourse discussion](https://discourse.julialang.org/t/julia-1-3-1-4-on-macos-and-intel-mkl-error/36469/2) 
and
[this issue](https://github.com/JuliaPackaging/BinaryBuilder.jl/issues/700)
for context. 

A temporary workaround for this issue is to force the installation of
an older version of the `OpenSpecFun_jll` library. To install an
appropriate version, activate your MLJ environment and run

```julia
  using Pkg;
  Pkg.add(PackageSpec(url="https://github.com/tlienart/OpenSpecFun_jll.jl"))
```

#### Serialization for composite models with component models with custom serialization

See
[here](https://github.com/alan-turing-institute/MLJ.jl/issues/678). Workaround:
Instead of `XGBoost` models (the chief known case) use models from the
pure Julia package `EvoTrees`.

