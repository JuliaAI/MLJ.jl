<div align="center">
    <img src="material/MLJLogo2.svg" alt="MLJ" width="200">
</div>

<h2 align="center">A Machine Learning Framework for Julia
<p align="center">
  <a href="https://travis-ci.com/alan-turing-institute/MLJ.jl">
    <img src="https://travis-ci.com/alan-turing-institute/MLJ.jl.svg?branch=master"
         alt="Build Status">
  </a>
  <a href="https://slackinvite.julialang.org/">
    <img src="https://img.shields.io/badge/chat-on%20slack-yellow.svg"
         alt="#mlj">
  </a>
  <a href="https://alan-turing-institute.github.io/MLJ.jl/dev/">
    <img src="https://img.shields.io/badge/docs-stable-blue.svg"
         alt="Documentation">
  </a>
  </a>
  <!-- <a href="https://doi.org/10.5281/zenodo.3541506"> -->
  <!-- <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3541506.svg" -->
  <!--      alt="Cite MLJ"> -->
  <!-- </a> -->
  <a href="https://mybinder.org/v2/gh/alan-turing-institute/MLJ.jl/master?filepath=binder%2FMLJ_demo.ipynb">
  <img src="https://mybinder.org/badge_logo.svg"
       alt="Binder">
  </a>
  <a href="https://arxiv.org/abs/2007.12285">
    <img src="https://img.shields.io/badge/cite-arXiv-blue"
       alt="Cite MLJ">
  </a>

</p>
</h2>

**New to MLJ? Start [here](https://alan-turing-institute.github.io/MLJ.jl/dev/)**. 

**Wanting to integrate an existing machine learning model into the MLJ
framework? Start
[here](https://alan-turing-institute.github.io/MLJ.jl/dev/quick_start_guide_to_adding_models/)**.

The remaining information on this page will be of interest primarily
to developers interested in contributing to core packages in the MLJ
ecosystem, whose organization is described further below.

MLJ (Machine Learning in Julia) is a toolbox written in Julia
providing a common interface and meta-algorithms for selecting,
tuning, evaluating, composing and comparing machine over [150 machine
learning
models](https://alan-turing-institute.github.io/MLJ.jl/dev/list_of_supported_models/)
written in Julia and other languages.  MLJ is released under the MIT
licensed and sponsored by the [Alan Turing
Institute](https://www.turing.ac.uk/).

<br>
<p align="center">
<a href="#the-mlj-universe">MLJ Universe</a> &nbsp;•&nbsp; 
<a href="#known-issues">Known Issues</a> &nbsp;•&nbsp;
<a href="#customizing-behavior">Customizing Behavior</a> &nbsp;•&nbsp;
<a href="#citing-mlj">Citing MLJ</a> 
</p>
</br>


### The MLJ Universe

The functionality of MLJ is distributed over a number of repositories
illustrated in the dependency chart below.

<br>
<p align="center">
<a href="CONTRIBUTING.md">Contributing</a> &nbsp;•&nbsp; 
<a href="ORGANIZATION.md">Code Organization</a> &nbsp;•&nbsp;
<a href="ROADMAP.md">Road Map</a> 
</br>
<br>
  <a href="https://github.com/alan-turing-institute/MLJ">MLJ</a> &nbsp;•&nbsp;
  <a href="https://github.com/alan-turing-institute/MLJBase.jl">MLJBase</a> &nbsp;•&nbsp;
  <a href="https://github.com/alan-turing-institute/MLJModelInterface.jl">MLJModelInterface</a> &nbsp;•&nbsp;
  <a href="https://github.com/alan-turing-institute/MLJModels.jl">MLJModels</a> &nbsp;•&nbsp;
  <a href="https://github.com/alan-turing-institute/MLJTuning.jl">MLJTuning</a> &nbsp;•&nbsp;
  <a href="https://github.com/alan-turing-institute/MLJLinearModels.jl">MLJLinearModels</a> &nbsp;•&nbsp;
  <a href="https://github.com/alan-turing-institute/MLJFlux.jl">MLJFlux</a>
  <br>
  <a href="https://github.com/alan-turing-institute/MLJTutorials">MLJTutorials</a> &nbsp;•&nbsp;
  <a href="https://github.com/alan-turing-institute/MLJScientificTypes.jl">MLJScientificTypes</a> &nbsp;•&nbsp;
  <a href="https://github.com/alan-turing-institute/ScientificTypes.jl">ScientificTypes</a>
</p>
<p></p>
    <br>
<p></p>

<div align="center">
    <img src="material/MLJ_stack.svg" alt="Dependency Chart">
</div>

*Dependency chart for MLJ repositories. Repositories with dashed
connections do not currently exist but are planned/proposed.*


### Known Issues

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
  Pkg.develop(PackageSpec(url="https://github.com/tlienart/OpenSpecFun_jll.jl"))
```

#### Serialization for composite models with component models with custom serialization

See
[here](https://github.com/alan-turing-institute/MLJ.jl/issues/678). Workaround:
Instead of `XGBoost` models (the chief known case) use models from the
pure Julia package `EvoTrees`.


### Customizing behavior

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

For more on package management, see https://julialang.github.io/Pkg.jl/v1/ .



### Citing MLJ

<!-- <a href="https://doi.org/10.5281/zenodo.3541506"> -->
<!--   <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3541506.svg" -->
<!--        alt="Cite MLJ"> -->
<!-- </a> -->

<a href="https://arxiv.org/abs/2007.12285">
  <img src="https://img.shields.io/badge/cite-arXiv-blue"
       alt="Cite MLJ">
</a>

<!-- ```bibtex -->
<!-- @software{anthony_blaom_2019_3541506, -->
<!--   author       = {Anthony Blaom and -->
<!--                   Franz Kiraly and -->
<!--                   Thibaut Lienart and -->
<!--                   Sebastian Vollmer}, -->
<!--   title        = {alan-turing-institute/MLJ.jl: v0.5.3}, -->
<!--   month        = nov, -->
<!--   year         = 2019, -->
<!--   publisher    = {Zenodo}, -->
<!--   version      = {v0.5.3}, -->
<!--   doi          = {10.5281/zenodo.3541506}, -->
<!--   url          = {https://doi.org/10.5281/zenodo.3541506} -->
<!-- } -->
<!-- ``` -->

```bibtex
@misc{blaom2020mlj,
    title={MLJ: A Julia package for composable machine learning},
    author={Anthony D. Blaom and Franz Kiraly and Thibaut Lienart and Yiannis Simillides and Diego Arenas and Sebastian J. Vollmer},
    year={2020},
    eprint={2007.12285},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

#### Contributors

*Core design*: A. Blaom, F. Kiraly, S. Vollmer

*Active maintainers*: A. Blaom, T. Lienart, S. Okon

*Active collaborators*: D. Arenas, D. Buchaca, J. Hoffimann, S. Okon, J. Samaroo, S. Vollmer

*Past collaborators*: D. Aluthge, E. Barp, G. Bohner, M. K. Borregaard, V. Churavy, H. Devereux, M. Giordano, M. Innes, F. Kiraly, M. Nook, Z. Nugent, P. Oleśkiewicz, A. Shridar, Y. Simillides, A. Sengupta, A. Stechemesser.

#### License

MLJ is supported by the Alan Turing Institute and released under the MIT "Expat" License.
