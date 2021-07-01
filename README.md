<div align="center">
    <img src="material/MLJLogo2.svg" alt="MLJ" width="200">
</div>

<h2 align="center">A Machine Learning Framework for Julia
<p align="center">
  <a href="https://github.com/alan-turing-institute/MLJ.jl/actions">
    <img src="https://github.com/alan-turing-institute/MLJ.jl/workflows/CI/badge.svg"
         alt="Build Status">
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
  <a href="BIBLIOGRAPHY.md">
    <img src="https://img.shields.io/badge/cite-bibtex-blue"
       alt="bibtex">
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
tuning, evaluating, composing and comparing over [160 machine
learning
models](https://alan-turing-institute.github.io/MLJ.jl/dev/list_of_supported_models/)
written in Julia and other languages.  

MLJ was initially created as a Tools,
Practices and Systems project at the [Alan Turing
Institute](https://www.turing.ac.uk/) in 2019. Current funding is
provided by a [New Zealand Strategic Science Investment
Fund](https://www.mbie.govt.nz/science-and-technology/science-and-innovation/funding-information-and-opportunities/investment-funds/strategic-science-investment-fund/ssif-funded-programmes/university-of-auckland/).

MLJ is released under the MIT license, and has been developed with the
support of the following organizations:

<div align="center">
    <img src="material/Turing_logo.png" width = 100/>
    <img src="material/UoA_logo.png" width = 100/>
    <img src="material/IQVIA_logo.png" width = 100/>
    <img src="material/warwick.png" width = 100/>
    <img src="material/julia.png" width = 100/>
</div>

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
illustrated in the dependency chart below. These repositories live at the [JuliaAI umbrella](https://github.com/JuliaAI).

<div align="center">
    <img src="material/MLJ_stack.svg" alt="Dependency Chart">
</div>

*Dependency chart for MLJ repositories. Repositories with dashed
connections do not currently exist but are planned/proposed.*

<br>
<p align="center">
<a href="CONTRIBUTING.md">Contributing</a> &nbsp;•&nbsp; 
<a href="ORGANIZATION.md">Code Organization</a> &nbsp;•&nbsp;
<a href="ROADMAP.md">Road Map</a> 
</br>





#### Contributors

*Core design*: A. Blaom, F. Kiraly, S. Vollmer

*Active maintainers*: A. Blaom, T. Lienart, S. Okon

*Active collaborators*: D. Arenas, D. Buchaca, J. Hoffimann, S. Okon, J. Samaroo, S. Vollmer

*Past collaborators*: D. Aluthge, E. Barp, G. Bohner, M. K. Borregaard, V. Churavy, H. Devereux, M. Giordano, M. Innes, F. Kiraly, M. Nook, Z. Nugent, P. Oleśkiewicz, A. Shridar, Y. Simillides, A. Sengupta, A. Stechemesser.

#### License

MLJ is supported by the Alan Turing Institute and released under the MIT "Expat" License.
´
