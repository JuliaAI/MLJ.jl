<div align="center">
    <img src="https://alan-turing-institute.github.io/MLJTutorials/assets/infra/MLJLogo2.svg" alt="MLJ" width="200">
</div>

<h2 align="center">A Machine Learning Toolbox for Julia.
<p align="center">
  <a href="https://travis-ci.com/alan-turing-institute/MLJ.jl">
    <img src="https://travis-ci.com/alan-turing-institute/MLJ.jl.svg?branch=master"
         alt="Build Status">
  </a>
  <a href="https://slackinvite.julialang.org/">
    <img src="https://img.shields.io/badge/chat-on%20slack-yellow.svg"
         alt="#mlj">
  </a>
  <a href="https://alan-turing-institute.github.io/MLJ.jl/stable/">
    <img src="https://img.shields.io/badge/docs-stable-blue.svg"
         alt="Documentation">
  </a>
  </a>
  <a href="https://doi.org/10.5281/zenodo.3541506">
  <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3541506.svg"
       alt="Cite MLJ">
  </a>
</p>
</h2>

MLJ is a machine learning framework for Julia aiming to provide a convenient way to use and combine a multitude of tools and models available in the Julia ML/Stats ecosystem.
MLJ is released under the MIT licensed and sponsored by the [Alan Turing Institute](https://www.turing.ac.uk/).

<br>
<p align="center">
  <a href="#using-mlj">Using MLJ</a> •
  <a href="#available-models">Models Available</a> •
  <a href="#the-mlj-universe">MLJ Universe</a> •
  <a href="CONTRIBUTING.md">Contributing</a> •
  <a href="https://github.com/alan-turing-institute/MLJ.jl/blob/master/docs/src/mlj_cheatsheet.md">MLJ Cheatsheet</a> •
  <a href="#citing-mlj">Citing MLJ</a>
</p>

### Key goals

* Offer a consistent way to use, compose and tune machine learning models in Julia,
* Promote the improvement of the Julia ML/Stats ecosystem by making it easier to use models from a wide range of packages,
* Unlock performance gains by exploiting Julia's support for parallelism, automatic differentiation, GPU, optimisation etc.

### Key features

* Data agnostic, train models on any data supported by the [Tables.jl](https://github.com/JuliaData/Tables.jl) interface,
* Extensive support for model composition (*pipelines* and *learning networks*),
* Convenient syntax to tune and evaluate (composite) models,
* Consistent interface to handle probabilistic predictions.

---

### Using MLJ

Initially it is recommended that MLJ and associated packages be
installed in a new
[environment](https://julialang.github.io/Pkg.jl/v1/environments/) to
avoid package conflicts. You can do this with

```julia
julia> using Pkg; Pkg.activate("My_MLJ_env", shared=true)
```

Installing MLJ is also done with the package manager:

```julia
julia> Pkg.add(["MLJ", "MLJModels"])
```

It is important to note that MLJ is essentially a big wrapper
providing a unified access to _model providing packages_ and so you
will also need to make sure these packages are available in your
environment.  For instance, if you want to use a **Decision Tree
Classifier**, you need to have
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl)
installed:

```julia
julia> Pkg.add("DecisionTree");
julia> using MLJ;
julia> @load DecisionTreeClassifier
```

For a list of models and their packages see the [table below](#available-models), or run

```julia
using MLJ
models()
```

We recommend you start with models marked as coming from _mature_ packages such as _DecisionTree_, _ScikitLearn_ or _XGBoost_.


#### Tutorials

The best place to get started with MLJ is to go the [MLJ
Tutorials](https://alan-turing-institute.github.io/MLJTutorials/)
website.  Each of the tutorial can be downloaded as a notebook or
Julia script to facilitate experimentation with the packages. For more
comprehensive documentation, see the user
[manual](https://alan-turing-institute.github.io/MLJ.jl/stable/).

You're also welcome to join the `#mlj` Julia slack channel to ask
questions and make suggestions.

---

### Available Models 

MLJ provides access to to a wide variety of machine learning models.
We are always looking for [help](CONTRIBUTING.md) adding new models or
test existing ones.  Currently available models are listed below; for
the most up-to-date list, run `using MLJ; models()`.

* *experimental*: indicates the package is fairly new and/or is under active development; you can help by testing these packages and making them more robust,
* *medium*: indicates the package is fairly mature but may benefit from optimisations and/or extra features; you can help by suggesting either,
* *high*: indicates the package is very mature and functionalities are expected to have been fairly optimised and tested.

| Package | Models | Maturity | Note
| ------- | ------ | -------- | ----
[Clustering.jl] | KMeans, KMedoids | high | †
[DecisionTree.jl] | DecisionTreeClassifier, DecisionTreeRegressor, AdaBoostStumpClassifier | high | †
[EvoTrees.jl] | EvoTreeRegressor, EvoTreeClassifier, EvoTreeCount, EvoTreeGaussian | low | gradient boosting models
[GLM.jl] | LinearRegressor, LinearBinaryClassifier, LinearCountRegressor | medium | †
[LIBSVM.jl] | LinearSVC, SVC, NuSVC, NuSVR, EpsilonSVR, OneClassSVM | high | also via ScikitLearn.jl
[MLJModels.jl] (builtins) | StaticTransformer, FeatureSelector, FillImputer, UnivariateStandardizer, Standardizer, UnivariateBoxCoxTransformer, OneHotEncoder, ConstantRegressor, ConstantClassifier | medium |
[MLJLinearModels.jl] | LinearRegressor, RidgeRegressor, LassoRegressor, ElasticNetRegressor, QuantileRegressor, HuberRegressor, RobustRegressor, LADRegressor, LogisticClassifier, MultinomialClassifier | experimental |
[MultivariateStats.jl] | RidgeRegressor, PCA, KernelPCA, ICA, LDA, BayesianLDA, SubspaceLDA, BayesianSubspaceLDA | high | †
[NaiveBayes.jl] | GaussianNBClassifier, MultinomialNBClassifier, HybridNBClassifier | low |
[NearestNeighbors.jl] | KNNClassifier, KNNRegressor | high |
[ScikitLearn.jl] | ARDRegressor, AdaBoostClassifier, AdaBoostRegressor, AffinityPropagation, AgglomerativeClustering, BaggingClassifier, BaggingRegressor, BayesianLDA, BayesianQDA, BayesianRidgeRegressor, BernoulliNBClassifier, Birch, ComplementNBClassifier, DBSCAN, DummyClassifier, DummyRegressor, ElasticNetCVRegressor, ElasticNetRegressor, ExtraTreesClassifier, ExtraTreesRegressor, FeatureAgglomeration, GaussianNBClassifier, GaussianProcessClassifier, GaussianProcessRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HuberRegressor, KMeans, KNeighborsClassifier, KNeighborsRegressor, LarsCVRegressor, LarsRegressor, LassoCVRegressor, LassoLarsCVRegressor, LassoLarsICRegressor, LassoLarsRegressor, LassoRegressor, LinearRegressor, LogisticCVClassifier, LogisticClassifier, MeanShift, MiniBatchKMeans, MultiTaskElasticNetCVRegressor, MultiTaskElasticNetRegressor, MultiTaskLassoCVRegressor, MultiTaskLassoRegressor, MultinomialNBClassifier, OPTICS, OrthogonalMatchingPursuitCVRegressor, OrthogonalMatchingPursuitRegressor, PassiveAggressiveClassifier, PassiveAggressiveRegressor, PerceptronClassifier, ProbabilisticSGDClassifier, RANSACRegressor, RandomForestClassifier, RandomForestRegressor, RidgeCVClassifier, RidgeCVRegressor, RidgeClassifier, RidgeRegressor, SGDClassifier, SGDRegressor, SVMClassifier, SVMLClassifier, SVMLRegressor, SVMNuClassifier, SVMNuRegressor, SVMRegressor, SpectralClustering, TheilSenRegressor | high | †
[XGBoost.jl] | XGBoostRegressor, XGBoostClassifier, XGBoostCount | high |

**Note** (†): some models are missing, your help is welcome to complete the interface. Get in touch with Thibaut Lienart on Slack if you would like to help, thanks!

[Clustering.jl]: https://github.com/JuliaStats/Clustering.jl
[DecisionTree.jl]: https://github.com/bensadeghi/DecisionTree.jl
[EvoTrees.jl]: https://github.com/Evovest/EvoTrees.jl
[GaussianProcesses.jl]: https://github.com/STOR-i/GaussianProcesses.jl
[GLM.jl]: https://github.com/JuliaStats/GLM.jl
[LIBSVM.jl]: https://github.com/mpastell/LIBSVM.jl
[MLJ.jl]: https://github.com/alan-turing-institute/MLJ.jl
[MLJTutorials.jl]: https://github.com/alan-turing-institute/MLJTutorials.jl
[MLJBase.jl]: https://github.com/alan-turing-institute/MLJBase.jl
[MLJModelInterface.jl]: https://github.com/alan-turing-institute/MLJModelInterface.jl
[MLJModels.jl]: https://github.com/alan-turing-institute/MLJModels.jl
[MLJTuning.jl]: https://github.com/alan-turing-institute/MLJTuning.jl
[MLJLinearModels.jl]: https://github.com/alan-turing-institute/MLJLinearModels.jl
[MLJFlux.jl]: https://github.com/alan-turing-institute/MLJFlux.jl
[MLJScientificTypes.jl]: https://github.com/alan-turing-institute/MLJScientificTypes.jl
[ScientificTypes.jl]: https://github.com/alan-turing-institute/ScientificTypes.jl
[MultivariateStats.jl]: https://github.com/JuliaStats/MultivariateStats.jl
[NaiveBayes.jl]: https://github.com/dfdx/NaiveBayes.jl
[NearestNeighbors.jl]: https://github.com/KristofferC/NearestNeighbors.jl
[ScikitLearn.jl]: https://github.com/cstjean/ScikitLearn.jl
[XGBoost.jl]: https://github.com/dmlc/XGBoost.jl

---

### The MLJ Universe

The functionality of MLJ is distributed over a number of repositories
illustrated in the dependency chart below. Click on the appropriate
link for further information:

<br>
<p align="center">
  <a href="ORGANIZATION.md">Code Organization</a> &nbsp;•&nbsp;
  <a href="ROADMAP.md">Road Map</a>  &nbsp;•&nbsp;
  <a href="CONTRIBUTING.md">Contributing</a>
</p>
<p align="center">
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

---

### Citing MLJ

<a href="https://doi.org/10.5281/zenodo.3541506">
  <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3541506.svg"
       alt="Cite MLJ">
</a>

```bibtex
@software{anthony_blaom_2019_3541506,
  author       = {Anthony Blaom and
                  Franz Kiraly and
                  Thibaut Lienart and
                  Sebastian Vollmer},
  title        = {alan-turing-institute/MLJ.jl: v0.5.3},
  month        = nov,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {v0.5.3},
  doi          = {10.5281/zenodo.3541506},
  url          = {https://doi.org/10.5281/zenodo.3541506}
}
```

#### Contributors

*Core design*: A. Blaom, F. Kiraly, S. Vollmer

*Active maintainers*: A. Blaom, T. Lienart

*Active collaborators*: D. Arenas, D. Buchaca, J. Hoffimann, S. Okon, J. Samaroo, S. Vollmer

*Past collaborators*: D. Aluthge, E. Barp, G. Bohner, M. K. Borregaard, V. Churavy, H. Devereux, M. Giordano, M. Innes, F. Kiraly, M. Nook, Z. Nugent, P. Oleśkiewicz, A. Shridar, Y. Simillides, A. Sengupta, A. Stechemesser.

#### License

MLJ is supported by the Alan Turing Institute and released under the MIT "Expat" License.
