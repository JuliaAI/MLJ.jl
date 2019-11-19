<div align="center">
    <img src="https://alan-turing-institute.github.io/MLJTutorials/assets/infra/MLJLogo2.svg" alt="MLJ" width="200">
</div>

<h2 align="center">A Machine Learning Toolbox for Julia.
<p align="center">
  <a href="https://travis-ci.com/alan-turing-institute/MLJ.jl">
    <img src="https://travis-ci.com/alan-turing-institute/MLJ.jl.svg?branch=master"
         alt="Build Status">
  </a>
  <a href="https://coveralls.io/github/alan-turing-institute/MLJ.jl?branch=master">
    <img src="https://coveralls.io/repos/github/alan-turing-institute/MLJ.jl/badge.svg?branch=master"
         alt="Coverage">
  </a>
  <a href="https://slackinvite.julialang.org/">
    <img src="https://img.shields.io/badge/chat-on%20slack-yellow.svg"
         alt="#mlj">
  </a>
  <a href="https://alan-turing-institute.github.io/MLJ.jl/stable/">
    <img src="https://img.shields.io/badge/docs-stable-blue.svg"
         alt="Documentation">
  </a>
</p>
</h2>

MLJ is a machine learning framework for Julia aiming to provide a convenient way to use and combine a multitude of tools and models available in the Julia ML/Stats ecosystem.
MLJ is released under the MIT licensed and sponsored by the [Alan Turing Institute](https://www.turing.ac.uk/).

<br>
<p align="center">
  <a href="#using-mlj">Using MLJ</a> •
  <a href="#the-mlj-universe">MLJ Universe</a> •
  <a href="#contributing-to-mlj">Contributing</a> •
  <a href="#models-available">Available Models</a> •
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

It is a good idea to use a [separate environment](https://julialang.github.io/Pkg.jl/v1/environments/) for MLJ in order to avoid version clashes with other packages you may be using.
You can do so with

```julia
julia> using Pkg; Pkg.activate("My_MLJ_env", shared=true)
```

Installing MLJ is also done with the package manager:

```julia
julia> Pkg.add(["MLJ", "MLJModels"])
```

It is important to note that MLJ is essentially a big wrapper providing a unified access to _model providing packages_ and so you will also need to make sure these packages are available in your environment.
For instance, if you want to use a **Decision Tree Classifier**, you need to have [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) installed:

```julia
julia> Pkg.add("DecisionTree");
julia> using MLJ;
julia> @load DecisionTreeClassifier
```

For a list of models and their packages see the [table below](#models-available), or run

```julia
using MLJ
models()
```

We recommend you start with models marked as coming from _mature_ packages such as _DecisionTree_, _ScikitLearn_ or _XGBoost_.

#### Tutorials

The best place to get started with MLJ is to go the [MLJ Tutorials](https://alan-turing-institute.github.io/MLJTutorials/) website.
Each of the tutorial can be downloaded as a notebook or Julia script to facilitate experimentation with the packages.

You're also welcome to join the `#mlj` Julia slack channel to ask questions and make suggestions.

---

### The MLJ Universe

The MLJ universe is made out of several repositories some of which can be used independently of MLJ (indicated with a ⟂ symbol):

* (⟂) [MLJBase.jl](https://github.com/alan-turing-institute/MLJBase.jl) offers essential tools to load and interpret data, describe ML models and use metrics; it is the repository you should interface with if you wish to make your package accessible via MLJ,
* [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl) offers tools to compose, tune and evaluate models,
* [MLJModels.jl](https://github.com/alan-turing-institute/MLJModels.jl) contains interfaces to a number of important model-providing packages such as,  [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl), [ScikitLearn.jl](https://github.com/bensadeghi/ScikitLearn.jl) or [XGBoost.jl](https://github.com/dmlc/XGBoost.jl) as well as a few built-in transformations (one hot encoding, standardisation, ...), it also hosts the *model registry* which keeps track of all models accessible via MLJ,
* (⟂) [ScientificTypes.jl](https://github.com/alan-turing-institute/ScientificTypes.jl) a lightweight package to help specify the *interpretation* of data beyond how the data is currently encoded,
* (⟂) [MLJLinearModels.jl](https://github.com/alan-turing-institute/MLJLinearModels.jl) an experimental package for a wide range of penalised linear models such as Lasso, Elastic-Net, Robust regression, LAD regression, etc.
* [MLJFlux.jl](https://github.com/alan-turing-institute/MLJFlux.jl) an experimental package to use Flux within MLJ.

and maybe most importantly:

* [MLJTutorials](https://github.com/alan-turing-institute/MLJTutorials) which collects tutorials on how to use MLJ.

---

### Contributing to MLJ

MLJ is an ambitious project and we need all the help we can get!
There are multiple ways you can contribute; the table below helps indicate where you can help and what are the subjective requirements in terms of Julia and ML expertise.

Julia | ML         | What to do
----- | ---------- | ----------
=     | =          | use MLJ and give us feedback, help us write better tutorials, suggest missing features, test the less mature model packages
⭒     | =          | package to facilitate visualising results in MLJ
⭒     | ⭒          | add/improve data pre-processing tools
⭒     | ⭒          | add/improve interfaces to other model-providing packages
⭒     | ⭒          | functionalities for time series
⭒     | ⭒          | functionalities for systematic benchmarking of models
⭒     | ⭒          | functionalities for natural language processing (NLP)
⭒⭒    | =          | decrease the overhead incurred by MLJ
⭒⭒    | =          | improving support for sparse data
⭒⭒    | ⭒          | add parallelism and/or multithreading to MLJ (*there is an ongoing effort to interface with [Dagger.jl](https://github.com/JuliaParallel/Dagger.jl)*)
⭒     | ⭒⭒         | add  interface with probabilistic programming packages (*there is an ongoing effort to interface with [Soss.jl](https://github.com/cscherrer/Soss.jl)*)
⭒⭒    | ⭒⭒         | more sophisticated HP tuning (BO, Bandit, early stopping, ...) possibly as part of an external package(s), possibly integrating with Julia's optimisation and autodiff packages

If you're interested in one of these beyond the first one, please get in touch with either Anthony Blaom or Thibaut Lienart on Slack and we can further guide you.
Thank you!

You can also have a look at MLJ's [release notes](https://github.com/alan-turing-institute/MLJ.jl/releases) to get an idea for what's been happening recently.

---

### Models available

There is a wide range of models accessible via MLJ.
We are always looking for contributors to add new models or help us test existing ones.
The table below indicates the models that are accessible at present along with a subjective indication of how mature the underlying package is.

* *experimental*: indicates the package is fairly new and/or is under active development; you can help by testing these packages and making them more robust,
* *medium*: indicates the package is fairly mature but may benefit from optimisations and/or extra features; you can help by suggesting either,
* *high*: indicates the package is very mature and functionalities are expected to have been fairly optimised and tested.

| Package | Models | Maturity | Note
| ------- | ------ | -------- | ----
[Clustering.jl] | KMeans, KMedoids | high | †
[DecisionTree.jl] | DecisionTreeClassifier, DecisionTreeRegressor | high | †
[GaussianProcesses.jl] | GPClassifier | medium | †
[GLM.jl] | LinearRegressor, LinearBinaryClassifier, LinearCountRegressor | medium | †
[LIBSVM.jl] | LinearSVC, SVC, NuSVC, NuSVR, EpsilonSVR, OneClassSVM | high | also via ScikitLearn.jl
[MLJModels.jl] (builtins) | StaticTransformer, FeatureSelector, FillImputer, UnivariateStandardizer, Standardizer, UnivariateBoxCoxTransformer, OneHotEncoder, ConstantRegressor, ConstantClassifier | medium |
[MLJLinearModels.jl] | LinearRegressor, RidgeRegressor, LassoRegressor, ElasticNetRegressor, QuantileRegressor, HuberRegressor, RobustRegressor, LADRegressor, LogisticClassifier, MultinomialClassifier | experimental |
[MultivariateStats.jl] | RidgeRegressor, PCA, KernelPCA, ICA, LDA, BayesianLDA, SubspaceLDA, BayesianSubspaceLDA | high | †
[NaiveBayes.jl] | GaussianNBClassifier, MultinomialNBClassifier, HybridNBClassifier | medium |
[NearestNeighbors.jl] | KNNClassifier, KNNRegressor | high |
[ScikitLearn.jl] | SVMClassifier, SVMRegressor, SVMNuClassifier, SVMNuRegressor, SVMLClassifier, SVMLRegressor, ARDRegressor, BayesianRidgeRegressor, ElasticNetRegressor, ElasticNetCVRegressor, HuberRegressor, LarsRegressor, LarsCVRegressor, LassoRegressor, LassoCVRegressor, LassoLarsRegressor, LassoLarsCVRegressor, LassoLarsICRegressor, LinearRegressor, OrthogonalMatchingPursuitRegressor, OrthogonalMatchingPursuitCVRegressor, PassiveAggressiveRegressor, RidgeRegressor, RidgeCVRegressor, SGDRegressor, TheilSenRegressor, LogisticClassifier, LogisticCVClassifier, PerceptronClassifier, RidgeClassifier, RidgeCVClassifier, PassiveAggressiveClassifier, SGDClassifier, GaussianProcessRegressor, GaussianProcessClassifier, AdaBoostRegressor, AdaBoostClassifier, BaggingRegressor, BaggingClassifier, GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier, GaussianNB, MultinomialNB, ComplementNB, BayesianLDA, BayesianQDA | high | †
[XGBoost.jl] | XGBoostRegressor, XGBoostClassifier, XGBoostCount | high |

**Note** (†): some models are missing, your help is welcome to complete the interface. Get in touch with Thibaut Lienart on Slack if you want to help.

[Clustering.jl]: https://github.com/JuliaStats/Clustering.jl
[DecisionTree.jl]: https://github.com/bensadeghi/DecisionTree.jl
[GaussianProcesses.jl]: https://github.com/STOR-i/GaussianProcesses.jl
[GLM.jl]: https://github.com/JuliaStats/GLM.jl
[LIBSVM.jl]: https://github.com/mpastell/LIBSVM.jl
[MLJLinearModels.jl]: https://github.com/alan-turing-institute/MLJLinearModels.jl
[MLJModels.jl]: https://github.com/alan-turing-institute/MLJModels.jl
[MultivariateStats.jl]: https://github.com/JuliaStats/MultivariateStats.jl
[NaiveBayes.jl]: https://github.com/dfdx/NaiveBayes.jl
[NearestNeighbors.jl]: https://github.com/KristofferC/NearestNeighbors.jl
[ScikitLearn.jl]: https://github.com/cstjean/ScikitLearn.jl
[XGBoost.jl]: https://github.com/dmlc/XGBoost.jl

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
