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
  <a href="https://alan-turing-institute.github.io/MLJ.jl/stable/">
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

**New to MLJ? Start [here](https://alan-turing-institute.github.io/MLJ.jl/stable/)**. 

MLJ (Machine Learning in Julia) is a toolbox written in Julia
providing a common interface and meta-algorithms for selecting,
tuning, evaluating, composing and comparing machine learning models written in Julia and other languages.  MLJ is released
under the MIT licensed and sponsored by the [Alan Turing
Institute](https://www.turing.ac.uk/).

<br>
<p align="center">
<a href="#the-mlj-universe">MLJ Universe</a> &nbsp;•&nbsp; 
<a href="#list-of-wrapped-models">List of Wrapped Models</a> &nbsp;•&nbsp;
<a href="#known-issues">Known Issues</a> &nbsp;•&nbsp;
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

### List of Wrapped Models

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
[EvoTrees.jl] | EvoTreeRegressor, EvoTreeClassifier, EvoTreeCount, EvoTreeGaussian | medium | gradient boosting models
[GLM.jl] | LinearRegressor, LinearBinaryClassifier, LinearCountRegressor | medium | †
[LIBSVM.jl] | LinearSVC, SVC, NuSVC, NuSVR, EpsilonSVR, OneClassSVM | high | also via ScikitLearn.jl
[LightGBM.jl] | LightGBMClassifier, LightGBMRegressor | high | 
[MLJFlux.jl] | NeuralNetworkRegressor, NeuralNetworkClassifier, MultitargetNeuralNetworkRegressor, ImageClassifier | experimental |
[MLJLinearModels.jl] | LinearRegressor, RidgeRegressor, LassoRegressor, ElasticNetRegressor, QuantileRegressor, HuberRegressor, RobustRegressor, LADRegressor, LogisticClassifier, MultinomialClassifier | experimental |
[MLJModels.jl] (builtins) | StaticTransformer, FeatureSelector, FillImputer, UnivariateStandardizer, Standardizer, UnivariateBoxCoxTransformer, OneHotEncoder, ContinuousEncoder, ConstantRegressor, ConstantClassifier | medium |
[MultivariateStats.jl] | RidgeRegressor, PCA, KernelPCA, ICA, LDA, BayesianLDA, SubspaceLDA, BayesianSubspaceLDA | high | †
[NaiveBayes.jl] | GaussianNBClassifier, MultinomialNBClassifier, HybridNBClassifier | experimental |
[NearestNeighbors.jl] | KNNClassifier, KNNRegressor | high |
[ParallelKMeans.jl] | KMeans | experimental | 
[ScikitLearn.jl] | ARDRegressor, AdaBoostClassifier, AdaBoostRegressor, AffinityPropagation, AgglomerativeClustering, BaggingClassifier, BaggingRegressor, BayesianLDA, BayesianQDA, BayesianRidgeRegressor, BernoulliNBClassifier, Birch, ComplementNBClassifier, DBSCAN, DummyClassifier, DummyRegressor, ElasticNetCVRegressor, ElasticNetRegressor, ExtraTreesClassifier, ExtraTreesRegressor, FeatureAgglomeration, GaussianNBClassifier, GaussianProcessClassifier, GaussianProcessRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HuberRegressor, KMeans, KNeighborsClassifier, KNeighborsRegressor, LarsCVRegressor, LarsRegressor, LassoCVRegressor, LassoLarsCVRegressor, LassoLarsICRegressor, LassoLarsRegressor, LassoRegressor, LinearRegressor, LogisticCVClassifier, LogisticClassifier, MeanShift, MiniBatchKMeans, MultiTaskElasticNetCVRegressor, MultiTaskElasticNetRegressor, MultiTaskLassoCVRegressor, MultiTaskLassoRegressor, MultinomialNBClassifier, OPTICS, OrthogonalMatchingPursuitCVRegressor, OrthogonalMatchingPursuitRegressor, PassiveAggressiveClassifier, PassiveAggressiveRegressor, PerceptronClassifier, ProbabilisticSGDClassifier, RANSACRegressor, RandomForestClassifier, RandomForestRegressor, RidgeCVClassifier, RidgeCVRegressor, RidgeClassifier, RidgeRegressor, SGDClassifier, SGDRegressor, SVMClassifier, SVMLClassifier, SVMLRegressor, SVMNuClassifier, SVMNuRegressor, SVMRegressor, SpectralClustering, TheilSenRegressor | high | †
[XGBoost.jl] | XGBoostRegressor, XGBoostClassifier, XGBoostCount | high |

**Note** (†): some models are missing, your help is welcome to complete the interface. Get in touch with Thibaut Lienart on Slack if you would like to help, thanks!

[Clustering.jl]: https://github.com/JuliaStats/Clustering.jl
[DecisionTree.jl]: https://github.com/bensadeghi/DecisionTree.jl
[EvoTrees.jl]: https://github.com/Evovest/EvoTrees.jl
[GaussianProcesses.jl]: https://github.com/STOR-i/GaussianProcesses.jl
[GLM.jl]: https://github.com/JuliaStats/GLM.jl
[LIBSVM.jl]: https://github.com/mpastell/LIBSVM.jl
[LightGBM.jl]: https://github.com/IQVIA-ML/LightGBM.jl
[MLJ.jl]: https://github.com/alan-turing-institute/MLJ.jl
[MLJTutorials.jl]: https://github.com/alan-turing-institute/MLJTutorials.jl
[MLJBase.jl]: https://github.com/alan-turing-institute/MLJBase.jl
[MLJModelInterface.jl]: https://github.com/alan-turing-institute/MLJModelInterface.jl
[MLJModels.jl]: https://github.com/alan-turing-institute/MLJModels.jl
[MLJTuning.jl]: https://github.com/alan-turing-institute/MLJTuning.jl
[MLJLinearModels.jl]: https://github.com/alan-turing-institute/MLJLinearModels.jl
[MLJFlux.jl]: https://github.com/alan-turing-institute/MLJFlux.jl
[MLJScientificTypes.jl]: https://github.com/alan-turing-institute/MLJScientificTypes.jl
[ParallelKMeans.jl]: https://github.com/PyDataBlog/ParallelKMeans.jl
[ScientificTypes.jl]: https://github.com/alan-turing-institute/ScientificTypes.jl
[MultivariateStats.jl]: https://github.com/JuliaStats/MultivariateStats.jl
[NaiveBayes.jl]: https://github.com/dfdx/NaiveBayes.jl
[NearestNeighbors.jl]: https://github.com/KristofferC/NearestNeighbors.jl
[ScikitLearn.jl]: https://github.com/cstjean/ScikitLearn.jl
[XGBoost.jl]: https://github.com/dmlc/XGBoost.jl


### Known Issues

For users of Mac OS using Julia 1.3 or higher, using ScikitLearn
models can lead to unexpected MKL errors due to an issue not related
to MLJ. See
[this Julia Discourse discussion](https://discourse.julialang.org/t/julia-1-3-1-4-on-macos-and-intel-mkl-error/36469/2) 
and
[this issue](https://github.com/JuliaPackaging/BinaryBuilder.jl/issues/700)
for context. 

A temporary workaround for this issue is to force the installation of
an older version of the `OpenSpecFun_jll` library. To install an
appropriate version, activate your MLJ environment and run `using Pkg;
Pkg.develop(PackageSpec(url="https://github.com/tlienart/OpenSpecFun_jll.jl"))`.


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

```bitex
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
