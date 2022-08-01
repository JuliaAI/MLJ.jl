# [List of Supported Models](@id model_list)

MLJ provides access to a wide variety of machine learning models.
We are always looking for
[help](https://github.com/alan-turing-institute/MLJ.jl/blob/master/CONTRIBUTING.md)
adding new models or testing existing ones.  Currently available
models are listed below; for the most up-to-date list, run `using MLJ;
models()`. 

Indications of "maturity" in the table below are approximate,
surjective, and possibly out-of-date. A decision to use or not use a
model in a critical application should be based on a user's
independent assessment.

* *experimental*: indicates the package is fairly new and/or is under
  active development; you can help by testing these packages and
  making them more robust,
* *low*: indicate a package that has reached a roughly stable form in
  terms of interface and which is unlikely to contain serious bugs. It
  may be missing some functionality found in similar packages. It
  has not benefited from a high level of use
* *medium*: indicates the package is fairly mature but may benefit
  from optimizations and/or extra features; you can help by suggesting
  either,
* *high*: indicates the package is very mature and functionalities are
  expected to have been fairly optimiser and tested.

| Package | Models | Maturity | Note
| ------- | ------ | -------- | ----
[BetaML.jl](https://github.com/sylvaticus/BetaML.jl) | DecisionTreeClassifier, DecisionTreeRegressor, GMMClusterer, KMeans, KMedoids, KernelPerceptronClassifier, MissingImputator, PegasosClassifier, PerceptronClassifier, RandomForestClassifier, RandomForestRegressor | medium |
[Clustering.jl](https://github.com/JuliaStats/Clustering.jl) | KMeans, KMedoids | high | †
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl) | DecisionTreeClassifier, DecisionTreeRegressor, AdaBoostStumpClassifier, RandomForestClassifier, RandomForestRegressor | high | 
[EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl) | EvoTreeRegressor, EvoTreeClassifier, EvoTreeCount, EvoTreeGaussian | medium | gradient boosting models
[GLM.jl](https://github.com/JuliaStats/GLM.jl) | LinearRegressor, LinearBinaryClassifier, LinearCountRegressor | medium | †
[LIBSVM.jl](https://github.com/mpastell/LIBSVM.jl) | LinearSVC, SVC, NuSVC, NuSVR, EpsilonSVR, OneClassSVM | high | also via ScikitLearn.jl
[LightGBM.jl](https://github.com/IQVIA-ML/LightGBM.jl) | LGBMClassifier, LGBMRegressor | high | 
[MLJFlux.jl](https://github.com/FluxML/MLJFlux.jl) | NeuralNetworkRegressor, NeuralNetworkClassifier, MultitargetNeuralNetworkRegressor, ImageClassifier | low |
[MLJLinearModels.jl](https://github.com/JuliaAI/MLJLinearModels.jl) | LinearRegressor, RidgeRegressor, LassoRegressor, ElasticNetRegressor, QuantileRegressor, HuberRegressor, RobustRegressor, LADRegressor, LogisticClassifier, MultinomialClassifier | medium |
[MLJModels.jl](https://github.com/JuliaAI/MLJModels.jl) (built-in) | StaticTransformer, FeatureSelector, FillImputer, UnivariateStandardizer, Standardizer, UnivariateBoxCoxTransformer, OneHotEncoder, ContinuousEncoder, ConstantRegressor, ConstantClassifier, BinaryThreshholdPredictor | medium |
[MLJText.jl](https://github.com/JuliaAI/MLJText.jl) | TfidfTransformer, BM25Transformer, CountTransformer | low |
[MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl) | LinearRegressor, MultitargetLinearRegressor, RidgeRegressor, MultitargetRidgeRegressor, PCA, KernelPCA, ICA, LDA, BayesianLDA, SubspaceLDA, BayesianSubspaceLDA, FactorAnalysis, PPCA | high | 
[NaiveBayes.jl](https://github.com/dfdx/NaiveBayes.jl) | GaussianNBClassifier, MultinomialNBClassifier, HybridNBClassifier | low |
[NearestNeighborModels.jl](https://github.com/JuliaAI/NearestNeighborModels.jl) | KNNClassifier, KNNRegressor, MultitargetKNNClassifier, MultitargetKNNRegressor | high |
[OneRule.jl](https://github.com/roland-KA/OneRule.jl) | OneRuleClassifier | experimental |
[OutlierDetectionNeighbors.jl](https://github.com/OutlierDetectionJL/OutlierDetectionNeighbors.jl) | ABODDetector, COFDetector, DNNDetector, KNNDetector, LOFDetector | medium | 
[OutlierDetectionNetworks.jl](https://github.com/OutlierDetectionJL/OutlierDetectionNetworks.jl) | AEDetector, DSADDetector, ESADDetector | medium | 
[OutlierDetectionPython.jl](https://github.com/OutlierDetectionJL/OutlierDetectionPython.jl) | ABODDetector, CBLOFDetector, COFDetector, COPODDetector, HBOSDetector, IForestDetector, KNNDetector, LMDDDetector, LOCIDetector, LODADetector, LOFDetector, MCDDetector, OCSVMDetector, PCADetector, RODDetector, SODDetector, SOSDetector | high | 
[ParallelKMeans.jl](https://github.com/PyDataBlog/ParallelKMeans.jl) | KMeans | experimental |
[PartialLeastSquaresRegressor.jl](https://github.com/lalvim/PartialLeastSquaresRegressor.jl) | PLSRegressor, KPLSRegressor | experimental |
[ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl) | ARDRegressor, AdaBoostClassifier, AdaBoostRegressor, AffinityPropagation, AgglomerativeClustering, BaggingClassifier, BaggingRegressor, BayesianLDA, BayesianQDA, BayesianRidgeRegressor, BernoulliNBClassifier, Birch, ComplementNBClassifier, DBSCAN, DummyClassifier, DummyRegressor, ElasticNetCVRegressor, ElasticNetRegressor, ExtraTreesClassifier, ExtraTreesRegressor, FeatureAgglomeration, GaussianNBClassifier, GaussianProcessClassifier, GaussianProcessRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HuberRegressor, KMeans, KNeighborsClassifier, KNeighborsRegressor, LarsCVRegressor, LarsRegressor, LassoCVRegressor, LassoLarsCVRegressor, LassoLarsICRegressor, LassoLarsRegressor, LassoRegressor, LinearRegressor, LogisticCVClassifier, LogisticClassifier, MeanShift, MiniBatchKMeans, MultiTaskElasticNetCVRegressor, MultiTaskElasticNetRegressor, MultiTaskLassoCVRegressor, MultiTaskLassoRegressor, MultinomialNBClassifier, OPTICS, OrthogonalMatchingPursuitCVRegressor, OrthogonalMatchingPursuitRegressor, PassiveAggressiveClassifier, PassiveAggressiveRegressor, PerceptronClassifier, ProbabilisticSGDClassifier, RANSACRegressor, RandomForestClassifier, RandomForestRegressor, RidgeCVClassifier, RidgeCVRegressor, RidgeClassifier, RidgeRegressor, SGDClassifier, SGDRegressor, SVMClassifier, SVMLClassifier, SVMLRegressor, SVMNuClassifier, SVMNuRegressor, SVMRegressor, SpectralClustering, TheilSenRegressor | high | †
[TSVD.jl](https://github.com/JuliaLinearAlgebra/TSVD.jl) | TSVDTransformer | high | 
[XGBoost.jl](https://github.com/dmlc/XGBoost.jl) | XGBoostRegressor, XGBoostClassifier, XGBoostCount | high |

**Note** (†): Some models are missing and assistance is welcome to
complete the interface. Post a message on the Julia #mlj Slack channel
if you would like to help, thanks!
