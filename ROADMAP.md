Road map
========

February 2020

Please visit [contributing guidelines](CONTRIBUTING.md) if interested
in contributing to MLJ.


Adding models
-------------

- Deep learning 
  ([Flux.jl](https://github.com/FluxML/Flux.jl.git) deep learning ([POC](https://github.com/alan-turing-institute/MLJFlux.jl) complete)

-  Probabilistic programming
   ([Turing.jl](https://github.com/TuringLang/Turing.jl),
   [Gen](https://github.com/probcomp/Gen),
   [Soss.jl](https://github.com/cscherrer/Soss.jl.git) -
   [POC](https://github.com/tlienart/SossMLJ.jl) complete)
   [#157](https://github.com/alan-turing-institute/MLJ.jl/issues/157)
   [discourse](https://discourse.julialang.org/t/ppl-connection-to-mlj-jl/28736)

-   Feature engineering (python featuretools?, recursive feature
    elimination?)
    [#426](https://github.com/alan-turing-institute/MLJ.jl/issues/426)
	

Enhancing core functionality
-----------------------------

-   Iterative model control [#139](https://github.com/alan-turing-institute/MLJ.jl/issues/139)

-   Add more tuning
    strategies. [HyperOpt.jl](https://github.com/baggepinnen/Hyperopt.jl)
    integration. Particular focus on random search, Bayesian methods,
    and AD-powered gradient descent. See
    [here](https://github.com/alan-turing-institute/MLJTuning.jl#what-is-provided-here) for complete wish-list. [#74](https://github.com/alan-turing-institute/MLJ.jl/issues/74), [#38](https://github.com/alan-turing-institute/MLJ.jl/issues/38), [#37](https://github.com/alan-turing-institute/MLJ.jl/issues/37)

-   Systematic benchmarking, probably modeled on
    [MLaut](https://arxiv.org/abs/1901.03678) [#74](https://github.com/alan-turing-institute/MLJ.jl/issues/74)
	
-   Give `EnsembleModel` more extendible API and extend beyond bagging
    (boosting, etc) and migrate to separate repository?
    [#363](https://github.com/alan-turing-institute/MLJ.jl/issues/363)
	

Broadening scope 
----------------

-   Spin-off a stand-alone measures (loss functions) package
    (currently
    [here](https://github.com/alan-turing-institute/MLJBase.jl/tree/master/src/measures))

-   Add sparse data support (NLP); could use
    [NaiveBayes.jl](https://github.com/dfdx/NaiveBayes.jl) as test
    case (currently wrapped only for dense input)

-   POC for implementation of time series models
    [#303](https://github.com/alan-turing-institute/MLJ.jl/issues/303),
    ScientificTypes
    [#14](https://github.com/alan-turing-institute/ScientificTypes.jl/issues/14)
	
-   Add tools or separate repository for visualization in MLJ. Only
    end-to-end visualization provided now is for two-parameter model
    tuning (see [closed #85](https://github.com/alan-turing-institute/MLJ.jl/issues/85))
    [#416](https://github.com/alan-turing-institute/MLJ.jl/issues/416)
    [#342](https://github.com/alan-turing-institute/MLJ.jl/issues/342)
	
-   Add more pre-processing tools, enhance MLJScientificType's
    `autotype` method.

Scalability 
-----------

-   Online learning support and distributed data
    [#60](https://github.com/alan-turing-institute/MLJ.jl/issues/60)

-   DAG scheduling [#72](https://github.com/alan-turing-institute/MLJ.jl/issues/72)

-   Automated estimates of cpu/memory requirements
    [#71](https://github.com/alan-turing-institute/MLJ.jl/issues/71)




