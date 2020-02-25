# Road map

February 2020

Please visit [contributing guidelines](CONTRIBUTING.md) if interested
in contributing to MLJ.

### Guiding goals

-   **Usability, interoperability, extensibility, reproducibility,**
    and **code transparency**.

-   Offer state-of-art tools for model **composition** and model
    **optimization** (hyper-parameter tuning)

-   Avoid common **pain-points** of other frameworks:

    -   identifying all models that solve a given task

    -   routine operations requiring a lot of code

    -   passage from data source to algorithm-specific data format

    -   probabilistic predictions: inconsistent representations, lack
        of options for performance evaluation

-   Add some focus to julia machine learning software development more
    generally
	
### Priorities

Priorities are somewhat fluid, depending on funding offers and
available talent. Rough priorities for the core development team at
present are marked with **†** below. However, we are always keen to review
external contributions in any area.

## Future enhancements


### Adding models

- **†** Integrate deep learning using
  [Flux.jl](https://github.com/FluxML/Flux.jl.git) deep learning
  ([POC](https://github.com/alan-turing-institute/MLJFlux.jl))

-  **†** Probabilistic programming:
   [Turing.jl](https://github.com/TuringLang/Turing.jl),
   [Gen](https://github.com/probcomp/Gen),
   [Soss.jl](https://github.com/cscherrer/Soss.jl.git) ([POC](https://github.com/tlienart/SossMLJ.jl))
   [#157](https://github.com/alan-turing-institute/MLJ.jl/issues/157)
   [discourse thread](https://discourse.julialang.org/t/ppl-connection-to-mlj-jl/28736)

-   Feature engineering (python featuretools?, recursive feature
    elimination?)
    [#426](https://github.com/alan-turing-institute/MLJ.jl/issues/426)
	

### Enhancing core functionality

-   **†** Iterative model control [#139](https://github.com/alan-turing-institute/MLJ.jl/issues/139)

-   **†** Add more tuning
    strategies. [HyperOpt.jl](https://github.com/baggepinnen/Hyperopt.jl)
    integration. Particular focus on random search, Bayesian methods,
    and AD-powered gradient descent. See
    [here](https://github.com/alan-turing-institute/MLJTuning.jl#what-is-provided-here) for complete wish-list. [#74](https://github.com/alan-turing-institute/MLJ.jl/issues/74) [#38](https://github.com/alan-turing-institute/MLJ.jl/issues/38) [#37](https://github.com/alan-turing-institute/MLJ.jl/issues/37)

-   Systematic benchmarking, probably modeled on
    [MLaut](https://arxiv.org/abs/1901.03678) [#74](https://github.com/alan-turing-institute/MLJ.jl/issues/74)
	
-   Give `EnsembleModel` more extendible API and extend beyond bagging
    (boosting, etc) and migrate to separate repository?
    [#363](https://github.com/alan-turing-institute/MLJ.jl/issues/363)
	
-   **†** Enhance complex model compostition, in particular stacking
    ([POC](https://alan-turing-institute.github.io/MLJTutorials/getting-started/stacking/index.html))
    [#311](https://github.com/alan-turing-institute/MLJ.jl/issues/311)
    [#282](https://github.com/alan-turing-institute/MLJ.jl/issues/282)
	

### Broadening scope 

-   Spin-off a stand-alone measures (loss functions) package
    (currently
    [here](https://github.com/alan-turing-institute/MLJBase.jl/tree/master/src/measures))

-   Add sparse data support (NLP); could use
    [NaiveBayes.jl](https://github.com/dfdx/NaiveBayes.jl) as test
    case (currently wrapped only for dense input)

-   POC for implementation of time series models
    [#303](https://github.com/alan-turing-institute/MLJ.jl/issues/303),
    [ScientificTypes #14](https://github.com/alan-turing-institute/ScientificTypes.jl/issues/14)
	
-   Add tools or separate repository for visualization in MLJ. Only
    end-to-end visualization provided now is for two-parameter model
    tuning
    [#85](https://github.com/alan-turing-institute/MLJ.jl/issues/85)
    (closed)
    [#416](https://github.com/alan-turing-institute/MLJ.jl/issues/416)
    [#342](https://github.com/alan-turing-institute/MLJ.jl/issues/342)
	
-   Add more pre-processing tools, enhance MLJScientificType's
    `autotype` method.

### Scalability 

-   Online learning support and distributed data
    [#60](https://github.com/alan-turing-institute/MLJ.jl/issues/60)

-   DAG scheduling for learning network training
    [#72](https://github.com/alan-turing-institute/MLJ.jl/issues/72)
    (multithreading first?)

-   Automated estimates of cpu/memory requirements
    [#71](https://github.com/alan-turing-institute/MLJ.jl/issues/71)

-   Add multithreading to tuning [MLJTuning #15](https://github.com/alan-turing-institute/MLJTuning.jl/issues/15)
