# Road map

February 2020; updated, January 2021

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

The following road map is more big-picture; see also [this
list](https://github.com/alan-turing-institute/MLJ.jl/issues/673).


### Adding models

- ~~**Integrate deep learning using [Flux.jl](https://github.com/FluxML/Flux.jl.git) deep learning~~
  [Done](https://github.com/alan-turing-institute/MLJFlux.jl) but can
  improve experience by finishing
  [#139](https://github.com/alan-turing-institute/MLJ.jl/issues/139)
  and get better performance by implementing data front-end after [MLJBase
  #501](https://github.com/alan-turing-institute/MLJBase.jl/pull/501)
  is merged.

-  **†** ~~Probabilistic programming:
   [Turing.jl](https://github.com/TuringLang/Turing.jl),
   [Gen](https://github.com/probcomp/Gen),
   [Soss.jl](https://github.com/cscherrer/Soss.jl.git)
   [#157](https://github.com/alan-turing-institute/MLJ.jl/issues/157)
   [discourse
   thread](https://discourse.julialang.org/t/ppl-connection-to-mlj-jl/28736)~~
   [done](https://github.com/tlienart/SossMLJ.jl) but experimental and
   requires extension of probabilistic scoring functions to
   "distributions" that can only be sampled.

-   Feature engineering (python featuretools?, recursive feature
	elimination?)
	[#426](https://github.com/alan-turing-institute/MLJ.jl/issues/426) [MLJModels #314](https://github.com/alan-turing-institute/MLJModels.jl/issues/314)


### Enhancing core functionality

-   **†** Iterative model control [#139](https://github.com/alan-turing-institute/MLJ.jl/issues/139)

-   **†** Add more tuning
	strategies. [HyperOpt.jl](https://github.com/baggepinnen/Hyperopt.jl)
	integration? Particular focus on ~~random search  ([#37](https://github.com/alan-turing-institute/MLJ.jl/issues/37))~~ (done), Bayesian methods (starting with Gaussian Process methods, a la PyMC3) and a POC for AD-powered gradient descent. See
	[here](https://github.com/alan-turing-institute/MLJTuning.jl#what-is-provided-here) for complete wish-list. [#74](https://github.com/alan-turing-institute/MLJ.jl/issues/74) [#38](https://github.com/alan-turing-institute/MLJ.jl/issues/38) [#37](https://github.com/alan-turing-institute/MLJ.jl/issues/37)

-   Systematic benchmarking, probably modeled on
	[MLaut](https://arxiv.org/abs/1901.03678) [#69](https://github.com/alan-turing-institute/MLJ.jl/issues/74)

-   Give `EnsembleModel` more extendible API and extend beyond bagging
	(boosting, etc) and migrate to separate repository?
	[#363](https://github.com/alan-turing-institute/MLJ.jl/issues/363)

-   **†** Enhance complex model compostition, in introduce a canned
	stacking model
	([POC](https://alan-turing-institute.github.io/DataScienceTutorials.jl/getting-started/stacking/))
	Get rid of macros for creating pipelines and possibly implement
	target transforms as wrapper [MLJBase
	#594](https://github.com/alan-turing-institute/MLJ.jl/issues/594)


### Broadening scope

-   Spin-off a stand-alone measures (loss functions) package
	(currently
	[here](https://github.com/alan-turing-institute/MLJBase.jl/tree/master/src/measures)). Introduce
	measures for multi-targets [MLJBase
	#502](https://github.com/alan-turing-institute/MLJBase.jl/issues/502).

-   Add sparse data support and better support for NLP models; we
	could use [NaiveBayes.jl](https://github.com/dfdx/NaiveBayes.jl)
	as a POC (currently wrapped only for dense input) but the API
	needs finalizing first
	{#731](https://github.com/alan-turing-institute/MLJ.jl/issues/731).

-   POC for implementation of time series models classification
	[#303](https://github.com/alan-turing-institute/MLJ.jl/issues/303),
	[ScientificTypes #14](https://github.com/alan-turing-institute/ScientificTypes.jl/issues/14)

-   POC for time series forecasting; probably needs [MLJBase
	#502](https://github.com/alan-turing-institute/MLJBase.jl/issues/502)
	first, and someone to finish [PR on time series
	CV](https://github.com/alan-turing-institute/MLJBase.jl/pull/331).
	
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

-   Roll out data front-ends for all models after  [MLJBase
  #501](https://github.com/alan-turing-institute/MLJBase.jl/pull/501)
  is merged. 

-   Online learning support and distributed data
	[#60](https://github.com/alan-turing-institute/MLJ.jl/issues/60)

-   DAG scheduling for learning network training
	[#72](https://github.com/alan-turing-institute/MLJ.jl/issues/72)
	(multithreading first?)

-   Automated estimates of cpu/memory requirements
	[#71](https://github.com/alan-turing-institute/MLJ.jl/issues/71)

-   ~~Add multithreading to tuning [MLJTuning #15](https://github.com/alan-turing-institute/MLJTuning.jl/issues/15)~~ Done.
